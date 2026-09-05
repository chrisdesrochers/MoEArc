//! The host expert kernels against the device's, on real weights and real shapes.
//!
//! Skipped unless both are set:
//!
//! ```text
//! MOEARC_TEST_GPU=1
//! MOEARC_QWEN3MOE_MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf
//! ```
//!
//! # Why this gate and not a unit test
//!
//! `host_experts`' own unit tests check the two quantisers against
//! `moearc_kernels::reference`, which is the executable specification and is itself checked
//! against llama.cpp's output. That is a real check and it is not sufficient: it uses synthetic
//! blocks, and it compares the host against a *host* implementation. The device kernels are the
//! ones this engine's token ids were established with, so the question that matters is how far
//! the CPU path lands from **them**, on the weights the model actually holds.
//!
//! 🔴 The two are not expected to agree bit for bit and it would be suspicious if they did. The
//! device dequantises a block and reduces in a f32 tree; the host folds the K-quant block
//! minimum out of its inner loop and reduces a row in order. Same value in exact arithmetic,
//! different rounding. What this file establishes is the size of that difference — recorded in
//! [`TOL`] — so that a future change which makes it larger is a test failure rather than a
//! plausible paragraph of prose.

#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::Arc;

use moearc_engine::host_experts::{
    BankSpec, BlockSpec, Geometry, HostExecutor, HostPolicy, expert_ffn, group_sums,
};
use moearc_engine::moe::Residency;
use moearc_engine::session::{Session, SessionOptions, StopConditions};
use moearc_kernels::{Context, QuantType};
use moearc_model::ModelInfo;
use moearc_model::tensors::{ExpertBank, MappedModel};

/// The largest relative disagreement tolerated between the host and device expert paths.
///
/// Measured, not chosen: see the printout each test emits. The scale is the device output's own
/// magnitude, floored at 1.0 so that a near-zero channel does not report an enormous relative
/// error from an absolute one that does not matter.
const TOL: f32 = 1e-5;

fn model_path() -> Option<PathBuf> {
    if std::env::var("MOEARC_TEST_GPU").ok().as_deref() != Some("1") {
        return None;
    }
    std::env::var("MOEARC_QWEN3MOE_MODEL").ok().map(PathBuf::from)
}

/// A deterministic activation of roughly the scale an RMSNorm produces.
fn activation(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 32) as u32 as f64 / u32::MAX as f64) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn quant(v: &moearc_model::tensors::TensorView<'_>) -> QuantType {
    QuantType::from_type_id(v.quant.id).expect("an expert bank this build can expand")
}

fn bank_spec(m: &MappedModel, block: u32, bank: ExpertBank) -> BankSpec {
    let v = m.expert(block, bank, 0).expect("expert 0 exists");
    let (&n_cols, rest) = v.dims.split_first().expect("a matrix has dimensions");
    BankSpec {
        ty: quant(&v),
        n_rows: rest.iter().product::<u64>() as usize,
        n_cols: n_cols as usize,
    }
}

fn block_spec(m: &MappedModel, block: u32) -> BlockSpec {
    BlockSpec {
        gate: bank_spec(m, block, ExpertBank::Gate),
        up: bank_spec(m, block, ExpertBank::Up),
        down: bank_spec(m, block, ExpertBank::Down),
    }
}

fn n_blocks(m: &MappedModel) -> u32 {
    let mut b = 0;
    while m.expert(b, ExpertBank::Gate, 0).is_ok() {
        b += 1;
    }
    b
}

/// `down . (silu(gate . x) * (up . x))` on the device, through the same kernels `moe.rs` uses.
fn device_expert(
    ctx: &Context,
    spec: BlockSpec,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    x: &[f32],
) -> Vec<f32> {
    let n_ff = spec.gate.n_rows;
    let n_embd = spec.down.n_rows;

    let up_buf = |b: &[u8]| {
        let d = ctx.alloc(b.len()).expect("alloc");
        ctx.upload(&d, b).expect("upload");
        d
    };
    let g = up_buf(gate);
    let u = up_buf(up);
    let dn = up_buf(down);

    let xd = ctx.alloc_n::<f32>(x.len()).expect("alloc");
    ctx.upload_slice(&xd, x).expect("upload");
    let gv = ctx.alloc_n::<f32>(n_ff).expect("alloc");
    let uv = ctx.alloc_n::<f32>(n_ff).expect("alloc");
    let act = ctx.alloc_n::<f32>(n_ff).expect("alloc");
    let out = ctx.alloc_n::<f32>(n_embd).expect("alloc");

    ctx.matvec_q(spec.gate.ty, &gv, &g, &xd, n_ff, spec.gate.n_cols).expect("gate");
    ctx.matvec_q(spec.up.ty, &uv, &u, &xd, n_ff, spec.up.n_cols).expect("up");
    ctx.swiglu(&act, &gv, &uv, n_ff).expect("swiglu");
    ctx.matvec_q(spec.down.ty, &out, &dn, &act, n_embd, spec.down.n_cols).expect("down");

    let mut host = vec![0.0f32; n_embd];
    ctx.download_slice(&mut host, &out).expect("download");
    host
}

/// Largest `|a - b| / max(|b|, 1)`.
fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0f32, f32::max)
}

#[test]
fn the_host_expert_matches_the_device_on_real_weights() {
    let Some(path) = model_path() else {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_QWEN3MOE_MODEL");
        return;
    };
    let m = MappedModel::open(&path).expect("open");
    let ctx = Context::new().expect("device");
    let blocks = n_blocks(&m);
    assert!(blocks > 1, "a MoE model has blocks");

    // 🔴 Both ends of the file, not one. This model quantises `ffn_down_exps` at Q6_K in half
    // its blocks and Q4_K in the rest, so a check that only ever saw one of them would leave
    // one of the two host kernels untested against the device.
    let mut worst_seen = 0.0f32;
    let mut seen: Vec<(QuantType, f32)> = Vec::new();
    for block in [0u32, 1, blocks / 2, blocks - 2, blocks - 1] {
        let spec = block_spec(&m, block);
        for expert in [0u32, 7, 63] {
            let g = m.expert(block, ExpertBank::Gate, expert).expect("gate");
            let u = m.expert(block, ExpertBank::Up, expert).expect("up");
            let d = m.expert(block, ExpertBank::Down, expert).expect("down");
            let x = activation(spec.gate.n_cols, u64::from(block) * 977 + u64::from(expert));

            let want = device_expert(&ctx, spec, g.data, u.data, d.data, &x);
            let got = expert_ffn(spec, g.data, u.data, d.data, &x);
            let e = worst(&got, &want);
            worst_seen = worst_seen.max(e);
            seen.push((spec.down.ty, e));
            assert!(
                e < TOL,
                "block {block} expert {expert} ({:?} gate, {:?} down): host is {e} off the device",
                spec.gate.ty,
                spec.down.ty
            );
        }
    }
    let q4 = seen.iter().filter(|(t, _)| *t == QuantType::Q4K).count();
    let q6 = seen.iter().filter(|(t, _)| *t == QuantType::Q6K).count();
    println!(
        "host vs device: worst relative difference {worst_seen:e} over {} experts",
        seen.len()
    );
    println!("  down banks covered: {q4} Q4_K, {q6} Q6_K");
    assert!(q4 > 0 && q6 > 0, "both down-bank quantisations must be covered, saw {q4}/{q6}");
}

#[test]
fn the_threaded_executor_reproduces_the_single_threaded_expert() {
    let Some(path) = model_path() else {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_QWEN3MOE_MODEL");
        return;
    };
    let m = Arc::new(MappedModel::open(&path).expect("open"));
    let info = ModelInfo::from_header(m.header()).expect("header");
    let blocks = n_blocks(&m);
    let specs: Vec<BlockSpec> = (0..blocks).map(|b| block_spec(&m, b)).collect();
    let geom = Geometry {
        n_block: blocks as usize,
        n_expert: info.total_experts as usize,
        n_expert_used: info.active_experts as usize,
        n_embd: specs[0].gate.n_cols,
        n_ff: specs[0].gate.n_rows,
    };
    // Deliberately more than one thread and more than one expert: the fork-join, the barrier
    // between the two phases and the router-weighted combine are all only exercised together.
    let exec = HostExecutor::new(Arc::clone(&m), geom, &specs, 4).expect("executor");

    for block in [0u32, blocks - 1] {
        let spec = specs[block as usize];
        let picks: Vec<(u16, f32)> = vec![(3, 0.5), (11, 0.25), (40, 0.125), (2, 0.125)];
        let x = activation(geom.n_embd, u64::from(block) + 5);

        let mut want = vec![0.0f32; geom.n_embd];
        for (e, w) in &picks {
            let g = m.expert(block, ExpertBank::Gate, u32::from(*e)).expect("gate");
            let u = m.expert(block, ExpertBank::Up, u32::from(*e)).expect("up");
            let d = m.expert(block, ExpertBank::Down, u32::from(*e)).expect("down");
            let one = expert_ffn(spec, g.data, u.data, d.data, &x);
            for (acc, v) in want.iter_mut().zip(&one) {
                *acc += w * v;
            }
        }

        // Twice, because a job that leaves state behind would only show on the second one.
        for _ in 0..2 {
            let mut got = vec![0.0f32; geom.n_embd];
            let job = exec.submit(block as usize, spec, &picks, &x).expect("submit");
            exec.sync(job, &mut got).expect("sync");
            let e = worst(&got, &want);
            assert!(e < 1e-5, "block {block}: threaded executor is {e} off the plain one");
        }
    }
    let s = exec.stats();
    assert_eq!(s.jobs, 4);
    assert_eq!(s.experts, 16);
    println!("executor: {} jobs, {} experts, busy {} us", s.jobs, s.experts, s.busy_nanos / 1000);
}

#[test]
fn group_sums_are_what_the_q4k_kernel_assumes() {
    // Not a formality: the Q4_K kernel indexes `xsum[2 * j]` and `xsum[2 * j + 1]` inside a
    // 256-element super-block, so a group size other than 32 would read another block's sums and
    // produce a finite, plausible, wrong dot product.
    let x = activation(2048, 1);
    let s = group_sums(&x);
    assert_eq!(s.len(), 64);
    for (i, v) in s.iter().enumerate() {
        let want: f32 = x[i * 32..i * 32 + 32].iter().sum();
        assert_eq!(*v, want);
    }
}

/// `def fibonacci(n):\n    `, the prompt `qwen3moe_forward.rs` gates on — chosen there because
/// both of llama.cpp's backends agree on its whole continuation, which makes it robust to
/// exactly the class of rounding difference the host path introduces.
const PROMPT: [u32; 5] = [750, 75698, 1445, 982, 257];

fn greedy(path: &std::path::Path, host: HostPolicy, n: usize) -> Vec<u32> {
    let opts = SessionOptions {
        n_ctx: Some(256),
        // Low enough that most experts miss, so the policy has something to route.
        residency: Residency::Slots(264),
        host,
    };
    let s = Session::load_with(path, opts).expect("load");
    let stop = StopConditions { max_tokens: n, stop_tokens: Vec::new() };
    let mut ids = Vec::new();
    s.generate(&PROMPT, &stop, &mut |t| {
        ids.push(t);
        true
    })
    .expect("generate");
    ids
}

#[test]
fn a_host_policy_changes_where_an_expert_runs_and_not_what_it_computes() {
    let Some(path) = model_path() else {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 and MOEARC_QWEN3MOE_MODEL");
        return;
    };
    // 🔴 This is the gate for the whole feature. Residency already has one of these
    // (`constraining_residency_does_not_change_a_single_token_id`) and it means the same thing:
    // a budget decides what has to *move*, a host policy decides where a miss is *computed*, and
    // neither may change the function. The host path rounds differently from the device's —
    // `the_host_expert_matches_the_device_on_real_weights` measures how differently — so this
    // asserts the difference stays below the decision boundary of a greedy argmax.
    //
    // One session at a time: at 264 slots each holds 770 MiB of pool plus 951 MiB of dense
    // weights, and three at once do not fit on this card.
    let control = greedy(&path, HostPolicy::Off, 32);
    assert_eq!(control.len(), 32);
    for policy in [HostPolicy::Fraction(0.5), HostPolicy::Fraction(1.0), HostPolicy::Over(4)] {
        let got = greedy(&path, policy, 32);
        assert_eq!(got, control, "{policy} changed the continuation");
    }
}
