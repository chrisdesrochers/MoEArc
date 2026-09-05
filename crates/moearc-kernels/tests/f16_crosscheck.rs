//! Cross-check the f32 -> f16 conversion against **llama.cpp's own converter**.
//!
//! `forward_pass_gpu.rs` already checks the GPU against this crate's CPU reference over every
//! f16 bit pattern. Both are mine, and the rounding rule — ties to even, subnormals rounded
//! rather than flushed, overflow to infinity — is exactly the sort of thing two implementations
//! by one author get consistently wrong together. `ggml_fp32_to_fp16_row` is the arbiter.
//!
//! `tools/ggml_f16_ref.c` produces the golden files; point `MOEARC_ATTN_GOLDEN` at the output
//! directory. Without it the test skips loudly.
//!
//! # 🔴 A measured divergence, kept deliberately
//!
//! The two agree bit for bit on every input in the f16 range — but **not** on inputs that
//! overflow it, and MoEArc is the one that is right. Measured on the llama.cpp build in this
//! environment by walking f32 bit patterns:
//!
//! * every input below **65568.0078** converts identically, including all 4096 in
//!   `[65520, 65536)` that must saturate to infinity;
//! * from 65568.0078 upward ggml stops saturating. 70000 becomes `0x7c46`, a **NaN**;
//! * above 2^17 the exponent overflows out of its field into the sign bit: 131072 becomes
//!   `0x8000`, **negative zero**, and 645252 becomes `0x08ec`, an ordinary small number.
//!
//! IEEE-754 says an overflowing conversion saturates to infinity, which is what this crate
//! does. Matching ggml here would be actively dangerous: a KV cache that silently turned a
//! large activation into a small one would corrupt an answer with no signal at all, where an
//! infinity is at least loud. So the cross-check asserts equality over the representable domain
//! and asserts *IEEE* behaviour outside it, and this comment is the record of why the two
//! differ.
//!
//! This was observed on one build (llama.cpp compiled with `icx`) and has not been checked
//! against a stock gcc build, so it is reported as an environment finding rather than as a
//! defect in llama.cpp.

mod common;

use common::gpu_available;
use moearc_kernels::{Context, reference};
use std::path::PathBuf;

#[test]
fn f32_to_f16_is_bit_identical_to_llama_cpps_converter() {
    if !gpu_available() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 to run against a real GPU");
        return;
    }
    let Some(dir) = std::env::var_os("MOEARC_ATTN_GOLDEN").map(PathBuf::from) else {
        eprintln!(
            "skipped: set MOEARC_ATTN_GOLDEN to a directory produced by tools/ggml_f16_ref.c"
        );
        return;
    };
    let inb = std::fs::read(dir.join("f16.in.f32")).expect("f16.in.f32");
    let outb = std::fs::read(dir.join("f16.out.u16")).expect("f16.out.u16");
    let inputs: Vec<f32> =
        inb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let want: Vec<u16> = outb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    assert_eq!(inputs.len(), want.len(), "golden input and output lengths disagree");
    assert!(inputs.len() > 100_000, "golden sweep is too small to mean much");

    let ctx = Context::new().unwrap();
    let n = inputs.len();
    let dsrc = ctx.alloc_n::<f32>(n).unwrap();
    let ddst = ctx.alloc(n * 2).unwrap();
    ctx.upload_slice(&dsrc, &inputs).unwrap();
    ctx.quantize_f16(&ddst, &dsrc, n).unwrap();
    let mut got = vec![0u16; n];
    ctx.download_slice(&mut got, &ddst).unwrap();

    // The domain over which ggml's converter is IEEE-correct on this build. Measured, not
    // guessed: see the module comment. 65536 is comfortably below the first divergence at
    // 65568.0078 and is the natural boundary to state.
    const GGML_SATURATES_BELOW: f32 = 65536.0;

    let mut compared = 0usize;
    let mut subnormals = 0usize;
    let mut rounded = 0usize;
    let mut overflow_checked = 0usize;
    let mut ggml_overflow_wrong = 0usize;

    for i in 0..n {
        if inputs[i].is_nan() {
            // A NaN must stay a NaN; *which* NaN is not specified by anything.
            assert!(
                reference::f16_to_f32(got[i]).is_nan(),
                "input {i} was NaN but converted to {:#06x}",
                got[i]
            );
            continue;
        }

        if inputs[i].is_finite() && inputs[i].abs() >= GGML_SATURATES_BELOW {
            // Outside ggml's correct domain. Assert IEEE behaviour of *our* converter, and
            // count how often ggml disagrees so the divergence is measured rather than assumed.
            let want_inf = if inputs[i] > 0.0 { 0x7C00 } else { 0xFC00 };
            assert_eq!(
                got[i], want_inf,
                "input {i} ({:e}) overflows f16 and must saturate to infinity, got {:#06x}",
                inputs[i], got[i]
            );
            assert_eq!(reference::f32_to_f16(inputs[i]), want_inf, "CPU reference at {i}");
            if want[i] != want_inf {
                ggml_overflow_wrong += 1;
            }
            overflow_checked += 1;
            continue;
        }

        assert_eq!(
            got[i], want[i],
            "input {i} ({:e}): MoEArc gave {:#06x}, llama.cpp gave {:#06x}",
            inputs[i], got[i], want[i]
        );
        // The CPU reference must agree with ggml too — otherwise both of mine could be wrong
        // together and only the GPU-vs-CPU test would notice.
        assert_eq!(reference::f32_to_f16(inputs[i]), want[i], "CPU reference disagreed at {i}");

        if got[i] & 0x7C00 == 0 && got[i] & 0x03FF != 0 {
            subnormals += 1;
        }
        if inputs[i] != reference::f16_to_f32(got[i]) {
            rounded += 1;
        }
        compared += 1;
    }

    assert!(compared > 100_000, "only {compared} values were actually compared");
    assert!(overflow_checked > 0, "the sweep contained no overflowing input");
    eprintln!(
        "f32 -> f16 bit-identical to ggml_fp32_to_fp16_row over {compared} in-range values \
         ({subnormals} landed on subnormals, {rounded} required rounding); \
         {overflow_checked} overflowing inputs saturated to infinity as IEEE requires, \
         of which ggml got {ggml_overflow_wrong} wrong on this build"
    );
}
