//! What one kernel launch costs, separated from what the kernel computes.
//!
//! ```text
//! cargo run -p moearc-kernels --example launch_overhead
//! ```
//!
//! The engine issues on the order of a thousand kernels per decoded token and waits on each
//! one. Whether that is expensive is not a matter of opinion, but it is not visible in a
//! profile of the forward pass either — there, launch overhead and arithmetic are added
//! together inside every call. This measures them apart: the same trivial kernel, submitted the
//! two ways, over enough repetitions that the difference is the submission model and nothing
//! else.
//!
//! `n = 1` on purpose. A kernel over one element does no work worth measuring, so whatever time
//! it takes is the cost of asking.

use std::time::Instant;

use moearc_kernels::{Context, KvType};

const REPS: usize = 2000;

fn main() {
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no device: {e}");
            return;
        }
    };
    println!("device {}", ctx.device_name().unwrap_or_default());

    let a = ctx.alloc_n::<f32>(1024).unwrap();
    let b = ctx.alloc_n::<f32>(1024).unwrap();
    let out = ctx.alloc_n::<f32>(1024).unwrap();
    ctx.upload_slice(&a, &[1.0f32; 1024]).unwrap();
    ctx.upload_slice(&b, &[2.0f32; 1024]).unwrap();

    // Warm up: the first submission on a queue builds command lists and loads the module.
    for _ in 0..64 {
        ctx.add(&out, &a, &b, 1).unwrap();
    }
    ctx.sync().unwrap();

    let t = Instant::now();
    for _ in 0..REPS {
        ctx.add(&out, &a, &b, 1).unwrap();
    }
    ctx.sync().unwrap();
    let per_launch = t.elapsed().as_secs_f64() / REPS as f64;

    let t = Instant::now();
    for _ in 0..REPS {
        ctx.add(&out, &a, &b, 1).unwrap();
        ctx.sync().unwrap();
    }
    let per_launch_sync = t.elapsed().as_secs_f64() / REPS as f64;

    let mut host = [0.0f32; 1];
    let t = Instant::now();
    for _ in 0..REPS {
        ctx.download_slice(&mut host, &out).unwrap();
    }
    let per_readback = t.elapsed().as_secs_f64() / REPS as f64;

    // ---- attention: the launch that was untracked until now ---------------------------------
    //
    // `moearc_attn_decode` now calls `moearc_track`, so attention is attributable from an
    // ordinary asynchronous run instead of only under `MOEARC_SYNC_EACH`. With
    // `MOEARC_PROFILE_EVENTS` unset that call returns on one branch — but "should be free" is
    // an opinion, and this file exists because whether a launch is expensive is not one.
    //
    // `n_kv = 1`, so the kernel body attends to a single key and does no work worth timing;
    // what is left is the cost of asking. Run this arm on a build with the tracking call and
    // one without it: any cost the call has on the hot path is the difference between them.
    const HEADS: usize = 64;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;
    const PAGE_TOKENS: usize = 64;
    let pool = PAGE_TOKENS * KV_HEADS * HEAD_DIM;
    let dk = ctx.alloc_n::<f32>(pool).unwrap();
    let dv = ctx.alloc_n::<f32>(pool).unwrap();
    let dq = ctx.alloc_n::<f32>(HEADS * HEAD_DIM).unwrap();
    let dattn = ctx.alloc_n::<f32>(HEADS * HEAD_DIM).unwrap();
    let dbt = ctx.alloc_n::<u32>(1).unwrap();
    ctx.upload_slice(&dk, &vec![0.0f32; pool]).unwrap();
    ctx.upload_slice(&dv, &vec![0.0f32; pool]).unwrap();
    ctx.upload_slice(&dq, &vec![0.0f32; HEADS * HEAD_DIM]).unwrap();
    ctx.upload_slice(&dbt, &[0u32]).unwrap();
    let submit_attn = || {
        ctx.attn_decode(
            &dattn,
            &dq,
            &dk,
            &dv,
            &dbt,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            1,
            PAGE_TOKENS,
            0.125,
            KvType::F32,
        )
        .unwrap();
    };
    for _ in 0..64 {
        submit_attn();
    }
    ctx.sync().unwrap();
    let t = Instant::now();
    for _ in 0..REPS {
        submit_attn();
    }
    ctx.sync().unwrap();
    let per_attn = t.elapsed().as_secs_f64() / REPS as f64;

    println!("submit only          {:8.1} us/launch", per_launch * 1e6);
    println!("submit + wait        {:8.1} us/launch", per_launch_sync * 1e6);
    println!("4-byte device->host  {:8.1} us/readback", per_readback * 1e6);
    println!(
        "attn_decode submit   {:8.3} us/launch  (n_heads {HEADS}, kv_heads {KV_HEADS}, \
         head_dim {HEAD_DIM}, n_kv 1, {REPS} reps)",
        per_attn * 1e6
    );
    println!(
        "\nsynchronising costs {:.1} us more per launch than not.",
        (per_launch_sync - per_launch) * 1e6
    );
}
