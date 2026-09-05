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

use moearc_kernels::Context;

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

    println!("submit only          {:8.1} us/launch", per_launch * 1e6);
    println!("submit + wait        {:8.1} us/launch", per_launch_sync * 1e6);
    println!("4-byte device->host  {:8.1} us/readback", per_readback * 1e6);
    println!(
        "\nsynchronising costs {:.1} us more per launch than not.",
        (per_launch_sync - per_launch) * 1e6
    );
}
