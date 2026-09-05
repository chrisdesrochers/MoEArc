//! Proof that the seam works end to end: Rust -> C ABI -> SYCL -> Arc GPU -> back.
//!
//! These need a real GPU, so they are gated behind `MOEARC_TEST_GPU=1`. CI without an Arc card
//! stays green, and a machine with one can prove the whole path in one command.

use moearc_kernels::{Context, KernelError};

fn gpu_available() -> bool {
    std::env::var_os("MOEARC_TEST_GPU").is_some()
}

#[test]
fn opens_a_queue_on_a_real_device() {
    if !gpu_available() {
        eprintln!("skipped: set MOEARC_TEST_GPU=1 to run against a real GPU");
        return;
    }
    let ctx = Context::new().expect("no GPU context");
    let name = ctx.device_name().expect("device name");
    eprintln!("device: {name}");
    assert!(!name.is_empty());
}

#[test]
fn a_round_trip_returns_what_was_sent() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let buf = ctx.alloc(src.len()).unwrap();
    ctx.upload(&buf, &src).unwrap();
    let mut back = vec![0u8; src.len()];
    ctx.download(&mut back, &buf).unwrap();
    assert_eq!(src, back, "device round trip corrupted the data");
}

#[test]
fn the_expert_gather_selects_the_right_slots() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();

    // A pool of 64 slots, each 1 KiB, where every word in slot N holds N. Then gather a
    // scattered, out-of-order, repeating selection — the shape a router actually produces.
    const SLOTS: usize = 64;
    const SLOT_BYTES: usize = 1024;
    const WORDS: usize = SLOT_BYTES / 4;

    let mut pool_host = vec![0u32; SLOTS * WORDS];
    for slot in 0..SLOTS {
        for w in 0..WORDS {
            pool_host[slot * WORDS + w] = slot as u32;
        }
    }
    let pool_bytes: &[u8] = bytemuck_cast(&pool_host);
    let pool = ctx.alloc(pool_bytes.len()).unwrap();
    ctx.upload(&pool, pool_bytes).unwrap();

    let idx: Vec<u32> = vec![63, 0, 17, 17, 42, 1, 8, 55];
    let dst = ctx.alloc(idx.len() * SLOT_BYTES).unwrap();
    ctx.gather_experts(&dst, &pool, &idx, SLOT_BYTES).unwrap();

    let mut out = vec![0u8; idx.len() * SLOT_BYTES];
    ctx.download(&mut out, &dst).unwrap();
    let out_words: &[u32] = bytemuck_cast_back(&out);

    for (i, &want) in idx.iter().enumerate() {
        for w in 0..WORDS {
            let got = out_words[i * WORDS + w];
            assert_eq!(got, want, "gathered slot {i} word {w}: expected expert {want}, got {got}");
        }
    }
    eprintln!("gathered {} slots x {SLOT_BYTES} B correctly, including a repeat", idx.len());
}

#[test]
fn a_misaligned_slot_size_is_refused_rather_than_corrupting() {
    if !gpu_available() {
        return;
    }
    let ctx = Context::new().unwrap();
    let pool = ctx.alloc(4096).unwrap();
    let dst = ctx.alloc(4096).unwrap();
    let err = ctx.gather_experts(&dst, &pool, &[0], 1023).unwrap_err();
    assert_eq!(err, KernelError::Misaligned { slot_bytes: 1023 });
}

// Small local casts so the crate needs no dependency for a test helper.
fn bytemuck_cast(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
fn bytemuck_cast_back(v: &[u8]) -> &[u32] {
    assert!(v.len().is_multiple_of(4));
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() / 4) }
}
