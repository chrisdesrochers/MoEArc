#![allow(dead_code)] // each test binary uses a different subset

//! Shared fixtures for the GPU tests: a deterministic RNG, synthetic quantised blocks, and an
//! error reporter that prints what it measured instead of only asserting.

use moearc_kernels::QuantType;

/// SplitMix64. Deterministic and seeded per test, so a failure is reproducible from the
/// failure message alone — a randomised fixture that cannot be replayed is worse than a fixed
/// one.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / 4_294_967_296.0) * 2.0 - 1.0
    }

    pub fn vec_unit(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

/// A finite, plausible f16 bit pattern for a super-block scale.
///
/// Sign 0, biased exponent 4..=10 — magnitudes around `2^-11` to `2^-5` — and a random
/// mantissa. Random *bits* would be inf or NaN one time in 32 and one comparison in 32 would
/// then be meaningless, so the field that matters most is the one field not left to chance.
fn scale_bits(rng: &mut Rng) -> u16 {
    0x1000 + (rng.next_u32() % 0x1800) as u16
}

/// Synthesise `nblocks` well-formed quantised blocks.
///
/// The quant payload is random bytes, which is legitimate: every bit pattern in `qs`/`qh`/
/// `scales` is a valid quantised value, and random ones exercise every nibble, every 6-bit
/// scale split and every high-bit position. Only the f16 scale fields are constrained.
pub fn synth_blocks(ty: QuantType, nblocks: usize, rng: &mut Rng) -> Vec<u8> {
    let bb = ty.block_bytes();
    let mut v: Vec<u8> = (0..nblocks * bb).map(|_| rng.next_u32() as u8).collect();
    for i in 0..nblocks {
        let blk = &mut v[i * bb..(i + 1) * bb];
        match ty {
            // Q6_K keeps its single scale at the end of the block; Q4_K and Q5_K keep two at
            // the front. Getting this backwards is exactly the kind of mistake the tests are
            // for, so it is written out per type rather than parameterised.
            QuantType::Q6K => blk[208..210].copy_from_slice(&scale_bits(rng).to_le_bytes()),
            // Q8_0 carries one delta and nothing else.
            QuantType::Q80 => blk[0..2].copy_from_slice(&scale_bits(rng).to_le_bytes()),
            // f32 and f16 are blocks of one, so the "payload" is the value itself and random
            // bytes would be NaN or infinity often enough to make every comparison meaningless.
            QuantType::F32 => blk[0..4].copy_from_slice(&rng.unit().to_le_bytes()),
            QuantType::F16 => blk[0..2]
                .copy_from_slice(&moearc_kernels::reference::f32_to_f16(rng.unit()).to_le_bytes()),
            QuantType::Q4K | QuantType::Q5K => {
                blk[0..2].copy_from_slice(&scale_bits(rng).to_le_bytes());
                blk[2..4].copy_from_slice(&scale_bits(rng).to_le_bytes());
            }
        }
    }
    v
}

/// Largest absolute difference between two equal-length slices, and where it was.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f64, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (f64::from(*x) - f64::from(*y)).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// Assert `got` matches `want` to within `tol`, and report the margin either way.
///
/// The report is the point. A test that only says "ok" leaves the tolerance unexamined; one
/// that prints the measured error every run makes it obvious when a change moves the number,
/// even while the assertion still passes.
pub fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f64) {
    let (err, at) = max_abs_diff(got, want);
    eprintln!("{label}: max |gpu - cpu| = {err:.3e} (tolerance {tol:.3e}, worst at index {at})");
    assert!(
        err <= tol,
        "{label}: max absolute error {err:.6e} exceeds tolerance {tol:.6e} at index {at} \
         (gpu {}, cpu {})",
        got[at],
        want[at]
    );
}

/// Whether the GPU tests should run at all.
pub fn gpu_available() -> bool {
    std::env::var_os("MOEARC_TEST_GPU").is_some()
}
