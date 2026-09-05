//! Turning a logits vector into one token id.
//!
//! Everything here is pure arithmetic over `&[f32]`. No device, no model, no async — which is
//! the point: sampling is the part of inference that can be tested to the last bit long before
//! a kernel exists, and it is also the part users notice when it is wrong. A GPU is never
//! needed to answer "does seed 42 give the same completion twice".
//!
//! **Order of operations matters and is not arbitrary.** Repetition penalty is applied to raw
//! logits, before temperature, because it is defined on the pre-scaled values (HF's
//! `RepetitionPenaltyLogitsProcessor`); scaling first would make the penalty's strength depend
//! on the temperature. Truncation (top-k, then top-p) happens after temperature because top-p
//! is a statement about the *sampled* distribution, and the sampled distribution is the one
//! temperature already shaped.

use serde::{Deserialize, Serialize};

/// How to turn logits into tokens.
///
/// Defaults match OpenAI's documented defaults where one exists (`temperature` 1.0, `top_p`
/// 1.0), and are no-ops where OpenAI has no equivalent (`top_k` 0, `repetition_penalty` 1.0),
/// so a request that sets nothing gets plain temperature sampling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// `0.0` means greedy: argmax, no RNG consulted at all.
    pub temperature: f32,
    /// Nucleus threshold in `0.0..=1.0`. `1.0` disables.
    pub top_p: f32,
    /// Keep only the `k` highest-logit candidates. `0` disables.
    pub top_k: usize,
    /// `> 1.0` discourages tokens already present, `< 1.0` encourages them. `1.0` disables.
    pub repetition_penalty: f32,
    /// `None` draws a seed from the clock, so repeated identical requests differ. Any `Some`
    /// value makes the whole completion reproducible.
    pub seed: Option<u64>,
    /// Hard cap on generated tokens.
    pub max_tokens: usize,
    /// Ids that end the completion. Filled from the tokeniser's EOS, plus anything the model
    /// declares.
    pub stop_tokens: Vec<u32>,
    /// Substrings that end the completion when they appear in the decoded text.
    pub stop_strings: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
            max_tokens: 128,
            stop_tokens: Vec::new(),
            stop_strings: Vec::new(),
        }
    }
}

impl SamplingParams {
    /// True when no RNG is involved and the result is a pure function of the logits.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// A small, self-contained PRNG (xoshiro256\*\*) used for token sampling.
///
/// 🔴 Deliberately not `rand`. `rand` does not promise value-stability across major versions,
/// and this crate's contract to the user is stronger than that: *the same seed must give the
/// same completion*, indefinitely, across upgrades. A reproducibility guarantee cannot rest on
/// a dependency that does not make one. Twenty lines here buys that outright.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        // SplitMix64 to spread one word across the four-word state. Seeding xoshiro directly
        // from a small integer leaves most of the state zero and correlates early output.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`. 53 bits, the most an `f64` represents exactly.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// One candidate token during sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: u32,
    /// Logit while truncating, probability after [`softmax_in_place`].
    pub score: f32,
}

/// Apply the repetition penalty in place, over the ids seen so far.
///
/// HF's asymmetric form: a positive logit is divided, a negative one multiplied. Dividing both
/// would *raise* an already-disfavoured token towards zero, i.e. penalising it would make it
/// more likely — which is the bug this shape exists to avoid.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    for &id in history {
        let Some(l) = logits.get_mut(id as usize) else { continue };
        *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
    }
}

/// Build the candidate set from logits, applying temperature then top-k then top-p.
///
/// Returned candidates hold *probabilities* summing to 1.0 and are sorted descending. Exposed
/// (rather than kept inside [`sample`]) so tests can assert on the surviving set directly:
/// "top-k restricted the candidates" is a claim about this list, not about which token came
/// out of one draw.
pub fn candidates(logits: &[f32], params: &SamplingParams) -> Vec<Candidate> {
    let temp = if params.temperature > 0.0 { params.temperature } else { 1.0 };
    let mut cands: Vec<Candidate> = logits
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .map(|(i, l)| Candidate { id: i as u32, score: l / temp })
        .collect();

    // Descending by score. `total_cmp` rather than `partial_cmp().unwrap()`: NaN is already
    // filtered above, but a sort comparator that can panic is a latent crash in a server.
    cands.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

    if params.top_k > 0 && params.top_k < cands.len() {
        cands.truncate(params.top_k);
    }

    softmax_in_place(&mut cands);

    if params.top_p < 1.0 {
        // Keep the smallest prefix whose mass reaches top_p. The token that crosses the
        // threshold is kept, so top_p can never produce an empty set — including the case
        // where one token already holds more mass than top_p.
        let mut acc = 0.0f32;
        let mut keep = cands.len();
        for (i, c) in cands.iter().enumerate() {
            acc += c.score;
            if acc >= params.top_p {
                keep = i + 1;
                break;
            }
        }
        cands.truncate(keep);
        softmax_renormalise(&mut cands);
    }

    cands
}

/// Softmax over candidate scores, in place. Shifted by the max for numerical stability — the
/// scores here are post-temperature, and a low temperature makes them large enough that a
/// naive `exp` overflows to infinity.
pub fn softmax_in_place(cands: &mut [Candidate]) {
    let max = cands.iter().map(|c| c.score).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for c in cands.iter_mut() {
        c.score = (c.score - max).exp();
        sum += c.score;
    }
    if sum > 0.0 {
        for c in cands.iter_mut() {
            c.score /= sum;
        }
    }
}

fn softmax_renormalise(cands: &mut [Candidate]) {
    let sum: f32 = cands.iter().map(|c| c.score).sum();
    if sum > 0.0 {
        for c in cands.iter_mut() {
            c.score /= sum;
        }
    }
}

/// Pick one token id from a logits vector.
///
/// `history` is the token ids the repetition penalty considers — prompt plus what has been
/// generated so far. `logits` is borrowed immutably and copied only when a penalty is actually
/// in effect, so the common path allocates once for the candidate list and no more.
pub fn sample(logits: &[f32], history: &[u32], params: &SamplingParams, rng: &mut Rng) -> u32 {
    let penalised;
    let logits = if (params.repetition_penalty - 1.0).abs() < f32::EPSILON {
        logits
    } else {
        penalised = {
            let mut v = logits.to_vec();
            apply_repetition_penalty(&mut v, history, params.repetition_penalty);
            v
        };
        &penalised
    };

    if params.is_greedy() {
        // Greedy does not build a candidate list, does not touch the RNG, and does not depend
        // on top_k/top_p: argmax of a truncated set is the argmax of the whole set. Ties go to
        // the lower id so the result is a function of the logits alone.
        return logits
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map_or(0, |(i, _)| i as u32);
    }

    let cands = candidates(logits, params);
    if cands.is_empty() {
        return 0;
    }
    let draw = rng.next_f64() as f32;
    let mut acc = 0.0f32;
    for c in &cands {
        acc += c.score;
        if draw < acc {
            return c.id;
        }
    }
    // Floating-point shortfall: the accumulated mass can land a hair under the draw. Falling
    // through to the last candidate is correct, and is the only branch that can be reached
    // without the loop returning.
    cands[cands.len() - 1].id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(f: impl FnOnce(&mut SamplingParams)) -> SamplingParams {
        let mut p = SamplingParams::default();
        f(&mut p);
        p
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        let p = params(|p| p.temperature = 0.0);
        for _ in 0..64 {
            let mut rng = Rng::seed_from_u64(9);
            assert_eq!(sample(&logits, &[], &p, &mut rng), 1);
        }
    }

    #[test]
    fn temperature_zero_ignores_the_rng() {
        // Greedy must not consume randomness: if it did, an interleaved greedy request would
        // shift a seeded one's stream.
        let logits = [1.0f32, 3.0, 2.0];
        let p = params(|p| p.temperature = 0.0);
        let mut rng = Rng::seed_from_u64(1);
        let before = rng.clone();
        sample(&logits, &[], &p, &mut rng);
        assert_eq!(rng.s, before.s);
    }

    #[test]
    fn same_seed_same_sequence() {
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 17) as f32 * 0.3).collect();
        let p = params(|p| {
            p.temperature = 0.9;
            p.top_p = 0.95;
        });
        let draw = |seed| {
            let mut rng = Rng::seed_from_u64(seed);
            (0..32).map(|_| sample(&logits, &[], &p, &mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42), "same seed must replay exactly");
        assert_ne!(draw(42), draw(43), "different seeds must diverge");
    }

    #[test]
    fn top_k_restricts_the_candidate_set() {
        let logits: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let p = params(|p| p.top_k = 5);
        let cands = candidates(&logits, &p);
        assert_eq!(cands.len(), 5);
        assert_eq!(cands.iter().map(|c| c.id).collect::<Vec<_>>(), vec![99, 98, 97, 96, 95]);
        let mass: f32 = cands.iter().map(|c| c.score).sum();
        assert!((mass - 1.0).abs() < 1e-5, "surviving mass must renormalise to 1, got {mass}");

        // And the restriction is observed by actual draws, not just by the list.
        let mut rng = Rng::seed_from_u64(7);
        for _ in 0..500 {
            assert!(sample(&logits, &[], &p, &mut rng) >= 95);
        }
    }

    #[test]
    fn top_p_restricts_the_candidate_set() {
        // Probabilities 0.5 / 0.25 / 0.125 / ... — top_p 0.7 must keep exactly two.
        let logits: Vec<f32> = (0..8).map(|i| -(i as f32) * std::f32::consts::LN_2).collect();
        let p = params(|p| p.top_p = 0.7);
        let cands = candidates(&logits, &p);
        assert_eq!(cands.iter().map(|c| c.id).collect::<Vec<_>>(), vec![0, 1]);

        let mut rng = Rng::seed_from_u64(3);
        for _ in 0..500 {
            assert!(sample(&logits, &[], &p, &mut rng) <= 1);
        }
    }

    #[test]
    fn top_p_never_empties_the_set() {
        // One token holds 0.999 of the mass and top_p is 0.1: the crossing token is kept.
        let logits = [20.0f32, 0.0, 0.0];
        let p = params(|p| p.top_p = 0.1);
        assert_eq!(candidates(&logits, &p).len(), 1);
    }

    #[test]
    fn repetition_penalty_pushes_seen_tokens_down() {
        let mut logits = [4.0f32, 4.0, -4.0, -4.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert_eq!(logits[0], 2.0, "positive logit is divided");
        assert_eq!(logits[1], 4.0, "unseen token untouched");
        assert_eq!(logits[2], -8.0, "negative logit is multiplied, i.e. pushed further down");
        assert_eq!(logits[3], -4.0);
    }

    #[test]
    fn repetition_penalty_can_flip_the_greedy_choice() {
        let logits = [5.0f32, 4.0];
        let p = params(|p| {
            p.temperature = 0.0;
            p.repetition_penalty = 2.0;
        });
        let mut rng = Rng::seed_from_u64(0);
        assert_eq!(sample(&logits, &[], &p, &mut rng), 0);
        assert_eq!(sample(&logits, &[0], &p, &mut rng), 1);
    }

    #[test]
    fn low_temperature_does_not_overflow() {
        // 0.01 scales a logit of 30 to 3000; exp(3000) is +inf. The max-shift is what stops
        // this returning NaN, and this is the regression test for it.
        let logits = [30.0f32, 29.0, 1.0];
        let p = params(|p| p.temperature = 0.01);
        let cands = candidates(&logits, &p);
        assert!(cands.iter().all(|c| c.score.is_finite()));
        assert!((cands.iter().map(|c| c.score).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rng_is_uniform_enough_to_trust() {
        let mut rng = Rng::seed_from_u64(12345);
        let n = 100_000;
        let mut buckets = [0u32; 10];
        for _ in 0..n {
            buckets[(rng.next_f64() * 10.0) as usize] += 1;
        }
        for b in buckets {
            assert!((b as i64 - n / 10).unsigned_abs() < 1000, "bucket skew: {buckets:?}");
        }
    }
}
