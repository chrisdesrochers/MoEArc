//! The token loop: what actually happens to produce one token.
//!
//! This is where the pieces meet. Weight storage, device compute and routing are traits, so
//! the orchestration can be written, tested and reasoned about before any of them exist — and
//! so a mistake in the *order* of operations is caught here rather than discovered later as
//! wrong output from a kernel that was working correctly all along.
//!
//! The order is the substance. For each block, in sequence:
//!
//! 1. Ask the router which experts this token needs.
//! 2. Ask the cache which are resident and which must be fetched, getting back slot
//!    assignments.
//! 3. **Stage every miss before computing anything.** A block cannot start until its experts
//!    are in VRAM.
//! 4. Compute the block against those slots.
//!
//! Step 3 is not negotiable and it is the reason blocks cannot overlap: block N+1's router runs
//! on block N's output, so its experts cannot even be *named* until N finishes. Measurement on
//! an Arc B580 showed this costs about 2% versus one bulk transfer — see `docs/calibration.md`
//! — because each block's fetch already saturates the link. That is why this loop is written
//! plainly rather than around a prefetch pipeline.

use crate::cache::{CacheError, ExpertCache, Slot};
use crate::kv::{KvError, PageId, PagedKvCache, SeqId};
use crate::residency::ExpertRef;

/// Where expert weights live, and how they reach the device.
pub trait ExpertStore {
    /// Bytes one expert occupies. Used for traffic accounting.
    fn expert_bytes(&self) -> u64;
    /// Copy one expert's weights into a device slot.
    fn stage(&mut self, expert: ExpertRef, into_slot: Slot) -> Result<(), StepError>;
}

/// Which experts a block wants for the current token.
///
/// In the real engine this is a device-side top-k over router logits. It is a trait because the
/// routing decision is the input to everything else here, and being able to replay a recorded
/// trace through the loop is worth more than the abstraction costs.
pub trait Router {
    fn select(&mut self, block: u16) -> Vec<ExpertRef>;
}

/// Running one block on the device.
pub trait BlockCompute {
    /// Compute block `block`, reading its experts from `slots` (in the order the router named
    /// them) and writing KV into `kv_page` at `kv_slot`.
    fn run_block(
        &mut self,
        block: u16,
        slots: &[Slot],
        kv_page: PageId,
        kv_slot: u32,
    ) -> Result<(), StepError>;
}

/// Why a token could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    Cache(CacheError),
    Kv(KvError),
    /// A device operation failed. Carries a static description; the device layer logs detail.
    Device(&'static str),
}

impl From<CacheError> for StepError {
    fn from(e: CacheError) -> Self {
        Self::Cache(e)
    }
}
impl From<KvError> for StepError {
    fn from(e: KvError) -> Self {
        Self::Kv(e)
    }
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(e) => write!(f, "expert cache: {e}"),
            Self::Kv(e) => write!(f, "kv cache: {e}"),
            Self::Device(what) => write!(f, "device: {what}"),
        }
    }
}

impl std::error::Error for StepError {}

/// What one token cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenCost {
    pub blocks: u32,
    pub expert_reads: u64,
    pub expert_hits: u64,
    pub expert_misses: u64,
    /// Bytes staged across the bus for this token.
    pub bytes_staged: u64,
}

impl TokenCost {
    pub fn hit_rate(&self) -> f64 {
        if self.expert_reads == 0 {
            0.0
        } else {
            self.expert_hits as f64 / self.expert_reads as f64
        }
    }
}

/// Drives one sequence through the model.
pub struct Runtime {
    blocks: u16,
    cache: ExpertCache,
    kv: PagedKvCache,
}

impl Runtime {
    pub fn new(blocks: u16, cache: ExpertCache, kv: PagedKvCache) -> Self {
        Self { blocks, cache, kv }
    }

    pub fn cache(&self) -> &ExpertCache {
        &self.cache
    }
    pub fn kv(&self) -> &PagedKvCache {
        &self.kv
    }

    /// Begin a sequence with a prompt of `prompt_tokens`.
    pub fn begin(&mut self, seq: SeqId, prompt_tokens: u32) -> Result<(), StepError> {
        self.kv.begin(seq, prompt_tokens)?;
        Ok(())
    }

    pub fn end(&mut self, seq: SeqId) -> Result<(), StepError> {
        self.kv.end(seq)?;
        Ok(())
    }

    /// Produce one token: walk every block, staging experts and computing in order.
    pub fn step<S, R, C>(
        &mut self,
        seq: SeqId,
        store: &mut S,
        router: &mut R,
        compute: &mut C,
    ) -> Result<TokenCost, StepError>
    where
        S: ExpertStore,
        R: Router,
        C: BlockCompute,
    {
        // One KV slot for this token, taken once and shared by every block that caches.
        let (kv_page, kv_slot) = self.kv.append(seq)?;

        let mut cost = TokenCost { blocks: self.blocks as u32, ..Default::default() };
        let before = self.cache.stats();

        for block in 0..self.blocks {
            let wanted = router.select(block);
            if wanted.is_empty() {
                // A dense block, or one whose experts are always resident. Still computed.
                compute.run_block(block, &[], kv_page, kv_slot)?;
                continue;
            }

            let plan = self.cache.admit(&wanted)?;

            // Every miss is staged BEFORE the block runs. Computing against a slot whose
            // contents are still in flight is the failure that produces plausible-looking
            // wrong output rather than an error.
            for load in &plan.loads {
                store.stage(load.expert, load.into_slot)?;
                cost.bytes_staged += store.expert_bytes();
            }

            let slots = plan.slots_for(&wanted);
            compute.run_block(block, &slots, kv_page, kv_slot)?;
        }

        let after = self.cache.stats();
        cost.expert_reads = after.demands - before.demands;
        cost.expert_hits = after.hits - before.hits;
        cost.expert_misses = after.misses - before.misses;
        Ok(cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::{ExpertRef, synthetic_trace};

    /// Records what was staged, so the loop's behaviour can be asserted rather than assumed.
    #[derive(Default)]
    struct RecordingStore {
        staged: Vec<(ExpertRef, Slot)>,
        bytes: u64,
    }
    impl ExpertStore for RecordingStore {
        fn expert_bytes(&self) -> u64 {
            self.bytes
        }
        fn stage(&mut self, expert: ExpertRef, into_slot: Slot) -> Result<(), StepError> {
            self.staged.push((expert, into_slot));
            Ok(())
        }
    }

    /// Replays a recorded routing decision, one block at a time.
    struct ScriptedRouter {
        per_block: Vec<Vec<ExpertRef>>,
    }
    impl Router for ScriptedRouter {
        fn select(&mut self, block: u16) -> Vec<ExpertRef> {
            self.per_block.get(block as usize).cloned().unwrap_or_default()
        }
    }

    /// Records the order blocks ran in and the slots each was given.
    #[derive(Default)]
    struct RecordingCompute {
        ran: Vec<(u16, Vec<Slot>)>,
    }
    impl BlockCompute for RecordingCompute {
        fn run_block(
            &mut self,
            block: u16,
            slots: &[Slot],
            _kv_page: PageId,
            _kv_slot: u32,
        ) -> Result<(), StepError> {
            self.ran.push((block, slots.to_vec()));
            Ok(())
        }
    }

    fn e(l: u16, x: u16) -> ExpertRef {
        ExpertRef::new(l, x)
    }

    fn runtime(blocks: u16, slots: u32) -> Runtime {
        Runtime::new(
            blocks,
            ExpertCache::new(slots).unwrap(),
            PagedKvCache::new(64, 16, 4096).unwrap(),
        )
    }

    #[test]
    fn blocks_run_in_order_and_each_gets_its_own_slots() {
        let mut rt = runtime(3, 16);
        rt.begin(1, 4).unwrap();
        let mut store = RecordingStore { bytes: 1000, ..Default::default() };
        let mut router = ScriptedRouter {
            per_block: vec![vec![e(0, 5), e(0, 9)], vec![e(1, 2)], vec![e(2, 7), e(2, 1)]],
        };
        let mut compute = RecordingCompute::default();

        let cost = rt.step(1, &mut store, &mut router, &mut compute).unwrap();

        let order: Vec<u16> = compute.ran.iter().map(|(b, _)| *b).collect();
        assert_eq!(order, vec![0, 1, 2], "blocks must run in sequence");
        assert_eq!(compute.ran[0].1.len(), 2);
        assert_eq!(compute.ran[1].1.len(), 1);
        assert_eq!(cost.expert_reads, 5);
        assert_eq!(cost.expert_misses, 5, "cold cache");
        assert_eq!(cost.bytes_staged, 5000);
    }

    #[test]
    fn every_expert_is_staged_before_its_block_computes() {
        // The invariant the whole loop exists to hold. Computing against a slot still being
        // filled yields plausible wrong output, never an error, so it is asserted directly.
        // A store and a compute that share one record of what has been staged, via a cell,
        // so the compute side can check that every slot it is handed was filled first.
        use std::cell::RefCell;
        use std::rc::Rc;

        #[derive(Default)]
        struct Shared {
            staged: Vec<Slot>,
            failures: Vec<String>,
        }
        struct CheckStore(Rc<RefCell<Shared>>);
        impl ExpertStore for CheckStore {
            fn expert_bytes(&self) -> u64 {
                0
            }
            fn stage(&mut self, _e: ExpertRef, into_slot: Slot) -> Result<(), StepError> {
                self.0.borrow_mut().staged.push(into_slot);
                Ok(())
            }
        }
        struct CheckCompute(Rc<RefCell<Shared>>);
        impl BlockCompute for CheckCompute {
            fn run_block(
                &mut self,
                block: u16,
                slots: &[Slot],
                _p: PageId,
                _s: u32,
            ) -> Result<(), StepError> {
                let mut sh = self.0.borrow_mut();
                for s in slots {
                    if !sh.staged.contains(s) {
                        sh.failures.push(format!("block {block} read unstaged slot {s}"));
                    }
                }
                Ok(())
            }
        }

        // One object plays both roles, so it observes staging and compute interleaved exactly
        // as the runtime issues them — which is the only way to check their relative order.
        // Capacity 4 with 8 experts wanted forces evictions, so slots really are reused.
        let shared = Rc::new(RefCell::new(Shared::default()));
        let mut rt = runtime(4, 4);
        rt.begin(1, 1).unwrap();
        let mut router =
            ScriptedRouter { per_block: (0..4).map(|b| vec![e(b, 1), e(b, 2)]).collect() };
        let mut store = CheckStore(Rc::clone(&shared));
        let mut compute = CheckCompute(Rc::clone(&shared));

        rt.step(1, &mut store, &mut router, &mut compute).unwrap();

        let sh = shared.borrow();
        assert!(sh.failures.is_empty(), "{:?}", sh.failures);
        assert!(!sh.staged.is_empty(), "the test staged nothing, so it proved nothing");
    }

    #[test]
    fn a_warm_cache_stages_nothing() {
        let mut rt = runtime(2, 16);
        rt.begin(1, 1).unwrap();
        let mut store = RecordingStore { bytes: 100, ..Default::default() };
        let mut router = ScriptedRouter { per_block: vec![vec![e(0, 1)], vec![e(1, 1)]] };
        let mut compute = RecordingCompute::default();

        rt.step(1, &mut store, &mut router, &mut compute).unwrap();
        let staged_after_first = store.staged.len();
        let cost = rt.step(1, &mut store, &mut router, &mut compute).unwrap();

        assert_eq!(store.staged.len(), staged_after_first, "nothing should move on a repeat");
        assert_eq!(cost.expert_misses, 0);
        assert_eq!(cost.bytes_staged, 0);
        assert!((cost.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn a_dense_block_still_computes() {
        let mut rt = runtime(3, 8);
        rt.begin(1, 1).unwrap();
        let mut store = RecordingStore { bytes: 1, ..Default::default() };
        // Block 1 has no experts — a dense or always-resident layer.
        let mut router = ScriptedRouter { per_block: vec![vec![e(0, 1)], vec![], vec![e(2, 1)]] };
        let mut compute = RecordingCompute::default();
        rt.step(1, &mut store, &mut router, &mut compute).unwrap();
        assert_eq!(compute.ran.len(), 3, "the dense block must still run");
        assert!(compute.ran[1].1.is_empty());
    }

    #[test]
    fn each_token_consumes_exactly_one_kv_slot() {
        let mut rt = runtime(2, 8);
        rt.begin(1, 0).unwrap();
        let mut store = RecordingStore { bytes: 1, ..Default::default() };
        let mut router = ScriptedRouter { per_block: vec![vec![e(0, 1)], vec![e(1, 1)]] };
        let mut compute = RecordingCompute::default();
        for _ in 0..40 {
            rt.step(1, &mut store, &mut router, &mut compute).unwrap();
        }
        assert_eq!(rt.kv().pages_of(1).unwrap().tokens, 40);
        // 40 tokens over 16-token pages.
        assert_eq!(rt.kv().pages_of(1).unwrap().pages.len(), 3);
    }

    #[test]
    fn traffic_over_a_synthetic_trace_matches_the_cache_accounting() {
        // Drives the full loop with a realistic-shaped trace and checks the runtime's
        // per-token accounting sums to what the cache reports over the whole run. A drift
        // between them would make every throughput number wrong.
        let blocks = 8u16;
        let trace = synthetic_trace(60, blocks, 32, 3, 0.7, 99);
        let mut rt = runtime(blocks, 40);
        rt.begin(1, 0).unwrap();
        let mut store = RecordingStore { bytes: 2_039_808, ..Default::default() };
        let mut compute = RecordingCompute::default();

        let mut total = TokenCost::default();
        for step in &trace.steps {
            let mut per_block: Vec<Vec<ExpertRef>> = vec![Vec::new(); blocks as usize];
            for &x in step {
                per_block[x.layer as usize].push(x);
            }
            let mut router = ScriptedRouter { per_block };
            let c = rt.step(1, &mut store, &mut router, &mut compute).unwrap();
            total.expert_reads += c.expert_reads;
            total.expert_hits += c.expert_hits;
            total.expert_misses += c.expert_misses;
            total.bytes_staged += c.bytes_staged;
        }

        let s = rt.cache().stats();
        assert_eq!(total.expert_reads, s.demands);
        assert_eq!(total.expert_hits, s.hits);
        assert_eq!(total.expert_misses, s.misses);
        assert_eq!(total.bytes_staged, s.misses * 2_039_808);
        assert_eq!(store.staged.len() as u64, s.misses, "one stage per miss, no more");
    }
}
