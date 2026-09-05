//! The live expert cache: what is in VRAM right now, and what must move.
//!
//! [`super::residency`] answers "which policy would be best" over a whole trace, offline. This
//! is the online half: a stateful map of slots to experts that, given the experts a step needs,
//! returns the transfers required to satisfy it.
//!
//! It deliberately performs no I/O and touches no device. It returns a [`StepPlan`] describing
//! what to move, and the caller executes it. That keeps the eviction logic — where the subtle
//! bugs live — testable on any machine, and it means the same logic can be replayed against a
//! recorded trace to validate a policy change without a GPU.
//!
//! # The invariant that matters
//!
//! A slot being reused within a step must not evict an expert that same step still needs. That
//! is the failure mode that produces silently wrong output rather than an error: the gather
//! reads a slot whose contents were overwritten before it ran. Every expert named in the
//! current step is pinned for its duration, and capacity is checked against the step's demand
//! up front so pinning can never starve eviction.

use std::collections::HashMap;

use super::residency::ExpertRef;

/// A slot index in the resident pool.
pub type Slot = u32;

/// One expert that must be brought into VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Load {
    /// The expert to fetch.
    pub expert: ExpertRef,
    /// The slot it will occupy.
    pub into_slot: Slot,
    /// The expert being displaced, if the slot was occupied. Recorded so the caller can
    /// account for it and so tests can assert on eviction order.
    pub evicted: Option<ExpertRef>,
}

/// What a step needs, split into what is already there and what must move.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StepPlan {
    /// Experts already resident, with the slot holding each.
    pub hits: Vec<(ExpertRef, Slot)>,
    /// Experts that must be fetched, in the order they should be issued.
    pub loads: Vec<Load>,
}

impl StepPlan {
    /// Bytes this step moves across the bus.
    pub fn bytes_to_fetch(&self, per_expert_bytes: u64) -> u64 {
        self.loads.len() as u64 * per_expert_bytes
    }

    /// The slot holding each requested expert once this plan has been executed, in the order
    /// the caller asked for them. This is what a gather kernel needs.
    pub fn slots_for(&self, requested: &[ExpertRef]) -> Vec<Slot> {
        let mut map: HashMap<ExpertRef, Slot> = HashMap::with_capacity(requested.len());
        for &(e, s) in &self.hits {
            map.insert(e, s);
        }
        for l in &self.loads {
            map.insert(l.expert, l.into_slot);
        }
        requested.iter().map(|e| map[e]).collect()
    }
}

/// Why a step could not be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// The step needs more distinct experts than the cache has slots. Unservable at any hit
    /// rate, so it is refused rather than thrashed.
    StepExceedsCapacity { needed: usize, capacity: u32 },
    /// Capacity of zero.
    ZeroCapacity,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepExceedsCapacity { needed, capacity } => write!(
                f,
                "this step activates {needed} distinct experts but the cache holds {capacity} \
                 — no policy can serve it; raise residency or lower the experts-per-token"
            ),
            Self::ZeroCapacity => write!(f, "the expert cache has no slots"),
        }
    }
}

impl std::error::Error for CacheError {}

/// Running totals, so the server can report what the cache is actually achieving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheStats {
    pub steps: u64,
    pub demands: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.demands == 0 { 0.0 } else { self.hits as f64 / self.demands as f64 }
    }
}

/// The resident expert pool.
pub struct ExpertCache {
    capacity: u32,
    /// slot -> occupant. `None` means never filled.
    occupant: Vec<Option<ExpertRef>>,
    /// expert -> slot, the reverse index so a hit is O(1).
    slot_of: HashMap<ExpertRef, Slot>,
    /// expert -> logical time of last use, for LRU.
    last_used: HashMap<ExpertRef, u64>,
    clock: u64,
    stats: CacheStats,
}

impl ExpertCache {
    /// Create a cache with `capacity` slots.
    pub fn new(capacity: u32) -> Result<Self, CacheError> {
        if capacity == 0 {
            return Err(CacheError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            occupant: vec![None; capacity as usize],
            slot_of: HashMap::with_capacity(capacity as usize),
            last_used: HashMap::with_capacity(capacity as usize),
            clock: 0,
            stats: CacheStats::default(),
        })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Zero the counters without disturbing what is resident.
    ///
    /// Measuring a *warm* cache means exactly this: keep the residency, forget the history that
    /// built it. Rebuilding the cache instead would throw the residency away and measure a cold
    /// one under a warm name, and subtracting two snapshots at the call site loses the `steps`
    /// denominator as soon as anything else shares the cache.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// Experts currently resident.
    pub fn resident_count(&self) -> usize {
        self.slot_of.len()
    }

    /// Whether `e` is in VRAM right now.
    ///
    /// 🔴 This exists so a caller can decide what to do with a miss **before** committing to it.
    /// [`Self::admit`] plans and commits in one step on purpose — a dropped plan would leave the
    /// map describing a device state that never happened — so a policy that sends some misses
    /// somewhere other than the bus has to ask first and then admit only what it kept.
    pub fn resident(&self, e: ExpertRef) -> bool {
        self.slot_of.contains_key(&e)
    }

    /// Plan and commit the transfers for one step.
    ///
    /// The cache's state is updated as though the returned plan has been executed, because the
    /// caller is expected to execute it before the next step. Splitting plan from commit would
    /// let a caller drop a plan and leave the map describing a device state that never
    /// happened — a divergence that would surface as wrong output, not an error.
    pub fn admit(&mut self, needed: &[ExpertRef]) -> Result<StepPlan, CacheError> {
        let mut distinct: Vec<ExpertRef> = needed.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() > self.capacity as usize {
            return Err(CacheError::StepExceedsCapacity {
                needed: distinct.len(),
                capacity: self.capacity,
            });
        }

        self.stats.steps += 1;
        let mut plan = StepPlan::default();
        let mut fetched_this_step: Vec<ExpertRef> = Vec::new();

        // Walk the demands in the order the router produced them, advancing recency once per
        // demand.
        //
        // 🔴 Order is load-bearing and it is not obvious. Every expert this step needs is
        // pinned, so ordering cannot change which of THEM is evicted — but it does change the
        // recency each one ends up holding, and therefore which of them is evicted several
        // steps later. An earlier version iterated a sorted, deduplicated list and disagreed
        // with the offline simulator on total misses. Two implementations of "LRU" that
        // process the same trace in different orders are two different policies.
        for &e in needed {
            self.stats.demands += 1;
            self.clock += 1;

            if let Some(&slot) = self.slot_of.get(&e) {
                self.last_used.insert(e, self.clock);
                if fetched_this_step.contains(&e) {
                    // Already fetched earlier in this step: a second read of a filled slot.
                    self.stats.hits += 1;
                } else if plan.hits.iter().any(|&(h, _)| h == e) {
                    self.stats.hits += 1;
                } else {
                    plan.hits.push((e, slot));
                    self.stats.hits += 1;
                }
                continue;
            }

            // A miss: one transfer, counted once.
            self.stats.misses += 1;
            fetched_this_step.push(e);

            let (slot, evicted) = match self.first_empty_slot() {
                Some(s) => (s, None),
                None => {
                    // Evict the least recently used expert this step does not need. The
                    // capacity check above guarantees one exists.
                    let victim = self
                        .lru_victim(&distinct)
                        .expect("capacity >= step demand guarantees an unpinned slot");
                    let s = self.slot_of.remove(&victim).expect("victim was resident");
                    self.last_used.remove(&victim);
                    self.stats.evictions += 1;
                    (s, Some(victim))
                }
            };

            self.occupant[slot as usize] = Some(e);
            self.slot_of.insert(e, slot);
            self.last_used.insert(e, self.clock);
            plan.loads.push(Load { expert: e, into_slot: slot, evicted });
        }

        Ok(plan)
    }

    fn first_empty_slot(&self) -> Option<Slot> {
        self.occupant.iter().position(Option::is_none).map(|i| i as Slot)
    }

    fn lru_victim(&self, pinned: &[ExpertRef]) -> Option<ExpertRef> {
        self.slot_of
            .keys()
            .filter(|e| !pinned.contains(e))
            .min_by_key(|e| self.last_used.get(e).copied().unwrap_or(0))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(layer: u16, expert: u16) -> ExpertRef {
        ExpertRef::new(layer, expert)
    }

    #[test]
    fn a_cold_cache_loads_everything_once() {
        let mut c = ExpertCache::new(8).unwrap();
        let need = [e(0, 0), e(0, 1), e(0, 2)];
        let plan = c.admit(&need).unwrap();
        assert!(plan.hits.is_empty());
        assert_eq!(plan.loads.len(), 3);
        assert!(plan.loads.iter().all(|l| l.evicted.is_none()));
        assert_eq!(c.stats().misses, 3);

        // Same step again: all hits, nothing moves.
        let plan = c.admit(&need).unwrap();
        assert_eq!(plan.hits.len(), 3);
        assert!(plan.loads.is_empty());
        assert_eq!(c.stats().hits, 3);
    }

    #[test]
    fn slots_are_reused_by_least_recent_use() {
        let mut c = ExpertCache::new(2).unwrap();
        c.admit(&[e(0, 0)]).unwrap();
        c.admit(&[e(0, 1)]).unwrap();
        c.admit(&[e(0, 0)]).unwrap(); // refresh 0, so 1 is now the victim
        let plan = c.admit(&[e(0, 2)]).unwrap();
        assert_eq!(plan.loads.len(), 1);
        assert_eq!(plan.loads[0].evicted, Some(e(0, 1)));
    }

    #[test]
    fn an_expert_needed_this_step_is_never_evicted_for_another() {
        // The bug this exists to prevent: filling the last slot by displacing something the
        // same step is about to read. It would not error — the gather would read overwritten
        // memory and produce quietly wrong output.
        let mut c = ExpertCache::new(3).unwrap();
        c.admit(&[e(0, 0), e(0, 1), e(0, 2)]).unwrap();
        // Now ask for two resident experts plus one new: the newcomer must displace the one
        // resident expert NOT named in this step.
        let plan = c.admit(&[e(0, 0), e(0, 1), e(0, 9)]).unwrap();
        assert_eq!(plan.loads.len(), 1);
        assert_eq!(plan.loads[0].evicted, Some(e(0, 2)));
        let evicted: Vec<_> = plan.loads.iter().filter_map(|l| l.evicted).collect();
        for needed in [e(0, 0), e(0, 1), e(0, 9)] {
            assert!(!evicted.contains(&needed), "{needed:?} was evicted while needed");
        }
    }

    #[test]
    fn a_step_larger_than_the_cache_is_refused_not_thrashed() {
        let mut c = ExpertCache::new(2).unwrap();
        let err = c.admit(&[e(0, 0), e(0, 1), e(0, 2)]).unwrap_err();
        assert_eq!(err, CacheError::StepExceedsCapacity { needed: 3, capacity: 2 });
        assert!(err.to_string().contains("no policy can serve it"));
    }

    #[test]
    fn slots_for_answers_in_the_order_asked() {
        let mut c = ExpertCache::new(8).unwrap();
        let need = [e(1, 5), e(0, 2), e(1, 5), e(3, 7)];
        let plan = c.admit(&need).unwrap();
        let slots = plan.slots_for(&need);
        assert_eq!(slots.len(), need.len());
        // A repeated expert must resolve to the same slot both times.
        assert_eq!(slots[0], slots[2]);
        // Distinct experts must occupy distinct slots.
        assert_ne!(slots[0], slots[1]);
        assert_ne!(slots[1], slots[3]);
    }

    #[test]
    fn a_repeat_within_a_step_is_one_fetch_and_then_a_hit() {
        // Three reads of one expert are three demands but only one transfer. The first read
        // pays the fetch; the other two hit the slot it filled. Charging all three as misses
        // would overstate bus traffic, which is the number the whole engine is optimising.
        let mut c = ExpertCache::new(8).unwrap();
        c.admit(&[e(0, 1), e(0, 1), e(0, 1)]).unwrap();
        let s = c.stats();
        assert_eq!(s.demands, 3);
        assert_eq!(s.misses, 1, "one transfer, not three");
        assert_eq!(s.hits, 2);
        assert_eq!(c.resident_count(), 1);
    }

    #[test]
    fn every_slot_is_used_before_anything_is_evicted() {
        let mut c = ExpertCache::new(4).unwrap();
        for i in 0..4u16 {
            c.admit(&[e(0, i)]).unwrap();
        }
        assert_eq!(c.stats().evictions, 0);
        assert_eq!(c.resident_count(), 4);
        c.admit(&[e(0, 99)]).unwrap();
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn the_cache_agrees_with_the_offline_simulator() {
        // Both implement LRU over the same trace, by different means: the simulator scans a
        // resident vector, this keeps a reverse index and a slot map. Agreement on miss count
        // is a real cross-check rather than a restatement.
        use super::super::residency::{Policy, simulate, synthetic_trace};
        let trace = synthetic_trace(150, 6, 32, 3, 0.6, 4242);
        let capacity = 24;

        let mut c = ExpertCache::new(capacity).unwrap();
        for step in &trace.steps {
            c.admit(step).unwrap();
        }
        let sim = simulate(&trace, capacity, Policy::Lru, 1).unwrap();

        assert_eq!(c.stats().demands, sim.demands);
        assert_eq!(c.stats().misses, sim.misses, "live cache and simulator disagree on misses");
        assert_eq!(c.stats().hits, sim.hits);
    }
}
