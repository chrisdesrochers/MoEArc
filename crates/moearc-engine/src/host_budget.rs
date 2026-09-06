//! How much of the model the **host** keeps in RAM.
//!
//! [`crate::memory`] answers "what lives in VRAM". This module answers the question underneath
//! it, and it is the one that actually decides whether a model is worth running: **when an
//! expert is not resident on the card, where do its bytes come from?**
//!
//! There are three answers and they are three different machines to a user:
//!
//! 1. **RAM.** The weights are in the page cache, so a miss is a PCIe copy.
//! 2. **Disk.** The weights are on NVMe, so a miss is a read *and then* a copy.
//! 3. **Nowhere.** The model is not on this machine and there is no room to fetch it.
//!
//! 🔴 **VRAM is not on that list, and that is the point of the project.** A 59 GiB model runs on
//! an 11.33 GiB card. Nothing is "too large for VRAM" here — the expert cache pages, the
//! mapping degrades, and the model still answers. So the question a user needs settled before
//! they start a download is not "does it fit the GPU" but "how much of it will be in RAM", and
//! that is a number they control rather than one we measure off their machine and impose.
//!
//! # Why an explicit budget at all
//!
//! Without one this tier is *implicit*: we `mmap` the file, the kernel decides what to keep,
//! and neither the user nor the engine can tell a RAM hit from a disk read. Both consequences
//! are bad. The user cannot answer "will this model be slow on my box" without running it. The
//! engine cannot make the one decision that matters on a miss — whether to wait for the copy or
//! to route the expert somewhere else — because it does not know what a miss costs.
//!
//! Making it explicit fixes both. [`HostBudget`] is a stated intent, [`place`] turns it into a
//! per-model verdict with wording a user can act on, and [`host_residency`] turns it into the
//! number the token loop needs.
//!
//! # Design notes
//!
//! Deliberately the same three rules as [`crate::memory`], because the two planners sit side by
//! side in the same output and a reader should not have to learn two idioms:
//!
//! 1. **Unsigned throughout, checked or saturating.** A budget cannot go negative.
//! 2. **The user's unit is bytes of model.** Not pages, not a percentage of RAM, not slots.
//! 3. **No inherited constants.** The one judgement call here — [`Reserve::DEFAULT`] — is
//!    named, documented as a judgement rather than a measurement, and printed in the rationale
//!    where the user can see it and argue with it.
//!
//! And, as there, the plan **explains itself**: every verdict carries a [`Reason`].
//!
//! # What this module does *not* claim
//!
//! 🔴 It never states how much slower the paging tier is. That number is a measurement nobody
//! here has taken, it depends on the drive, the queue depth and the routing skew, and a
//! plausible multiplier printed beside a real one is indistinguishable from a result. The
//! wording says *slower*, and stops.

use std::fmt;

use crate::memory::ModelFootprint;

// =======================================================================================
// The machine
// =======================================================================================

/// Host memory, measured before anything is loaded.
///
/// `available_bytes` is the operating system's own estimate of what can be used without
/// pushing something else out — `MemAvailable` on Linux. It is deliberately not
/// `total - used`: page cache counts as used and is reclaimable, so that subtraction
/// understates the budget by however much of the model is already cached, which is exactly
/// backwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// Free space where models are kept.
///
/// One number, for one question: is there room to *fetch* a model that is not here yet. A
/// model already on disk is never judged against it — its bytes are already spent, and telling
/// a user that a file they can see will not fit would be false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Storage {
    pub free_bytes: u64,
}

// =======================================================================================
// The budget
// =======================================================================================

/// Host memory the budget may never claim.
///
/// This is what makes it impossible to starve the machine: the reserve is subtracted from
/// `available` to form the *ceiling*, and no budget — default, flag, or key held down in the
/// interface — can be set above that ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reserve {
    /// Leave this fraction of available memory alone.
    Fraction(f64),
    /// Leave exactly this many bytes alone.
    Bytes(u64),
}

impl Reserve {
    /// 🔴 **A product judgement, not a measurement.** A fifth of what the kernel reports
    /// available is held back so that the rest of the machine — the compositor, the editor, the
    /// browser the user is watching the model in — keeps working while a model is mapped.
    ///
    /// It is a judgement rather than a guess at a physical quantity, which is the distinction
    /// that matters: there is no correct value to measure here, only a policy about how much of
    /// someone's machine a local inference tool should feel entitled to. A fifth is defensible
    /// and is stated on screen; a user who disagrees moves the budget down, and a user who
    /// wants it higher is the one case we refuse, on purpose.
    pub const DEFAULT: Self = Self::Fraction(0.20);

    fn take_from(self, available: u64) -> u64 {
        match self {
            // Round the reserve up: holding back one byte too many is the safe direction.
            Self::Fraction(f) => (available as f64 * f.clamp(0.0, 1.0)).ceil() as u64,
            Self::Bytes(b) => b.min(available),
        }
    }
}

/// Caller policy for the host tier. `default()` is a complete answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetPolicy {
    pub reserve: Reserve,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self { reserve: Reserve::DEFAULT }
    }
}

/// Where a budget's value came from. Rendered, because "we chose this" and "you chose this"
/// are different claims and only one of them is ours to defend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// Nobody asked for a number, so the ceiling was used. See [`HostBudget::ceiling`].
    Default,
    /// A number was asked for and honoured exactly.
    Requested,
    /// A number was asked for and was above the ceiling, so the ceiling was used instead.
    ///
    /// 🔴 Carried rather than silently applied. A tool that quietly rewrites what the user
    /// typed and then reports success has answered a different question from the one asked.
    Clamped { asked: u64 },
}

/// How many bytes of model weights we intend to keep in host RAM.
///
/// Not an allocation. Nothing here reserves, locks or touches a page — it is a stated intent,
/// used to classify models ([`place`]) and to tell the engine what a cache miss will cost
/// ([`host_residency`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostBudget {
    bytes: u64,
    memory: HostMemory,
    policy: BudgetPolicy,
    source: BudgetSource,
}

impl HostBudget {
    /// The most any budget may be: available memory less the policy reserve.
    pub fn ceiling(memory: HostMemory, policy: &BudgetPolicy) -> u64 {
        memory.available_bytes - policy.reserve.take_from(memory.available_bytes)
    }

    /// The budget for a user who has expressed no preference: the ceiling itself.
    ///
    /// 🔴 Deliberately the maximum, which is unusual for a default and is the honest choice
    /// here. The budget describes the **page cache**, and page cache is memory the kernel takes
    /// back the moment anything else wants it. Holding some of it back by default would buy the
    /// machine nothing it does not already have and would cost the user residency they were
    /// entitled to. The protection lives in the ceiling — which the reserve has already been
    /// taken out of, and which nothing can raise — not in a timid default underneath it.
    pub fn default_for(memory: HostMemory, policy: &BudgetPolicy) -> Self {
        Self {
            bytes: Self::ceiling(memory, policy),
            memory,
            policy: *policy,
            source: BudgetSource::Default,
        }
    }

    /// The budget for a user who asked for a number, clamped to the ceiling.
    pub fn requested(memory: HostMemory, policy: &BudgetPolicy, want: u64) -> Self {
        let ceiling = Self::ceiling(memory, policy);
        let (bytes, source) = if want > ceiling {
            (ceiling, BudgetSource::Clamped { asked: want })
        } else {
            (want, BudgetSource::Requested)
        };
        Self { bytes, memory, policy: *policy, source }
    }

    /// Move the budget to `want`, keeping the machine it was measured against.
    ///
    /// This is what a key press in the interface does, and it clamps to the ceiling **without**
    /// recording a [`BudgetSource::Clamped`].
    ///
    /// 🔴 The distinction is the whole reason this is not [`Self::requested`]. A clamp is
    /// reported because a user typed a number and did not get it — that is theirs to know. A
    /// key held down is not an assertion of a number: stepping into the ceiling is the control
    /// working, and the gauge already shows it is at the end. Routing the keys through
    /// `requested` printed *"16777216.0 TiB was asked for"* on the screen the moment anyone
    /// pressed the "maximum" key, which is a sentence about an implementation detail.
    pub fn set(self, want: u64) -> Self {
        Self { bytes: want.min(self.max_bytes()), source: BudgetSource::Requested, ..self }
    }

    /// Bytes of model weights this budget covers.
    pub fn bytes(self) -> u64 {
        self.bytes
    }

    pub fn memory(self) -> HostMemory {
        self.memory
    }

    pub fn policy(self) -> BudgetPolicy {
        self.policy
    }

    pub fn source(self) -> BudgetSource {
        self.source
    }

    /// The ceiling this budget was clamped against.
    pub fn max_bytes(self) -> u64 {
        Self::ceiling(self.memory, &self.policy)
    }

    /// Bytes held back from the budget by the reserve. Named so the interface can say what it
    /// is protecting rather than leaving a gap in a gauge unexplained.
    pub fn reserved_bytes(self) -> u64 {
        self.policy.reserve.take_from(self.memory.available_bytes)
    }

    /// The budget as a fraction of the ceiling, for a gauge. `1.0` when there is no ceiling to
    /// speak of, so a machine that reports nothing does not render an empty bar.
    pub fn fraction_of_ceiling(self) -> f64 {
        let ceiling = self.max_bytes();
        if ceiling == 0 { 1.0 } else { self.bytes as f64 / ceiling as f64 }
    }

    /// A sensible increment for a `+`/`-` control.
    ///
    /// A thirty-second of the range, rounded up to a whole GiB, so a full sweep is about thirty
    /// key presses on any machine — enough to land where you meant on a 128 GiB box, not so
    /// many that it is unusable on an 8 GiB one.
    pub fn step_bytes(self) -> u64 {
        const GIB: u64 = 1 << 30;
        (self.max_bytes() / 32).div_ceil(GIB).max(1) * GIB
    }
}

// =======================================================================================
// Placing one model against the budget
// =======================================================================================

/// What a model costs the host, and whether its bytes are here at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelBytes {
    /// The bytes the host must be able to reach: the whole mapped file.
    ///
    /// 🔴 The whole file, not the experts the VRAM plan left over. The page cache holds *file*
    /// pages, the dense weights are read from the same file at load, and the alternative —
    /// subtracting whatever the card happens to hold — makes this verdict depend on a VRAM plan
    /// that may not exist yet on a machine with no card in it. [`host_residency`] is where the
    /// finer number lives, because that is where a VRAM allocation is in hand.
    pub weights_bytes: u64,
    /// Whether the file is already on this machine.
    pub on_disk: bool,
}

/// Where a model's bytes come from on a cache miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Inside the budget: every miss is a copy from RAM.
    RunsFromRam,
    /// Past the budget: the excess is paged in from the drive on demand.
    ///
    /// 🔴 **This is a running model.** `mmap` degrades under pressure; it does not fail. A
    /// model here is slower than one in [`Self::RunsFromRam`] and is not an error, and any
    /// interface that colours it like one is lying about what this engine does.
    RunsPagesFromDisk,
    /// Not on this machine, and no room to put it.
    WillNotFit,
}

impl Tier {
    /// The words. Fixed here rather than at each call site so the plain renderer, the
    /// interface and the JSON payload cannot drift into three vocabularies for three tiers.
    pub fn label(self) -> &'static str {
        match self {
            Self::RunsFromRam => "runs from RAM",
            Self::RunsPagesFromDisk => "runs, pages from disk",
            Self::WillNotFit => "won't fit",
        }
    }

    /// Whether a model in this tier can be served at all.
    pub fn runs(self) -> bool {
        !matches!(self, Self::WillNotFit)
    }
}

/// Why a model landed in the tier it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The whole mapping fits the budget.
    WithinBudget { weights: u64, budget: u64 },
    /// Part of the mapping is past the budget and will be read from the drive.
    OverBudget { weights: u64, budget: u64, over: u64 },
    /// The model is not here and will not fit where models are kept.
    NoRoomToFetch { weights: u64, free: u64 },
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WithinBudget { weights, budget } => write!(
                f,
                "{} of weights inside a {} host budget — every cache miss is a copy from RAM",
                size(*weights),
                size(*budget)
            ),
            Self::OverBudget { weights, budget, over } => write!(
                f,
                "{} of weights against a {} host budget: {} of it is read from the drive on \
                 demand. It runs — the mapping degrades, it does not fail — just slower than \
                 the same model inside the budget",
                size(*weights),
                size(*budget),
                size(*over)
            ),
            Self::NoRoomToFetch { weights, free } => write!(
                f,
                "not on this machine, and {} will not fit in the {} free where models are kept",
                size(*weights),
                size(*free)
            ),
        }
    }
}

/// A model's host verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub tier: Tier,
    pub reason: Reason,
    /// Bytes of the mapping the budget covers.
    pub ram_bytes: u64,
    /// Bytes past it, read from the drive on demand.
    pub disk_bytes: u64,
}

impl Placement {
    /// Fraction of the mapping the budget covers, `0.0..=1.0`.
    ///
    /// The number that makes [`Tier::RunsPagesFromDisk`] readable: "past the budget" and
    /// "96% of it is in RAM" are the same fact, and only one of them tells a user whether to
    /// care.
    pub fn ram_fraction(self) -> f64 {
        let total = self.ram_bytes + self.disk_bytes;
        if total == 0 { 1.0 } else { self.ram_bytes as f64 / total as f64 }
    }
}

/// Classify one model against the budget and the drive.
///
/// The order is the whole rule, and it is short on purpose:
///
/// 1. A model that is **not here** and does not fit the free space cannot be fetched.
/// 2. Otherwise, weights **within** the budget run from RAM. The comparison is inclusive: a
///    model exactly the size of the budget is covered by it.
/// 3. Otherwise it runs and pages the excess from the drive.
///
/// 🔴 There is no VRAM step. Adding one would encode the belief this engine exists to
/// disprove.
pub fn place(model: ModelBytes, budget: HostBudget, storage: Storage) -> Placement {
    if !model.on_disk && model.weights_bytes > storage.free_bytes {
        return Placement {
            tier: Tier::WillNotFit,
            reason: Reason::NoRoomToFetch {
                weights: model.weights_bytes,
                free: storage.free_bytes,
            },
            ram_bytes: 0,
            disk_bytes: 0,
        };
    }
    let budget_bytes = budget.bytes();
    let ram_bytes = model.weights_bytes.min(budget_bytes);
    let disk_bytes = model.weights_bytes - ram_bytes;
    if disk_bytes == 0 {
        Placement {
            tier: Tier::RunsFromRam,
            reason: Reason::WithinBudget { weights: model.weights_bytes, budget: budget_bytes },
            ram_bytes,
            disk_bytes,
        }
    } else {
        Placement {
            tier: Tier::RunsPagesFromDisk,
            reason: Reason::OverBudget {
                weights: model.weights_bytes,
                budget: budget_bytes,
                over: disk_bytes,
            },
            ram_bytes,
            disk_bytes,
        }
    }
}

// =======================================================================================
// What the budget means to the engine
// =======================================================================================

/// The budget, translated into the only fact the token loop needs.
///
/// A miss on the VRAM expert cache is served from the host. Whether that costs a PCIe copy or
/// a drive read is currently invisible to the engine, which is why the miss path has exactly
/// one behaviour for two situations that differ by an order of magnitude in latency. This is
/// the number that separates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostResidency {
    /// Expert slots the budget can hold beyond the ones VRAM already has.
    pub slots: u32,
    /// What those slots cost in host RAM.
    pub bytes: u64,
    /// Slots that are neither in VRAM nor covered by the budget — the ones whose miss is a
    /// drive read.
    pub cold_slots: u32,
    /// Whether no miss can reach the drive: the budget covers every slot VRAM does not hold.
    pub covers_all_misses: bool,
}

/// How much of the expert bank the budget can keep in RAM, given what VRAM already holds.
///
/// 🔴 Expert slots only, and the dense weights are deliberately excluded. They are copied to
/// the card once at load and never read again, so counting them here would charge the steady
/// state for a cost the first token already paid. [`ModelBytes`] is the coarser question and
/// charges the whole file, because it is asked before any of this is known.
///
/// Modelling note, stated because it is an assumption and not a measurement: this treats the
/// budget as covering *whole* expert slots, ranked coldest-first by whatever admits them. It
/// says how many slots fit, not which — choosing which is [`crate::residency`]'s problem, and
/// it already ranks experts by access frequency for exactly this.
pub fn host_residency(
    budget: HostBudget,
    model: &ModelFootprint,
    vram_resident_slots: u32,
) -> HostResidency {
    if model.per_expert_bytes == 0 {
        return HostResidency { slots: 0, bytes: 0, cold_slots: 0, covers_all_misses: false };
    }
    let not_in_vram = model.total_experts.saturating_sub(vram_resident_slots);
    let affordable = (budget.bytes() / model.per_expert_bytes).min(u32::MAX as u64) as u32;
    let slots = affordable.min(not_in_vram);
    HostResidency {
        slots,
        bytes: model.per_expert_bytes.saturating_mul(slots as u64),
        cold_slots: not_in_vram - slots,
        covers_all_misses: slots == not_in_vram,
    }
}

/// Binary sizes, matching [`crate::memory`]'s so the two planners read as one output.
fn size(bytes: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    if bytes >= (1 << 30) {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.0} MiB", bytes as f64 / MIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    /// Roughly the reference machine: 91 GiB fitted, ~65 GiB available.
    fn machine() -> HostMemory {
        HostMemory { total_bytes: 91 * GIB, available_bytes: 65 * GIB }
    }

    fn budget(bytes: u64) -> HostBudget {
        HostBudget::requested(machine(), &BudgetPolicy::default(), bytes)
    }

    fn disk(free: u64) -> Storage {
        Storage { free_bytes: free }
    }

    fn here(bytes: u64) -> ModelBytes {
        ModelBytes { weights_bytes: bytes, on_disk: true }
    }

    #[test]
    fn the_default_is_the_ceiling_and_the_ceiling_leaves_the_reserve_alone() {
        let p = BudgetPolicy::default();
        let b = HostBudget::default_for(machine(), &p);
        assert_eq!(b.source(), BudgetSource::Default);
        assert_eq!(b.bytes(), b.max_bytes());
        // A fifth of available is held back, so the budget is four fifths of it.
        assert_eq!(b.bytes() + b.reserved_bytes(), machine().available_bytes);
        assert!(b.bytes() < machine().available_bytes, "something must be left for the machine");
    }

    #[test]
    fn a_budget_larger_than_the_machine_is_clamped_rather_than_honoured() {
        // The rule that makes it impossible to starve the OS. Asking for a terabyte on a 91 GiB
        // box is not an error the user needs to fix, but it must not be granted either.
        let b = budget(1024 * GIB);
        assert_eq!(b.bytes(), b.max_bytes());
        assert_eq!(b.source(), BudgetSource::Clamped { asked: 1024 * GIB });
        assert!(b.bytes() < machine().total_bytes, "never above what is fitted");
        assert!(b.bytes() < machine().available_bytes, "nor above what is available");
    }

    #[test]
    fn a_budget_above_available_but_below_total_is_still_clamped() {
        // 80 GiB is real memory on a 91 GiB box and is still not available memory. The
        // distinction is the whole reason the ceiling comes off `available`.
        let b = budget(80 * GIB);
        assert!(matches!(b.source(), BudgetSource::Clamped { .. }));
        assert_eq!(b.bytes(), b.max_bytes());
    }

    #[test]
    fn a_budget_within_the_ceiling_is_honoured_exactly() {
        let b = budget(8 * GIB);
        assert_eq!(b.bytes(), 8 * GIB);
        assert_eq!(b.source(), BudgetSource::Requested);
    }

    #[test]
    fn stepping_up_from_the_ceiling_cannot_walk_past_it() {
        let mut b = HostBudget::default_for(machine(), &BudgetPolicy::default());
        for _ in 0..64 {
            b = b.set(b.bytes() + b.step_bytes());
        }
        assert_eq!(b.bytes(), b.max_bytes());
        // And it is not reported as a clamp: nobody asked for a number here, they held a key.
        assert_eq!(b.source(), BudgetSource::Requested);
        assert_eq!(b.set(u64::MAX).source(), BudgetSource::Requested);
    }

    #[test]
    fn stepping_down_bottoms_out_at_nothing_rather_than_wrapping() {
        let mut b = HostBudget::default_for(machine(), &BudgetPolicy::default());
        for _ in 0..1_000 {
            b = b.set(b.bytes().saturating_sub(b.step_bytes()));
        }
        assert_eq!(b.bytes(), 0);
        // A zero budget is legal and means exactly one thing: everything pages from disk.
        let p = place(here(4 * GIB), b, disk(1024 * GIB));
        assert_eq!(p.tier, Tier::RunsPagesFromDisk);
        assert_eq!(p.ram_bytes, 0);
    }

    #[test]
    fn a_machine_that_reports_nothing_produces_a_zero_budget_not_a_panic() {
        let dead = HostMemory { total_bytes: 0, available_bytes: 0 };
        let b = HostBudget::default_for(dead, &BudgetPolicy::default());
        assert_eq!(b.bytes(), 0);
        assert_eq!(b.step_bytes(), GIB);
        assert_eq!(b.fraction_of_ceiling(), 1.0);
    }

    #[test]
    fn a_model_exactly_the_size_of_the_budget_runs_from_ram() {
        // The boundary, spelled out because both readings are defensible and only one of them
        // is what the comparison in `place` does.
        let b = budget(16 * GIB);
        let p = place(here(16 * GIB), b, disk(1024 * GIB));
        assert_eq!(p.tier, Tier::RunsFromRam);
        assert_eq!(p.disk_bytes, 0);
        assert_eq!(p.ram_fraction(), 1.0);

        // One byte more is the other tier, and nothing else about it changes.
        let over = place(here(16 * GIB + 1), b, disk(1024 * GIB));
        assert_eq!(over.tier, Tier::RunsPagesFromDisk);
        assert_eq!(over.disk_bytes, 1);
        assert!(over.tier.runs(), "one byte over the budget is not a failure");
    }

    #[test]
    fn a_model_far_larger_than_the_budget_still_runs() {
        // The thesis, at the host tier. 59 GiB of weights against an 8 GiB budget is the
        // ordinary case this engine is for, not an error state.
        let p = place(here(63_387_346_208), budget(8 * GIB), disk(3_000 * GIB));
        assert_eq!(p.tier, Tier::RunsPagesFromDisk);
        assert!(p.tier.runs());
        assert!(p.ram_fraction() > 0.0 && p.ram_fraction() < 0.2);
        assert!(p.reason.to_string().contains("It runs"), "{}", p.reason);
    }

    #[test]
    fn a_model_already_on_disk_is_never_judged_against_free_space() {
        // Its bytes are already spent. Telling a user that a file they can see will not fit
        // would be false, and it is the exact shape of the "too big for VRAM" claim this
        // module exists to avoid making at a different tier.
        let p = place(here(63 * GIB), budget(8 * GIB), disk(MIB));
        assert_eq!(p.tier, Tier::RunsPagesFromDisk);
    }

    #[test]
    fn a_model_that_is_not_here_needs_room_for_its_whole_file() {
        let away = |bytes| ModelBytes { weights_bytes: bytes, on_disk: false };
        let b = budget(8 * GIB);
        assert_eq!(place(away(142 * GIB), b, disk(45 * GIB)).tier, Tier::WillNotFit);
        // Exactly the free space fits: the comparison is on strictly-greater, so a model that
        // just fills the drive is a download rather than a refusal.
        assert_eq!(place(away(45 * GIB), b, disk(45 * GIB)).tier, Tier::RunsPagesFromDisk);
        assert!(!Tier::WillNotFit.runs());
    }

    #[test]
    fn every_verdict_says_why_in_a_sentence_that_names_both_numbers() {
        let b = budget(16 * GIB);
        let cases = [
            place(here(4 * GIB), b, disk(1024 * GIB)),
            place(here(59 * GIB), b, disk(1024 * GIB)),
            place(ModelBytes { weights_bytes: 512 * GIB, on_disk: false }, b, disk(45 * GIB)),
        ];
        for p in cases {
            let prose = p.reason.to_string();
            assert!(prose.contains("GiB"), "a verdict names bytes, not a category: {prose}");
            assert!(prose.len() > 40, "and it is a sentence: {prose}");
        }
    }

    /// The reference model's shape: 4,608 slots of 12.6 MiB.
    fn model() -> ModelFootprint {
        ModelFootprint {
            dense_weights_bytes: 2_460_250_368,
            per_expert_bytes: 13_219_200,
            total_experts: 4_608,
            active_experts: 144,
            kv_bytes_per_token: 73_728,
        }
    }

    #[test]
    fn the_budget_reaches_the_engine_as_a_count_of_slots() {
        let m = model();
        let r = host_residency(budget(16 * GIB), &m, 1_000);
        assert!(r.slots > 0);
        assert_eq!(r.slots + r.cold_slots, m.total_experts - 1_000);
        assert_eq!(r.bytes, r.slots as u64 * m.per_expert_bytes);
    }

    #[test]
    fn a_budget_that_covers_everything_vram_does_not_hold_means_no_miss_reaches_the_drive() {
        let m = model();
        // The whole expert bank is 58.1 GiB; a 64 GiB budget covers it several times over.
        let r = host_residency(budget(64 * GIB), &m, 1_000);
        assert!(r.covers_all_misses);
        assert_eq!(r.cold_slots, 0);
        assert_eq!(r.slots, m.total_experts - 1_000);
    }

    #[test]
    fn a_fully_resident_model_asks_nothing_of_the_host() {
        let m = model();
        let r = host_residency(budget(64 * GIB), &m, m.total_experts);
        assert_eq!(r.slots, 0);
        assert_eq!(r.cold_slots, 0);
        assert!(r.covers_all_misses, "there are no misses to cover");
    }

    #[test]
    fn a_zero_budget_leaves_every_miss_cold() {
        let m = model();
        let r = host_residency(budget(0), &m, 100);
        assert_eq!(r.slots, 0);
        assert_eq!(r.cold_slots, m.total_experts - 100);
        assert!(!r.covers_all_misses);
    }

    #[test]
    fn a_model_with_no_expert_size_is_answered_rather_than_divided_by() {
        let m = ModelFootprint { per_expert_bytes: 0, ..model() };
        assert_eq!(host_residency(budget(16 * GIB), &m, 0).slots, 0);
    }

    #[test]
    fn an_absolute_reserve_is_honoured_and_cannot_exceed_the_machine() {
        let p = BudgetPolicy { reserve: Reserve::Bytes(5 * GIB) };
        assert_eq!(HostBudget::ceiling(machine(), &p), 60 * GIB);
        // A reserve larger than the machine leaves nothing, rather than underflowing.
        let greedy = BudgetPolicy { reserve: Reserve::Bytes(1024 * GIB) };
        assert_eq!(HostBudget::ceiling(machine(), &greedy), 0);
    }
}
