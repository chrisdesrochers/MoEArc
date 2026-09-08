//! Turning "this card, this model" into settings — and being honest about how we got them.
//!
//! # The ladder
//!
//! 1. **Measured.** A profile exists for *this GPU, this model, this quantisation*. Its
//!    settings are used as they were measured.
//! 2. **Extrapolated.** A profile exists for the same model on a **different Arc card**, or for
//!    a **sibling model** on this one — same architecture, same routing width, same scale. The
//!    settings that are properties of the pairing come across; the ones that are properties of
//!    the *card* are recomputed. A starting point, and it is called that everywhere it appears.
//! 3. **Derived.** Nothing close. `moearc_engine::memory::plan` computes the split from the
//!    model's own geometry and this card's free VRAM, and
//!    `moearc_engine::memory::llama_split` states it in llama.cpp's flags. Real arithmetic,
//!    nothing measured.
//!
//! ⚠️ **One hop only.** A sibling model must have been measured on *this* card, and a different
//! card must carry *this* model. A profile that is two hops away — another model on another
//! card — falls through to derived rather than being carried twice. That is deliberate: the
//! second hop would buy `-fa` and a KV width while badging the whole answer as though someone
//! had reasoned about the pair, and the derived answer it replaces is exact arithmetic on the
//! user's own hardware and file.
//!
//! # The invariant that makes the badge mean something
//!
//! 🔴 **Only an exact match can produce a [`Origin::Measured`] field.** Not "usually" — never.
//! The moment a value is carried onto hardware or a model it was not measured on it is
//! extrapolated, whatever we believe about how well it transfers, and
//! [`only_an_exact_match_can_produce_a_measured_field`] asserts it over the whole surface. This
//! is the rule that stops a badge from decaying into decoration.
//!
//! # Two things that are recomputed even on an exact match
//!
//! * **`-ncmoe`, when this card has less free VRAM right now than the profile was measured
//!   with.** A compositor holding a gigabyte is enough. The measured value would be the first
//!   setting past the cliff, and llama.cpp's answer to that is `OUT_OF_DEVICE_MEMORY` after
//!   loading tens of gigabytes.
//! * **`-t`, when the host CPU is not the one it was measured on.** A thread count measured on
//!   a 20-core box is not a measurement about an 8-core one.

use moearc_engine::memory::{
    BlockGeometry, Context, DeviceMemory, LlamaSplit, ModelFootprint, Policy, plan_llama,
};
use serde::Serialize;

use super::coverage;
use super::schema::{ModelIdentity, Origin, Score, Setting, TuningProfile};
use super::store::Store;
use crate::source::{DeviceRow, ModelCard};

/// What this machine's CPU is, for the one setting that depends on it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct HostCpu {
    pub model: Option<String>,
    /// Physical cores. `None` when the machine would not say.
    pub physical_cores: Option<u32>,
    pub logical_cores: Option<u32>,
}

/// The share of a machine's cores the fallback claims when no profile measured `-t` on this
/// CPU.
///
/// 🔴 **One named constant, so changing the fallback for every unmeasured machine on earth is
/// one edit.** It is a policy, not a measurement, and [`HostCpu::derived_threads`] documents
/// what it costs on the one CPU this project has actually benchmarked.
pub const FALLBACK_THREAD_FRACTION: f64 = 0.50;

/// [`FALLBACK_THREAD_FRACTION`] of `cores`, never below one thread.
pub fn fallback_threads(cores: u32) -> u32 {
    ((cores as f64 * FALLBACK_THREAD_FRACTION).round() as u32).max(1)
}

impl HostCpu {
    /// The thread count to suggest on a CPU no profile has measured.
    ///
    /// # Why there is a fallback at all
    ///
    /// 🔴 **Not choosing is measurably the worst option.** llama.cpp does not scale `-t` to the
    /// machine: `common_cpu_get_num_math()` mis-reads Arrow Lake's hybrid topology and settles
    /// on **4 threads** on a 20-core part. On this project's own box that default cost
    /// **2.09×** on gpt-oss-120B — **14.11 ± 0.33 tok/s against 29.54 ± 0.04 at `-t 16`**, with
    /// the two arms interleaved (`-t 16,4,16,4,16`) so page-cache state is a common-mode error
    /// rather than the result. `bench/tuning-profiles.md`, *"`-t` — the most valuable flag"*.
    ///
    /// # What we ship, and what it costs
    ///
    /// ⚠️ **We ship half the cores. The measured optimum on the one CPU we have tested is
    /// ~80% of them — 16 of 20 — and half costs roughly a fifth of the throughput `-t 16`
    /// reaches on that machine.** The in-tree `-t 8` readings bracket it: Llama-4-Scout
    /// 14.27 → 18.52 (**−23%**), Qwen3-30B 68.20 → 73.04 (−6.6%), Qwen3.6-35B 53.18 → 54.04
    /// (−1.6%). The cost depends entirely on how much work is host-side, which is `-ncmoe`'s
    /// business, not this constant's.
    ///
    /// **50% is deliberately conservative for silicon nobody has benchmarked, and is not
    /// claimed to be optimal.** 16-of-20 is one measurement on one hybrid Intel part; half the
    /// cores leaves the other half of an unknown machine alone, the loss is bounded, and a user
    /// who measures moves it up. Every screen labels the value [`Origin::Derived`], never
    /// measured.
    ///
    /// # Never `nproc`
    ///
    /// 🔴 `-t 20` on this 20-core box **lost on five of the six models where threads matter at
    /// all** — gpt-oss-120B −8.3%, gpt-oss-20B −10.8%, olmoe at full offload −10.8%,
    /// Qwen3.6-35B −7.3%, Qwen3-30B −5.6%, with only Llama-4-Scout a tie — and its cells were
    /// three to four times noisier besides (stddev 1.0–2.2 against 0.0–0.5 below it). **8
    /// P-cores and 12 E-cores are not 20 equal cores**, and a thread pool that assumes they are
    /// runs at the pace of its slowest member. Physical cores are the base for the fraction for
    /// the same reason hyperthread siblings are not counted: they share load/store units, and a
    /// memory-bound expert gather does not get faster for being given two threads on one core.
    pub fn derived_threads(&self) -> Option<u32> {
        let cores = self.physical_cores.or(self.logical_cores).filter(|n| *n > 0)?;
        Some(fallback_threads(cores))
    }

    /// Whether a profile was measured on a machine with the same core count.
    fn matches(&self, cores: Option<u32>) -> Option<bool> {
        match (self.physical_cores, cores) {
            (Some(a), Some(b)) => Some(a == b),
            // Either side unknown: we cannot claim a match, and saying so is the answer.
            _ => None,
        }
    }
}

/// How the settings below were arrived at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "basis", rename_all = "snake_case")]
pub enum Basis {
    /// Benchmarked on this card, with this model, at this quantisation.
    Measured { profile: String, measured_at: String },
    /// Measured on the same model at a different quantisation on this card.
    OtherQuant { profile: String, quant: String },
    /// Measured on the same model on a different Arc card.
    OtherCard { profile: String, gpu: String, vram_bytes: u64 },
    /// Measured on a structurally identical model.
    SiblingModel { profile: String, model: String },
    /// Computed from the model's geometry and this card's free VRAM. Nothing was run.
    Derived,
    /// Not even derivable: the planner refused, and the reason is its own.
    Unplannable { reason: String },
    /// No inference device to plan against.
    NoDevice,
}

impl Basis {
    /// The badge. Three states, exactly as the ladder above.
    pub fn origin(&self) -> Origin {
        match self {
            Self::Measured { .. } => Origin::Measured,
            Self::OtherQuant { .. } | Self::OtherCard { .. } | Self::SiblingModel { .. } => {
                Origin::Extrapolated
            }
            Self::Derived => Origin::Derived,
            Self::Unplannable { .. } | Self::NoDevice => Origin::Untuned,
        }
    }

    /// The short word for a table cell.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Unplannable { .. } | Self::NoDevice => "none",
            other => other.origin().label(),
        }
    }

    /// The sentence under the table. 🔴 Every non-measured basis says *in words* that it was
    /// not measured; the glyph is a shorthand for people who already know, not the claim.
    pub fn sentence(&self) -> String {
        match self {
            Self::Measured { profile, measured_at } => {
                format!(
                    "measured on this card with this model — profile `{profile}`, {measured_at}"
                )
            }
            Self::OtherQuant { profile, quant } => format!(
                "a starting point, not a measurement: carried from `{profile}`, measured on \
                 this model at {quant}. A different quantisation has a different footprint, so \
                 the offload split was recomputed for this file"
            ),
            Self::OtherCard { profile, gpu, vram_bytes } => format!(
                "a starting point, not a measurement: carried from `{profile}`, measured on \
                 {gpu} with {} of VRAM. The offload split was recomputed for this card; \
                 nothing here was run on it",
                crate::format::bytes(*vram_bytes)
            ),
            Self::SiblingModel { profile, model } => format!(
                "a starting point, not a measurement: carried from `{profile}`, measured on \
                 `{model}` — same architecture, same routing width, same scale. The offload \
                 split was recomputed for this model; nothing here was run on it"
            ),
            Self::Derived => "computed from this model's geometry and this card's free VRAM. \
                              Real arithmetic, and nothing was ever run — expect to tune from \
                              here rather than to stop here"
                .to_string(),
            Self::Unplannable { reason } => format!("no settings could be computed: {reason}"),
            Self::NoDevice => "no Level Zero device to plan against".to_string(),
        }
    }

    /// The profile this basis cites, for a report that wants to name it.
    pub fn profile_id(&self) -> Option<&str> {
        match self {
            Self::Measured { profile, .. }
            | Self::OtherQuant { profile, .. }
            | Self::OtherCard { profile, .. }
            | Self::SiblingModel { profile, .. } => Some(profile),
            _ => None,
        }
    }
}

/// One flag, with the standing of the value in it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Flag {
    pub flag: &'static str,
    pub value: String,
    pub origin: Origin,
    /// What the flag is for, in one clause. Printed beside it, because `-ncmoe` explains
    /// itself to nobody.
    pub purpose: &'static str,
}

/// The answer: what we would run, and how much we stand behind each part of it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolved {
    pub model: String,
    /// The badge, from [`Basis::origin`].
    pub origin: Origin,
    pub basis: Basis,
    pub threads: Option<Setting<u32>>,
    pub n_cpu_moe: Option<Setting<u32>>,
    pub n_gpu_layers: Option<Setting<u32>>,
    pub ctx_size: Option<Setting<u32>>,
    pub batch_size: Option<Setting<u32>>,
    pub ubatch_size: Option<Setting<u32>>,
    pub kv_cache_type: Option<Setting<String>>,
    pub flash_attn: Option<Setting<bool>>,
    pub extra_args: Vec<String>,
    /// The number a tweak is climbed from, **only** when it came from an exact match and
    /// passes `PROTOCOL` §5.
    ///
    /// 🔴 Never carried across hardware. PROTOCOL §0: absolute throughput does not reproduce
    /// across machines and this project publishes shape, not absolutes. A baseline borrowed
    /// from another card would be a number to beat that means nothing here.
    pub baseline: Option<Score>,
    /// Fraction of the expert bank this configuration holds resident.
    pub bank_resident: Option<f64>,
    /// What that fraction is worth, when the model's own coverage curve is available.
    pub coverage: Option<coverage::Estimate>,
    /// Things the reader has to know before acting on the above.
    pub caveats: Vec<String>,
}

impl Resolved {
    /// The weakest origin among the settings actually present.
    ///
    /// Distinct from [`Self::origin`], which is the *basis*. They differ in the interesting
    /// case: an exact match whose `-ncmoe` had to be recomputed because this card has less
    /// free VRAM right now is still a measured profile, and that one field is not.
    pub fn weakest_field_origin(&self) -> Origin {
        self.flags().into_iter().fold(Origin::Measured, |acc, f| acc.weakest(f.origin))
    }

    /// Every flag, in the order a person reads a command line.
    pub fn flags(&self) -> Vec<Flag> {
        let mut out = Vec::new();
        let mut push = |flag, purpose, s: Option<(String, Origin)>| {
            if let Some((value, origin)) = s {
                out.push(Flag { flag, value, origin, purpose });
            }
        };
        push(
            "-ngl",
            "layers on the GPU",
            self.n_gpu_layers.as_ref().map(|s| (s.value.to_string(), s.origin)),
        );
        push(
            "-ncmoe",
            "MoE blocks whose experts stay in host RAM",
            self.n_cpu_moe.as_ref().map(|s| (s.value.to_string(), s.origin)),
        );
        push("-t", "host threads", self.threads.as_ref().map(|s| (s.value.to_string(), s.origin)));
        push(
            "-c",
            "context length, in tokens",
            self.ctx_size.as_ref().map(|s| (s.value.to_string(), s.origin)),
        );
        push(
            "-b",
            "logical batch",
            self.batch_size.as_ref().map(|s| (s.value.to_string(), s.origin)),
        );
        push(
            "-ub",
            "physical batch",
            self.ubatch_size.as_ref().map(|s| (s.value.to_string(), s.origin)),
        );
        push(
            "--cache-type-k",
            "KV cache width",
            self.kv_cache_type.as_ref().map(|s| (s.value.clone(), s.origin)),
        );
        push(
            "--cache-type-v",
            "KV cache width",
            self.kv_cache_type.as_ref().map(|s| (s.value.clone(), s.origin)),
        );
        // 🔴 `-fa` takes a value in this engine — `common_arg({"-fa", "--flash-attn"},
        // "[on|off|auto]", ...)`. A bare `-fa` does not mean "on": it swallows whatever token
        // follows it and then refuses it, or runs off the end of the command line. The value
        // is always printed, and `off` is printed too when a profile measured it that way,
        // because a pasteable command that omits a measured setting is not the command that
        // was measured.
        push(
            "-fa",
            "flash attention",
            self.flash_attn
                .as_ref()
                .map(|s| ((if s.value { "on" } else { "off" }).to_string(), s.origin)),
        );
        out
    }

    /// The command line, ready to paste.
    pub fn command(&self, binary: &str, model_path: Option<&str>) -> String {
        let mut parts = vec![binary.to_string()];
        if let Some(p) = model_path {
            parts.push("-m".to_string());
            parts.push(shell_quote(p));
        }
        for f in self.flags() {
            parts.push(f.flag.to_string());
            parts.push(f.value);
        }
        parts.extend(self.extra_args.iter().cloned());
        parts.join(" ")
    }
}

/// Quote a path for a shell only when it needs it. A quoted path that did not need quoting
/// reads as an error in the path.
fn shell_quote(s: &str) -> String {
    if s.chars().all(|c| c.is_ascii_alphanumeric() || "._-/=:+@".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// A normalised key for a GPU, so a profile keeps matching across driver versions.
///
/// `"Intel(R) Arc(TM) Pro B60 Graphics"` → `"arc-pro-b60"`. Vendor words and the marketing
/// suffix come off; the model designation is what identifies the card.
pub fn gpu_key(name: &str) -> String {
    const NOISE: [&str; 6] = ["intel", "r", "tm", "graphics", "gpu", "series"];
    let words: Vec<String> = name
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty() && !NOISE.contains(w))
        .map(str::to_string)
        .collect();
    words.join("-")
}

fn identity_of(card: &ModelCard) -> ModelIdentity {
    ModelIdentity {
        id: card.id.clone(),
        quant: card.quant.clone(),
        file_bytes: Some(card.file_bytes),
        moe_blocks: card.moe_blocks,
        experts_per_block: card.experts_per_block,
        active_experts_per_block: card.active_experts_per_block,
        expert_slots_total: card.expert_slots_total,
        parameters: Some(card.parameters),
    }
}

fn same_model(a: &ModelIdentity, b: &ModelIdentity) -> bool {
    a.id.eq_ignore_ascii_case(&b.id)
}

/// Rank two candidate profiles: newest measurement first, then the closest VRAM.
fn better(a: &TuningProfile, b: &TuningProfile, want: &ModelIdentity, vram: u64) -> bool {
    let key = |p: &TuningProfile| {
        (
            p.measured_at.clone(),
            // Negated distance so "larger is better" holds for the whole tuple.
            u64::MAX - p.hardware.vram_bytes.abs_diff(vram),
            ((p.model.name_affinity(want)) * 1000.0) as u64,
        )
    };
    key(a) > key(b)
}

/// Work out what to run.
pub fn resolve(
    store: &Store,
    device: Option<&DeviceRow>,
    card: &ModelCard,
    cpu: &HostCpu,
    ctx: Option<u32>,
) -> Resolved {
    let want = identity_of(card);
    let policy = Policy::default();

    let Some(device) = device.filter(|d| d.is_inference_target()) else {
        return empty(card, Basis::NoDevice);
    };
    let key = gpu_key(&device.name);

    // Pick the best profile at each rung, strongest rung first.
    let mut exact = None;
    let mut other_quant = None;
    let mut other_card = None;
    let mut sibling = None;
    for p in store.profiles() {
        let this_card = p.hardware.gpu_key.eq_ignore_ascii_case(&key);
        let this_model = same_model(&p.model, &want);
        let slot = if this_card && this_model && p.model.quant == want.quant {
            &mut exact
        } else if this_card && this_model {
            &mut other_quant
        } else if this_model {
            &mut other_card
        } else if this_card && p.model.is_same_family(&want) {
            &mut sibling
        } else {
            continue;
        };
        if slot.is_none_or(|c: &TuningProfile| better(p, c, &want, device.free_bytes)) {
            *slot = Some(p);
        }
    }

    // The derivation is both the fallback *and* the safety check every other branch is
    // measured against, so a failure here costs every setting. It does **not** cost the
    // recorded score: a baseline is a number somebody measured on this card with this model,
    // and it stays true while another process is holding the device. Losing it would make
    // `--candidate` — a comparison between two recorded numbers, needing no hardware at all —
    // unavailable for a reason that has nothing to do with it.
    let split = match derive(device, card, &policy, ctx) {
        Ok(s) => s,
        Err(reason) => {
            let mut r = empty(card, Basis::Unplannable { reason });
            r.baseline = exact.and_then(TuningProfile::baseline).cloned();
            return r;
        }
    };

    let mut resolved = match (exact, other_quant, other_card, sibling) {
        (Some(p), ..) => from_exact(p, card, device, cpu, ctx, split),
        (None, Some(p), ..) => from_nearby(
            p,
            card,
            cpu,
            split,
            Basis::OtherQuant { profile: p.id.clone(), quant: p.model.quant.clone() },
        ),
        (None, None, Some(p), _) => from_nearby(
            p,
            card,
            cpu,
            split,
            Basis::OtherCard {
                profile: p.id.clone(),
                gpu: p.hardware.gpu.clone(),
                vram_bytes: p.hardware.vram_bytes,
            },
        ),
        (None, None, None, Some(p)) => from_nearby(
            p,
            card,
            cpu,
            split,
            Basis::SiblingModel { profile: p.id.clone(), model: p.model.id.clone() },
        ),
        _ => from_derived(card, cpu, split),
    };
    annotate(&mut resolved, card, cpu);
    resolved
}

/// The sentences that belong beside a setting, decided from what actually reached the flags.
///
/// 🔴 Kept out of [`from_derived`] deliberately. It is the base case every other rung is built
/// on, and an exact match overwrites some of these fields with measured values — a paragraph
/// explaining a derivation that no longer applies is worse than no paragraph. Each note below
/// is emitted only while the field it explains is still ours.
fn annotate(r: &mut Resolved, card: &ModelCard, cpu: &HostCpu) {
    if r.flags().is_empty() {
        return; // No settings were produced at all; `basis.sentence()` has already said why.
    }
    if r.n_cpu_moe.as_ref().is_some_and(|s| !s.origin.is_measured()) {
        r.caveats.push(quant_note(card));
    }
    if let (Some(n), Some(c)) = (&r.n_cpu_moe, &r.ctx_size) {
        r.caveats.push(context_coupling_note(n.value, c.value));
    }
    if let Some(t) = r.threads.as_ref().filter(|s| s.origin == Origin::Derived)
        && let Some(cores) = cpu.physical_cores.or(cpu.logical_cores)
    {
        r.caveats.push(threads_note(cores, t.value));
    }
    let fa_is_ours = r.flash_attn.as_ref().is_some_and(|s| !s.origin.is_measured());
    let kv_is_ours = r.kv_cache_type.as_ref().is_some_and(|s| !s.origin.is_measured());
    if fa_is_ours || kv_is_ours {
        r.caveats.push(FA_AND_KV_NOTE.to_string());
    }
}

/// Why `-ncmoe` points the way it does for *this* quantisation.
fn quant_note(card: &ModelCard) -> String {
    if experts_belong_on_the_host(&card.quant) {
        format!(
            "🔴 `-ncmoe {}` puts **every** block's experts in host RAM, and for {} that is the \
             measured direction rather than a shortage of VRAM. A/B/A/B inside one process on \
             gpt-oss-20b: all experts on the CPU 43.42 ± 0.08 tok/s against 36.36 ± 0.01 with all \
             of them on the card — 1.19× on a model that fits entirely in this card's VRAM \
             (bench/tuning-profiles.md). Q4_K and Q3_K go the other way, so this is a \
             per-quantisation rule and not a global one. ⚠️ It was measured on Arc + SYCL with two \
             gpt-oss models and nothing here was run on *this* pairing; why it happens was never \
             profiled.",
            card.moe_blocks, card.quant
        )
    } else {
        format!(
            "`-ncmoe` above is the lowest this planner will stand behind, and for {} lower is \
             faster: Qwen3-30B measures 74.08 tok/s at `-ncmoe 18` against 58.58 at 30, monotone \
             across the range (bench/tuning-profiles.md). ⚠️ The headroom behind our floor is a \
             stated guess rather than a measurement, so the true floor may be a block or two below \
             it — llama.cpp answers one block too few with a clean OUT_OF_DEVICE_MEMORY, after \
             loading the whole model.",
            card.quant
        )
    }
}

/// The one thing a user must not do with a split: raise `-c` and leave `-ncmoe` alone.
fn context_coupling_note(n_cpu_moe: u32, ctx: u32) -> String {
    format!(
        "🔴 `-ncmoe {n_cpu_moe}` is the floor for the `-c {}` printed beside it, not for this \
         model. Context and expert slots come out of the same pool and the floor moves with depth \
         — Qwen3-30B measured 18 blocks offloaded at depth 0, 21 at 8K and 28 at 32K \
         (bench/tuning-profiles.md). Raising `-c` by hand without raising `-ncmoe` is how a run \
         dies with OUT_OF_DEVICE_MEMORY *after* loading tens of gigabytes; ask for the context you \
         want with `--ctx` and the split is recomputed for it.",
        crate::format::count(ctx as i64)
    )
}

/// What the fallback thread count is, and what it is not.
fn threads_note(cores: u32, threads: u32) -> String {
    format!(
        "`-t {threads}` is {:.0}% of this machine's {cores} cores — a deliberately conservative \
         fallback for a CPU nobody has benchmarked, not an optimum. 🔴 What it protects against is \
         llama.cpp's own default of **4 threads whatever your core count**, which cost 2.09× here \
         (14.11 against 29.54 tok/s on gpt-oss-120b). ⚠️ On the one CPU this project has measured \
         the optimum is 16 of 20 — about 80% — and half the cores gives up roughly a fifth of it \
         (Llama-4-Scout 14.27 → 18.52 tok/s between `-t 8` and `-t 16`). Raise it if you measure, \
         but never to `nproc`: `-t 20` on that 20-core hybrid part lost on five of six models and \
         was three to four times noisier as well.",
        FALLBACK_THREAD_FRACTION * 100.0
    )
}

/// The KV width every plan in this project is computed at, and the only one it recommends on
/// Arc. Named so the planner's assumption and the emitted flag cannot drift apart.
const KV_CACHE_TYPE: &str = "f16";

const FA_AND_KV_NOTE: &str = "`-fa on` and an f16 KV cache are stated rather than inherited. Flash attention is worth \
      1.72× at 8K (Qwen3-30B, 52.4 tok/s against 30.52 with `-fa off`) and llama.cpp's `auto` \
      already resolves to on — it is written out because `-fa off` is advice users copy from other \
      backends. 🔴 Quantised KV is the trap: `q8_0` measured **−19%** on this card (42.87 against \
      52.97), bought back exactly **one** `-ncmoe` block, and one block below that it hard-aborts \
      inside the SYCL backend instead of returning an OOM a tool can catch. f16 is also the width \
      the split above was planned at.";

/// Whether this quantisation's experts are measurably faster **on the host** than on the card.
///
/// 🔴 **The direction of `-ncmoe` flips with the quantisation, and a single derivation rule
/// would be wrong for half the catalogue.** Both halves are measured, on this Arc B580 with the
/// SYCL backend — `bench/tuning-profiles.md`, *"`-ncmoe` — the direction depends on the
/// quantisation, not the model size"*:
///
/// * **MXFP4 — put *every* expert on the CPU.** The cleanest evidence is an A/B/A/B inside a
///   single process on gpt-oss-20B (`-ncmoe 24,0,24,0`, which controls for drift and read only
///   22 MiB from disk): every expert on the **CPU** measured **43.42 ± 0.08 tok/s** against
///   **36.36 ± 0.01** with every expert on the **GPU**. **1.19×, on a model whose 11.28 GiB
///   fits entirely inside this card's VRAM.** The 10-point sweep between the two ends is
///   monotone in the same direction (+21%) and the conclusion survives at depth.
/// * **Q4_K / Q3_K — offload as little as VRAM allows.** Qwen3-30B is monotone the other way:
///   **74.08 tok/s at `-ncmoe 18` against 58.58 at 30**, a 1.26× spread. Qwen3.6-35B agrees
///   (55.58 at 22 against 50.85 at 28).
///
/// ⚠️ **What is measured is *that* it happens, not *why*.** The plausible reading is that the
/// SYCL MXFP4 matmul path is weak relative to this CPU's, but the kernel was never profiled, so
/// that is a hypothesis. The rule is therefore carried as a property of **this backend at this
/// quantisation**, and everything it produces is still [`Origin::Derived`]: a measurement of
/// gpt-oss on a B580 is not a measurement of the pairing in front of us.
fn experts_belong_on_the_host(quant: &str) -> bool {
    // ggml's own spelling, which `moearc-model` lowercases before it reaches a `ModelCard`.
    // Matched by prefix so a future `mxfp4_*` variant lands here too, and nothing else does.
    quant.to_ascii_lowercase().starts_with("mxfp4")
}

/// The engine's own answer for this card and model — the whole of the fallback's arithmetic.
///
/// [`plan_llama`] takes the model's geometry and the VRAM free *right now*, refuses in
/// arithmetic a plan it cannot stand behind, and returns `-ncmoe` and `-c` from **one**
/// calculation, so the two flags cannot contradict each other.
///
/// 🔴 **The context is part of the answer, not a detail beside it.** The `-ncmoe` floor moves
/// with depth — Qwen3-30B measured 18 blocks at depth 0, **21 at 8K and 28 at 32K** — so a
/// split is only the floor for the `-c` it was planned with. A caller who asks for a context
/// gets a split planned for that context; a caller who asks for nothing gets
/// [`Context::Largest`], deliberately, because `fit.rs` already refuses to invent a default
/// context and two modules quietly assuming different ones would put two contradictory numbers
/// on one screen.
fn derive(
    device: &DeviceRow,
    card: &ModelCard,
    policy: &Policy,
    ctx: Option<u32>,
) -> Result<LlamaSplit, String> {
    let memory = DeviceMemory { total_bytes: device.total_bytes, free_bytes: device.free_bytes };
    let footprint = ModelFootprint {
        dense_weights_bytes: card.dense_weights_bytes,
        per_expert_bytes: card.per_expert_bytes,
        total_experts: card.expert_slots_total,
        active_experts: card.expert_slots_active,
        kv_bytes_per_token: card.kv_bytes_per_token,
    };
    let geometry =
        BlockGeometry { moe_blocks: card.moe_blocks, experts_per_block: card.experts_per_block };
    let want = ctx.map_or(Context::Largest, Context::Tokens);
    let (_, mut split) =
        plan_llama(memory, &footprint, geometry, policy, want).map_err(|e| e.to_string())?;

    // 🔴 Applied *after* the plan succeeded, never instead of it. The plan that succeeded put
    // MORE on the card than this does, so moving every block's experts to the host spends
    // strictly less VRAM than the arithmetic already found room for: it cannot turn a feasible
    // plan into an OUT_OF_DEVICE_MEMORY.
    if experts_belong_on_the_host(&card.quant) {
        return Ok(LlamaSplit::all_experts_on_host(
            geometry,
            host_expert_context(
                memory,
                &footprint,
                geometry,
                policy,
                card,
                want,
                split.context_tokens,
            ),
        ));
    }

    // 🔴 The trained-context cap applies to EVERY path, not just the MXFP4 one.
    //
    // `Context::Largest` asks the planner for the most KV the card can hold, and it answers
    // honestly — but a KV cache longer than the model was trained for allocates pages whose
    // answers do not mean anything. Before this, `moearc info olmoe-1b-7b-0924-instruct`
    // printed `-c 47360` for a 4,096-token model, directly beside its own "Planned split"
    // saying 4,096: the same screen contradicting itself.
    //
    // Re-planning (rather than clamping the number) is what keeps `-c` and `-ncmoe` agreeing.
    // They come out of one pool, and a `-c` clamped after the fact would leave `-ncmoe` sized
    // for a context we are no longer asking for. If the re-plan fails we clamp instead, which
    // is safe in the one direction that matters: the surviving `-ncmoe` puts MORE on the host
    // than the shorter context needs, so it spends strictly less VRAM than the arithmetic
    // already found room for and cannot turn a feasible plan into an OUT_OF_DEVICE_MEMORY.
    let trained = card.trained_context_tokens.max(1);
    if split.context_tokens > trained {
        if let Ok((_, replanned)) =
            plan_llama(memory, &footprint, geometry, policy, Context::Tokens(trained))
        {
            return Ok(replanned);
        }
        split.context_tokens = trained;
    }
    Ok(split)
}

/// The context to pair with an all-experts-on-the-host split.
///
/// 🔴 Without this the recommendation would be self-defeating. `plan` fills expert slots
/// greedily, so the split it returns spends the card on residency and lands the context on the
/// policy floor — and then the MXFP4 rule takes every one of those slots away again, leaving
/// gigabytes of VRAM committed to nothing while `-c` still reads 2,048. The bytes the experts
/// were going to occupy are exactly what buys the context this configuration is worth: the
/// measured gpt-oss-20B recommendation reaches **65K tokens at 36.88 tok/s** precisely because
/// `-ncmoe 24` leaves 10,097 MiB free.
///
/// So it is re-planned with the bank held at its floor. Two things make the answer conservative
/// rather than optimistic, which is the direction to be wrong in:
///
/// * The floor is `active_experts`, not zero — [`Policy::max_resident_experts`] cannot go below
///   the slots one token touches. Those slots are budgeted here and then not placed, so the
///   card has *more* free than this context was planned against.
/// * It is capped at the context the model was trained for. A KV cache longer than the model
///   can attend over is memory spent on nothing.
///
/// A caller who asked for a specific context is answered with it, not with this.
fn host_expert_context(
    memory: DeviceMemory,
    footprint: &ModelFootprint,
    geometry: BlockGeometry,
    policy: &Policy,
    card: &ModelCard,
    want: Context,
    fallback: u32,
) -> u32 {
    let floor_only = Policy { max_resident_experts: Some(0), ..*policy };
    plan_llama(memory, footprint, geometry, &floor_only, want)
        .map_or(fallback, |(a, _)| a.context_tokens)
        .min(card.trained_context_tokens.max(1))
}

fn empty(card: &ModelCard, basis: Basis) -> Resolved {
    Resolved {
        model: card.id.clone(),
        origin: basis.origin(),
        caveats: vec![basis.sentence()],
        basis,
        threads: None,
        n_cpu_moe: None,
        n_gpu_layers: None,
        ctx_size: None,
        batch_size: None,
        ubatch_size: None,
        kv_cache_type: None,
        flash_attn: None,
        extra_args: Vec::new(),
        baseline: None,
        bank_resident: None,
        coverage: None,
    }
}

/// The base case: the planner's split, and nothing borrowed.
fn from_derived(card: &ModelCard, cpu: &HostCpu, split: LlamaSplit) -> Resolved {
    let mut r = empty(card, Basis::Derived);
    r.caveats.clear();
    r.n_cpu_moe = Some(Setting::derived(split.n_cpu_moe));
    r.n_gpu_layers = Some(Setting::derived(ALL_LAYERS));
    r.ctx_size = Some(Setting::derived(split.context_tokens));
    // The width the plan was computed at, so it is a consequence of the arithmetic rather
    // than a preference. `crate::fit::KvPrecision` is the other half of this statement.
    r.kv_cache_type = Some(Setting::derived(KV_CACHE_TYPE.to_string()));
    // 🔴 Stated, not left to `auto`. `auto` does resolve to on, so this changes nothing about
    // what runs — it changes what a user can be talked out of. See [`FA_AND_KV_NOTE`].
    r.flash_attn = Some(Setting::derived(true));
    r.threads = cpu.derived_threads().map(Setting::derived);
    r.bank_resident = bank_resident(card, split.n_cpu_moe);
    r.caveats.push(NGL_NOTE.to_string());
    if split.slots_dropped_to_whole_blocks > 0 {
        r.caveats.push(whole_block_note(&split));
    }
    if r.threads.is_none() {
        r.caveats.push(
            "this machine would not report a core count, so no thread count is suggested. \
             🔴 Do not leave it unset: llama.cpp defaults to 4 threads, which cost 2.1× on \
             this project's own 20-core box."
                .to_string(),
        );
    }
    r
}

/// An exact match: measured, with two recomputations that can override it.
fn from_exact(
    p: &TuningProfile,
    card: &ModelCard,
    device: &DeviceRow,
    cpu: &HostCpu,
    ctx: Option<u32>,
    split: LlamaSplit,
) -> Resolved {
    // 🔴 The context has to be settled *before* the split, not after. `-ncmoe` and `-c`
    // compete for the same bytes, so adopting a profile's measured 4,096-token context on top
    // of a split planned for the largest that fits would emit two flags that contradict each
    // other — and llama.cpp would discover the contradiction by running out of device memory.
    let (split, adopted_ctx, ctx_refused) = match (ctx, p.settings.ctx_size) {
        (None, Some(want)) if want != split.context_tokens => {
            match derive(device, card, &Policy::default(), Some(want)) {
                Ok(replanned) => (replanned, true, false),
                Err(_) => (split, false, true),
            }
        }
        _ => (split, false, false),
    };

    let mut r = from_derived(card, cpu, split);
    r.basis = Basis::Measured { profile: p.id.clone(), measured_at: p.measured_at.clone() };
    r.origin = r.basis.origin();

    // -ncmoe. The measured value wins unless this card has less free VRAM right now than the
    // plan needs, in which case it is the first setting past the cliff.
    if let Some(measured) = p.settings.n_cpu_moe {
        if measured >= split.n_cpu_moe {
            r.n_cpu_moe = Some(Setting::measured(measured));
        } else {
            r.n_cpu_moe = Some(Setting::derived(split.n_cpu_moe));
            r.caveats.push(format!(
                "🔴 the profile's `-ncmoe {measured}` puts more of the expert bank on the card \
                 than this run's plan has room for: with {} free and {} of dense weights, {} \
                 is the lowest this planner will stand behind, so it is used instead. \
                 ⚠️ The headroom behind that figure is a stated guess rather than a \
                 measurement, so it may be one block more conservative than the card needs — \
                 but the other direction is not symmetric: llama.cpp answers one block too few \
                 with OUT_OF_DEVICE_MEMORY, after loading the whole model.",
                crate::format::bytes(device.free_bytes),
                crate::format::bytes(card.dense_weights_bytes),
                split.n_cpu_moe
            ));
        }
    }

    // -t. A thread count is a property of the CPU, not of the card.
    match (p.settings.threads, cpu.matches(p.hardware.physical_cores)) {
        (Some(t), Some(true)) => r.threads = Some(Setting::measured(t)),
        (Some(t), Some(false)) => {
            r.caveats.push(format!(
                "the measured `-t {t}` was taken on a {}-core host and this machine has {}, so \
                 it is not carried; {} is derived from this machine's core count instead.",
                p.hardware.physical_cores.unwrap_or(0),
                cpu.physical_cores.unwrap_or(0),
                r.threads.as_ref().map_or(0, |s| s.value)
            ));
        }
        (Some(t), None) => {
            r.threads = Some(Setting::extrapolated(t));
            r.caveats.push(
                "the profile does not record the CPU it was measured on, so its thread count \
                 cannot be confirmed against this machine."
                    .to_string(),
            );
        }
        (None, _) => {}
    }

    // -c. A context the user typed is answered by the planner, not by the profile.
    match (ctx, p.settings.ctx_size) {
        (Some(asked), Some(measured)) if asked != measured => {
            r.caveats.push(format!(
                "the profile was measured at {} tokens of context and {} were asked for. \
                 Context and expert slots come out of the same pool, so the split above is \
                 recomputed rather than measured.",
                crate::format::count(measured as i64),
                crate::format::count(asked as i64)
            ));
        }
        (None, Some(measured)) if adopted_ctx => {
            r.ctx_size = Some(Setting::measured(measured));
        }
        (None, Some(measured)) if ctx_refused => {
            r.caveats.push(format!(
                "the profile was measured at {} tokens of context and that does not fit \
                 beside this model's experts in the VRAM free right now, so {} — what does \
                 fit — is planned instead.",
                crate::format::count(measured as i64),
                crate::format::count(r.ctx_size.as_ref().map_or(0, |s| s.value) as i64)
            ));
        }
        _ => {}
    }

    carry(&mut r, p, Origin::Measured);
    r.baseline = p.baseline().cloned();
    if r.baseline.is_none() && p.score.is_some() {
        r.caveats.push(format!(
            "🔴 this profile's own score is not a measurement — {} — so there is no baseline to \
             climb from. PROTOCOL §5.",
            p.score.as_ref().map_or(String::new(), |s| s.describe())
        ));
    }
    attach_coverage(&mut r, p, card);
    r
}

/// A profile from nearby: nothing it carries is a measurement *here*.
fn from_nearby(
    p: &TuningProfile,
    card: &ModelCard,
    cpu: &HostCpu,
    split: LlamaSplit,
    basis: Basis,
) -> Resolved {
    let mut r = from_derived(card, cpu, split);
    r.origin = basis.origin();
    // Not repeated into `caveats`: the renderers print `basis.sentence()` in the confidence
    // position, and one sentence printed twice on one screen reads as a rendering fault.
    r.basis = basis;

    // 🔴 `-ncmoe` is deliberately NOT carried. It is the setting most tightly coupled to the
    // exact card and the exact file, and it is the one whose floor is a crash. The planner's
    // value for *this* card and *this* model is strictly better information than a number
    // measured against different bytes.
    if let Some(measured) = p.settings.n_cpu_moe {
        r.caveats.push(format!(
            "`-ncmoe {measured}` from that profile is not carried across — it is the setting \
             whose floor is a crash. {} is computed for this card and this file instead.",
            split.n_cpu_moe
        ));
    }
    if let Some(t) = p.settings.threads
        && cpu.matches(p.hardware.physical_cores) == Some(true)
    {
        r.threads = Some(Setting::extrapolated(t));
    }
    carry(&mut r, p, Origin::Extrapolated);
    // 🔴 No baseline. PROTOCOL §0: absolute throughput does not reproduce across machines,
    // and a number to beat that was measured elsewhere is worse than none.
    r.baseline = None;
    if p.baseline().is_some() {
        r.caveats.push(
            "that profile's throughput figure is not carried across: absolute throughput is an \
             artefact of one machine (PROTOCOL §0). Measure this configuration to get a \
             baseline for this box."
                .to_string(),
        );
    }
    r
}

/// Bring across the settings that are neither card- nor CPU-specific, at `floor` or weaker.
fn carry(r: &mut Resolved, p: &TuningProfile, floor: Origin) {
    let s = &p.settings;
    if let Some(v) = s.kv_cache_type.clone() {
        r.kv_cache_type = Some(Setting { value: v, origin: floor });
    }
    if let Some(v) = s.flash_attn {
        r.flash_attn = Some(Setting { value: v, origin: floor });
    }
    if let Some(v) = s.batch_size {
        r.batch_size = Some(Setting { value: v, origin: floor });
    }
    if let Some(v) = s.ubatch_size {
        r.ubatch_size = Some(Setting { value: v, origin: floor });
    }
    if let Some(v) = s.n_gpu_layers {
        r.n_gpu_layers = Some(Setting { value: v, origin: floor });
        r.caveats.retain(|c| c != NGL_NOTE);
    }
    r.extra_args = s.extra_args.clone();
}

/// Attach the coverage estimate, and only when the curve describes *this* model.
///
/// 🔴 The guard is PROTOCOL §9 in code. A curve taken on one model's routing and read for
/// another is the mistake that made `docs/hardware-sizing.md` optimistic by 9 to 15 points and
/// had to be retracted.
fn attach_coverage(r: &mut Resolved, p: &TuningProfile, card: &ModelCard) {
    let Some(curve) = &p.coverage else { return };
    if !curve.model.eq_ignore_ascii_case(&card.id) {
        return;
    }
    let Some(fraction) = r.bank_resident else { return };
    let Some(estimate) = curve.at(fraction) else { return };
    if !estimate.within_measured_range {
        r.caveats.push(format!(
            "{:.1}% of the expert bank is resident, outside the {} this model's coverage curve \
             was measured over — the coverage figures are the nearest measured point, not an \
             extrapolation.",
            fraction * 100.0,
            curve
                .measured_span()
                .map(|(lo, hi)| format!("{:.0}–{:.0}%", lo * 100.0, hi * 100.0))
                .unwrap_or_default()
        ));
    }
    r.coverage = Some(estimate);
}

/// llama.cpp's idiom for "every layer".
///
/// 🔴 The *conclusion* is derived — [`plan_llama`] only succeeds when the dense weights fit on
/// the card, so every layer does belong on it. The *number* is a convention: llama.cpp takes
/// any value at or above the layer count to mean all of them, and 99 is what its own
/// documentation uses. It is not a measurement and [`NGL_NOTE`] says so on screen.
const ALL_LAYERS: u32 = 99;

const NGL_NOTE: &str = "`-ngl 99` means \"every layer\" — llama.cpp's own idiom for it, not a tuned number. That \
     every layer fits *is* derived: the planner refuses a card the dense weights do not fit on.";

fn whole_block_note(split: &LlamaSplit) -> String {
    format!(
        "the plan found room for {} expert slots and llama.cpp offloads whole blocks, so {} \
         slots' worth of card goes unused. That is margin against the `-ncmoe` floor, not waste.",
        crate::format::count(split.slots_planned as i64),
        crate::format::count(split.slots_dropped_to_whole_blocks as i64)
    )
}

fn bank_resident(card: &ModelCard, n_cpu_moe: u32) -> Option<f64> {
    if card.expert_slots_total == 0 {
        return None;
    }
    let on_gpu = card.moe_blocks.saturating_sub(n_cpu_moe) * card.experts_per_block;
    Some(on_gpu as f64 / card.expert_slots_total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{Backend, DeviceSource, StubCatalog, StubDeviceSource};

    fn store() -> Store {
        Store::from_json(crate::tuning::store::fixture::FILE, None)
    }

    fn b580() -> DeviceRow {
        StubDeviceSource
            .detect()
            .unwrap()
            .devices
            .into_iter()
            .find(DeviceRow::is_inference_target)
            .expect("the fixture has a usable Arc card")
    }

    fn cpu() -> HostCpu {
        HostCpu {
            model: Some("Intel Core Ultra 9 285K".into()),
            physical_cores: Some(20),
            logical_cores: Some(20),
        }
    }

    /// The four real GGUF geometries, read out of the files on 2026-09-05. The interface's
    /// own four-model fixture has `moe_blocks: 1`, which makes `-ncmoe` a two-valued dial and
    /// tests nothing about block rounding.
    fn find(id: &str) -> ModelCard {
        StubCatalog::as_measured()
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("no fixture model {id}"))
    }

    /// A profile file rewritten to describe whichever fixture model the test needs.
    fn store_for(card: &ModelCard, extra: &str) -> Store {
        let json = format!(
            r#"{{"schema":1,"profiles":[{{
              "id":"fixture/{id}",
              "hardware":{{"gpu":"Intel Arc B580 Graphics","gpu_key":"arc-b580",
                          "vram_bytes":12884901888,"cpu":"Intel Core Ultra 9 285K",
                          "physical_cores":20}},
              "model":{{"id":"{id}","quant":"{quant}","moe_blocks":{blocks},
                       "experts_per_block":{per},"active_experts_per_block":{act},
                       "expert_slots_total":{slots},"parameters":{params}}},
              "settings":{{"threads":16,"n_cpu_moe":{ncmoe},"flash_attn":true,
                          "kv_cache_type":"f16"{extra}}},
              "score":{{"metric":"decode_tokens_per_second","depth_tokens":0,
                       "mean":28.5,"stddev":0.2,"runs":5}},
              "measured_at":"2026-09-06"}}]}}"#,
            id = card.id,
            quant = card.quant,
            blocks = card.moe_blocks,
            per = card.experts_per_block,
            act = card.active_experts_per_block,
            slots = card.expert_slots_total,
            params = card.parameters,
            // High enough that it can never be below the planner's floor, so the test is
            // about provenance rather than about this card's free VRAM on the day.
            ncmoe = card.moe_blocks,
            extra = extra,
        );
        let s = Store::from_json(&json, None);
        assert!(s.load_error().is_none(), "{:?}", s.load_error());
        assert_eq!(s.len(), 1, "{:?}", s.rejected());
        s
    }

    #[test]
    fn nothing_at_all_still_produces_a_runnable_answer_and_calls_it_derived() {
        let card = find("gpt-oss-120b");
        let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
        assert_eq!(r.origin, Origin::Derived);
        assert_eq!(r.basis, Basis::Derived);
        let n = r.n_cpu_moe.as_ref().expect("a derived -ncmoe is still an -ncmoe");
        assert_eq!(n.origin, Origin::Derived);
        assert!(n.value <= card.moe_blocks);
        // Half of 20, from `FALLBACK_THREAD_FRACTION`. Deliberately not 20: `-t 20` lost on
        // five of six models on this very CPU, and deliberately not the measured 16 either,
        // because nothing was measured on *this* machine. See `HostCpu::derived_threads`.
        assert_eq!(r.threads.as_ref().unwrap().value, 10);
        assert_eq!(r.threads.as_ref().unwrap().origin, Origin::Derived);
        assert!(r.baseline.is_none(), "nothing was measured, so there is nothing to beat");
        assert!(r.command("llama-server", None).contains("-ncmoe"));
    }

    #[test]
    fn an_exact_match_is_measured_and_says_which_profile() {
        let card = find("gpt-oss-120b");
        let r = resolve(&store_for(&card, ""), Some(&b580()), &card, &cpu(), None);
        assert_eq!(r.origin, Origin::Measured);
        assert_eq!(r.n_cpu_moe.as_ref().unwrap().origin, Origin::Measured);
        assert_eq!(r.threads.as_ref().unwrap(), &Setting::measured(16u32));
        assert_eq!(r.baseline.as_ref().unwrap().mean, 28.5);
        assert!(r.basis.sentence().contains("measured on this card"));
    }

    #[test]
    fn only_an_exact_match_can_produce_a_measured_field() {
        // 🔴 The invariant the whole badge rests on, asserted over every field of every
        // non-exact basis.
        let card = find("gpt-oss-120b");
        let elsewhere = store_for(&card, "").profiles()[0].clone();
        let mut other_gpu = elsewhere.clone();
        other_gpu.hardware.gpu_key = "arc-pro-b60".into();
        other_gpu.hardware.gpu = "Intel Arc Pro B60 Graphics".into();
        other_gpu.hardware.vram_bytes = 25_769_803_776;
        let mut other_quant = elsewhere.clone();
        other_quant.model.quant = "q8_0".into();
        let mut sibling = elsewhere;
        sibling.model.id = format!("{}-instruct", card.id);

        for p in [other_gpu, other_quant, sibling] {
            let json = serde_json::to_string(&crate::tuning::schema::ProfileFile {
                schema: 1,
                generated_at: None,
                profiles: vec![p],
            })
            .unwrap();
            let store = Store::from_json(&json, None);
            assert_eq!(store.len(), 1, "{:?}", store.rejected());
            let r = resolve(&store, Some(&b580()), &card, &cpu(), None);
            assert_eq!(r.origin, Origin::Extrapolated, "{:?}", r.basis);
            for f in r.flags() {
                assert!(
                    !f.origin.is_measured(),
                    "{} came back measured from {:?}",
                    f.flag,
                    r.basis
                );
            }
            assert!(r.baseline.is_none(), "a score never crosses a machine: {:?}", r.basis);
            assert!(
                r.basis.sentence().contains("not a measurement"),
                "the words have to say it too: {}",
                r.basis.sentence()
            );
        }
    }

    #[test]
    fn a_measured_ncmoe_that_no_longer_fits_is_recomputed_and_the_reason_is_named() {
        // The compositor case: the profile was measured with the card idle, and today
        // something else is holding a slice of it.
        let card = find("gpt-oss-120b");
        let store = store_for(&card, "");
        // A gigabyte short, not a third of the card: the plan still has to succeed, or the
        // test would be about the planner refusing rather than about the recomputation.
        let mut squeezed = b580();
        squeezed.free_bytes -= 1 << 30;
        let mut profile = store.profiles()[0].clone();
        profile.settings.n_cpu_moe = Some(0); // everything on the card, measured when it fit
        let json = serde_json::to_string(&crate::tuning::schema::ProfileFile {
            schema: 1,
            generated_at: None,
            profiles: vec![profile],
        })
        .unwrap();
        let r = resolve(&Store::from_json(&json, None), Some(&squeezed), &card, &cpu(), None);
        assert_eq!(r.origin, Origin::Measured, "the profile still matches this card and model");
        assert_eq!(
            r.n_cpu_moe.as_ref().unwrap().origin,
            Origin::Derived,
            "but this one field is not measured any more"
        );
        assert_ne!(r.weakest_field_origin(), Origin::Measured);
        assert!(r.caveats.iter().any(|c| c.contains("OUT_OF_DEVICE_MEMORY")), "{:?}", r.caveats);
    }

    #[test]
    fn a_thread_count_does_not_cross_to_a_different_cpu() {
        let card = find("gpt-oss-120b");
        let store = store_for(&card, "");
        let small = HostCpu {
            model: Some("something smaller".into()),
            physical_cores: Some(8),
            logical_cores: Some(16),
        };
        let r = resolve(&store, Some(&b580()), &card, &small, None);
        assert_eq!(r.threads.as_ref().unwrap().value, 4, "this machine's cores, not theirs");
        assert_eq!(r.threads.as_ref().unwrap().origin, Origin::Derived);
        assert!(r.caveats.iter().any(|c| c.contains("20-core host")), "{:?}", r.caveats);
    }

    #[test]
    fn asking_for_a_different_context_recomputes_the_split_rather_than_reusing_it() {
        let card = find("gpt-oss-120b");
        let store = store_for(&card, ",\"ctx_size\":4096");
        let r = resolve(&store, Some(&b580()), &card, &cpu(), Some(32_768));
        assert!(
            r.caveats.iter().any(|c| c.contains("same pool")),
            "the tradeoff is named: {:?}",
            r.caveats
        );
        assert_eq!(r.ctx_size.as_ref().unwrap().origin, Origin::Derived);
    }

    #[test]
    fn a_coverage_curve_never_leaves_the_model_it_was_measured_on() {
        // PROTOCOL §9, as a test. The curve is attached to gpt-oss and the resolution is for
        // a different model that matches structurally.
        let card = find("gpt-oss-120b");
        let mut p = store_for(&card, "").profiles()[0].clone();
        p.coverage = Some(coverage::fixture::gpt_oss_120b());
        p.model.id = format!("{}-instruct", card.id);
        p.coverage.as_mut().unwrap().model = p.model.id.clone();
        let json = serde_json::to_string(&crate::tuning::schema::ProfileFile {
            schema: 1,
            generated_at: None,
            profiles: vec![p],
        })
        .unwrap();
        let r = resolve(&Store::from_json(&json, None), Some(&b580()), &card, &cpu(), None);
        assert!(matches!(r.basis, Basis::SiblingModel { .. }), "{:?}", r.basis);
        assert!(r.coverage.is_none(), "a curve is a property of one model's routing");
    }

    #[test]
    fn a_coverage_curve_for_this_model_is_read_and_reported() {
        let card = find("gpt-oss-120b");
        let mut p = store_for(&card, "").profiles()[0].clone();
        let mut curve = coverage::fixture::gpt_oss_120b();
        curve.model = card.id.clone();
        p.coverage = Some(curve);
        p.settings.n_cpu_moe = Some(card.moe_blocks / 2); // half the bank resident
        let json = serde_json::to_string(&crate::tuning::schema::ProfileFile {
            schema: 1,
            generated_at: None,
            profiles: vec![p],
        })
        .unwrap();
        let r = resolve(&Store::from_json(&json, None), Some(&b580()), &card, &cpu(), None);
        let e = r.coverage.expect("this model's own curve applies to it");
        assert!(e.code < e.prose);
        assert!((0.0..=1.0).contains(&e.worst_case_miss));
    }

    #[test]
    fn no_device_produces_no_settings_rather_than_a_plausible_default() {
        let card = find("gpt-oss-120b");
        let r = resolve(&store(), None, &card, &cpu(), None);
        assert_eq!(r.basis, Basis::NoDevice);
        assert!(r.flags().is_empty(), "an absent card must not yield a command line");
        assert_eq!(r.origin, Origin::Untuned);
    }

    #[test]
    fn an_integrated_gpu_is_not_a_planning_target() {
        // The device that "succeeded and lied" -- 85.6 GiB of system RAM reported as VRAM.
        let card = find("gpt-oss-120b");
        let igpu = StubDeviceSource
            .detect()
            .unwrap()
            .devices
            .into_iter()
            .find(|d| d.unusable.is_some())
            .expect("the fixture keeps the refused iGPU");
        assert_eq!(igpu.backend, Backend::LevelZero, "it is Level Zero and still not a target");
        let r = resolve(&Store::empty(), Some(&igpu), &card, &cpu(), None);
        assert_eq!(r.basis, Basis::NoDevice);
    }

    #[test]
    fn the_gpu_key_survives_a_driver_rename() {
        assert_eq!(gpu_key("Intel Arc B580 Graphics"), "arc-b580");
        assert_eq!(gpu_key("Intel(R) Arc(TM) B580 Graphics"), "arc-b580");
        assert_eq!(gpu_key("Intel(R) Arc(TM) Pro B60 Graphics"), "arc-pro-b60");
        assert_ne!(gpu_key("Intel Arc B580 Graphics"), gpu_key("Intel Arc A770 Graphics"));
    }

    #[test]
    fn the_command_line_is_pasteable_and_quotes_only_when_it_must() {
        let card = find("gpt-oss-120b");
        let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
        let plain = r.command("llama-server", Some("/zfs/models/gpt-oss.gguf"));
        assert!(plain.contains("-m /zfs/models/gpt-oss.gguf"), "{plain}");
        assert!(!plain.contains('\''), "a path that needs no quoting gets none: {plain}");
        let spaced = r.command("llama-server", Some("/my models/gpt oss.gguf"));
        assert!(spaced.contains("'/my models/gpt oss.gguf'"), "{spaced}");
    }

    #[test]
    fn every_flag_that_reaches_a_command_line_carries_an_origin() {
        let card = find("gpt-oss-120b");
        for store in [Store::empty(), store_for(&card, "")] {
            let r = resolve(&store, Some(&b580()), &card, &cpu(), None);
            assert!(!r.flags().is_empty());
            for f in r.flags() {
                assert!(!f.purpose.is_empty(), "{} explains itself", f.flag);
                assert_ne!(f.origin, Origin::Untuned, "an untuned setting is not emitted");
            }
        }
    }
    #[test]
    fn the_fallback_thread_count_is_half_the_cores_and_never_nproc() {
        // 🔴 The constant, asserted at the boundary rather than through a resolution, so the
        // one edit that changes it is visible as a failing test rather than as a silent
        // behaviour change on every unmeasured machine on earth.
        assert_eq!(FALLBACK_THREAD_FRACTION, 0.50);
        assert_eq!(fallback_threads(20), 10);
        assert_eq!(fallback_threads(8), 4);
        assert_eq!(fallback_threads(1), 1, "a single-core box still gets a thread");
        assert_eq!(fallback_threads(3), 2, "odd core counts round rather than truncate to zero");
        for cores in [1u32, 2, 4, 6, 8, 12, 16, 20, 24, 64, 128, 192] {
            let t = fallback_threads(cores);
            assert!(t >= 1, "{cores} cores produced {t} threads");
            assert!(t < cores.max(2), "{cores} cores must never derive nproc: {t}");
        }
    }

    #[test]
    fn a_machine_with_no_measured_profile_derives_every_flag_it_emits() {
        // The fallback, end to end. Nothing is measured, so nothing may render as measured --
        // and the answer still has to be runnable rather than absent.
        let card = find("qwen3-30b-a3b");
        let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
        assert_eq!(r.origin, Origin::Derived);
        assert_eq!(r.weakest_field_origin(), Origin::Derived);
        for f in r.flags() {
            assert_eq!(f.origin, Origin::Derived, "{} is not derived", f.flag);
        }
        let cmd = r.command("llama-server", None);
        for expected in ["-ngl 99", "-ncmoe", "-t 10", "-c ", "--cache-type-k f16", "-fa on"] {
            assert!(cmd.contains(expected), "{expected} missing from {cmd}");
        }
    }

    #[test]
    fn flash_attention_is_on_and_the_flag_carries_its_value() {
        // 🔴 `-fa` takes `[on|off|auto]` in this engine. A bare `-fa` swallows the next token
        // as its argument, so a command line that ends in one is not a command line.
        let card = find("gpt-oss-120b");
        let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
        assert_eq!(r.flash_attn, Some(Setting::derived(true)));
        let cmd = r.command("llama-server", Some("/m.gguf"));
        assert!(cmd.contains("-fa on"), "{cmd}");
        assert!(!cmd.ends_with("-fa"), "a bare -fa is not a value: {cmd}");
        assert!(
            r.caveats.iter().any(|c| c.contains("1.72×")),
            "what -fa is worth is stated: {:?}",
            r.caveats
        );
    }

    #[test]
    fn the_kv_cache_is_f16_and_q8_0_is_never_offered() {
        // Measured -19% on this card, buying back exactly one -ncmoe block and then aborting
        // rather than returning an OOM. It is not a default, and it is not an option here.
        for id in ["gpt-oss-120b", "qwen3-30b-a3b", "olmoe-1b-7b-0924-instruct"] {
            let card = find(id);
            let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
            assert_eq!(r.kv_cache_type, Some(Setting::derived("f16".to_string())), "{id}");
            assert!(!r.command("llama-server", None).contains("q8_0"), "{id}");
        }
    }

    #[test]
    fn mxfp4_puts_every_expert_on_the_host_and_q4_k_does_the_opposite() {
        // 🔴 The measurement that makes a single derivation rule wrong for half the catalogue.
        // gpt-oss fits far better on this card than Qwen3 does, and gets *more* offloaded.
        let mxfp4 = find("gpt-oss-120b");
        let r = resolve(&Store::empty(), Some(&b580()), &mxfp4, &cpu(), None);
        assert_eq!(
            r.n_cpu_moe.as_ref().unwrap().value,
            mxfp4.moe_blocks,
            "every block's experts belong in host RAM at MXFP4"
        );
        assert_eq!(r.n_cpu_moe.as_ref().unwrap().origin, Origin::Derived, "measured, elsewhere");
        assert_eq!(r.bank_resident, Some(0.0));
        assert!(r.caveats.iter().any(|c| c.contains("1.19×")), "{:?}", r.caveats);
        // 🔴 And the VRAM the experts vacated has to reach the context, or the recommendation
        // frees gigabytes and then spends them on nothing. The measured gpt-oss-20B config is
        // worth 65K tokens precisely because -ncmoe 24 leaves 10,097 MiB free.
        let ctx = r.ctx_size.as_ref().unwrap();
        assert!(ctx.value > 2_048, "the freed card has to buy context: {}", ctx.value);
        assert!(
            ctx.value <= mxfp4.trained_context_tokens,
            "and never past what the model was trained for: {} > {}",
            ctx.value,
            mxfp4.trained_context_tokens
        );
        // A context the caller asked for is still the callers, not the planners.
        let asked = resolve(&Store::empty(), Some(&b580()), &mxfp4, &cpu(), Some(4_096));
        assert_eq!(asked.ctx_size.as_ref().unwrap().value, 4_096);
        assert_eq!(asked.n_cpu_moe.as_ref().unwrap().value, mxfp4.moe_blocks);

        let q4 = find("qwen3-30b-a3b");
        let r = resolve(&Store::empty(), Some(&b580()), &q4, &cpu(), None);
        let n = r.n_cpu_moe.as_ref().unwrap().value;
        assert!(n < q4.moe_blocks, "Q4_K keeps what it can on the card: {n}");
        assert!(r.bank_resident.unwrap() > 0.0);
        assert!(r.caveats.iter().any(|c| c.contains("74.08")), "{:?}", r.caveats);
    }

    /// 🔴 A derived `-c` must never exceed the model's trained context on ANY path.
    ///
    /// Regression: the cap lived inside the MXFP4 branch only, so a non-MXFP4 model took
    /// `Context::Largest`'s answer verbatim. `moearc info olmoe-1b-7b-0924-instruct` printed
    /// `-c 47360` for a 4,096-token model, on the same screen as its own "Planned split"
    /// saying 4,096 — one panel contradicting the other.
    #[test]
    fn a_derived_context_never_exceeds_what_the_model_was_trained_for() {
        for id in ["olmoe-1b-7b-0924-instruct", "qwen3-30b-a3b", "gpt-oss-120b"] {
            let card = find(id);
            let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
            let ctx = r.ctx_size.as_ref().expect("a derived plan always names a context");
            assert!(
                ctx.value <= card.trained_context_tokens,
                "{id} derived -c {} past a trained context of {}",
                ctx.value,
                card.trained_context_tokens
            );
        }
    }

    #[test]
    fn the_split_names_the_context_it_is_the_floor_for() {
        // The measured trap: Qwen3-30B needs 18 blocks offloaded at depth 0 and 28 at 32K. A
        // user who takes the -ncmoe and raises -c by hand OOMs after loading the whole model.
        let card = find("qwen3-30b-a3b");
        let shallow = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), None);
        let deep = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), Some(32_768));
        assert!(
            deep.n_cpu_moe.as_ref().unwrap().value > shallow.n_cpu_moe.as_ref().unwrap().value,
            "a deeper context has to cost expert blocks: {:?} then {:?}",
            shallow.n_cpu_moe,
            deep.n_cpu_moe
        );
        for r in [&shallow, &deep] {
            assert!(
                r.caveats.iter().any(|c| c.contains("floor for the `-c")),
                "the coupling is named: {:?}",
                r.caveats
            );
        }
    }

    #[test]
    fn no_fallback_value_can_ever_render_as_measured() {
        // 🔴 The invariant, from the fallback's side: with nothing measured anywhere, no field
        // and no rendered flag may come back `measured` for any model or any context.
        for id in
            ["gpt-oss-120b", "qwen3-30b-a3b", "qwen3.6-35b-a3b-ud", "olmoe-1b-7b-0924-instruct"]
        {
            let card = find(id);
            for ctx in [None, Some(2_048), Some(4_096)] {
                let r = resolve(&Store::empty(), Some(&b580()), &card, &cpu(), ctx);
                assert!(!r.origin.is_measured(), "{id} at {ctx:?}");
                assert!(!r.weakest_field_origin().is_measured(), "{id} at {ctx:?}");
                assert!(r.baseline.is_none(), "{id}: nothing was run, so nothing may be climbed");
                for f in r.flags() {
                    assert!(!f.origin.is_measured(), "{id} at {ctx:?}: {} is measured", f.flag);
                }
            }
        }
    }

    #[test]
    fn the_built_in_measurements_reach_the_models_on_this_machine() {
        // The other half of the ladder, against the file that actually ships. Every model
        // `bench/tuning-profiles.md` measured must resolve to a measured basis with the
        // measured thread count, on a host with the same core count it was measured on.
        let store = Store::from_json(
            crate::tuning::store::BUILTIN.expect("the built-in set is compiled in"),
            None,
        );
        for (id, threads) in [("gpt-oss-120b", 16u32), ("qwen3-30b-a3b", 16)] {
            let card = find(id);
            let r = resolve(&store, Some(&b580()), &card, &cpu(), None);
            assert_eq!(r.origin, Origin::Measured, "{id}: {:?}", r.basis);
            assert_eq!(r.threads, Some(Setting::measured(threads)), "{id}");
            assert!(r.basis.profile_id().unwrap().starts_with("arc-b580/"), "{id}");
        }
    }
}
