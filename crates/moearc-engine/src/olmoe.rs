//! OLMoE, one decode step, on the device.
//!
//! This is the integration layer: it reads a GGUF with `moearc-model`, uploads every tensor to
//! the card, and issues the `moearc-kernels` calls that turn one token id into one logit
//! vector. It contains no arithmetic of its own — every operation below is a kernel that was
//! already checked against a CPU twin and, for the dequantisers, against llama.cpp itself.
//!
//! # Where the graph came from
//!
//! Transcribed from llama.cpp's `llama_model_olmoe::graph` (`src/models/olmoe.cpp`) and
//! `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`), not from the tensor names. Four
//! details are not guessable from the names, and each is wrong in an interesting way if
//! assumed:
//!
//! - **QK-norm spans the whole `n_embd`-wide vector**, applied after the Q/K projections and
//!   before the reshape into heads and before RoPE. Per-head normalisation is the natural
//!   guess, raises no error, and degrades the output subtly.
//! - **RoPE is NeoX-style.** `llama_model_rope_type` puts `LLM_ARCH_OLMOE` in the
//!   `LLAMA_ROPE_TYPE_NEOX` arm, and the loader prints `rope type = 2`. The pairs are
//!   `(i, i + n_dims/2)`, not `(2i, 2i+1)`.
//! - **The router softmaxes over all experts before the top-k**, and the selected weights are
//!   the raw probabilities: `build_moe_ffn` is called with `norm_w = false`, so they are **not**
//!   renormalised to sum to one.
//! - **`w_scale` is `hparams.expert_weights_scale`, which OLMoE never sets**, so it keeps its
//!   `0.0f` default and `build_moe_ffn`'s `if (w_scale != 0.0f && w_scale != 1.0f)` guard skips
//!   the scaling entirely.
//!
//! # Shape conventions
//!
//! GGUF stores dimensions fastest-varying first, so a weight `[d0, d1]` is `d1` rows of `d0`
//! contiguous elements and `d0` is the reduction axis. Every matvec below therefore passes
//! `n_rows = dims[1]`, `n_cols = dims[0]`.
//!
//! # Expert residency
//!
//! The expert banks are **not** uploaded at load. A pool of [`Residency`]-many slots is
//! allocated, and each block's step asks [`crate::cache::ExpertCache`] which of the eight
//! experts the router named are already there. Misses are staged from the memory-mapped file —
//! `MappedModel::expert` hands back the exact bytes with no copy — into the slots the cache
//! assigned, **before** the block computes anything. That ordering is the whole ballgame:
//! `runtime.rs` spells out why a matmul against a slot still being filled produces plausible
//! wrong output rather than an error.
//!
//! One slot holds one *(block, expert)* pair across all three banks, so the pool is three
//! parallel arrays of device buffers indexed by the same slot number. Each array's slot is
//! sized to the largest that bank reaches in any block — this file quantises `ffn_down_exps` at
//! Q6_K in 8 of 16 blocks and Q4_K in the rest, and a slot has to hold either.
//!
//! # What is still slow, and known to be
//!
//! Three things, all deliberate and all measured — run the `olmoe_profile` example for the
//! current breakdown:
//!
//! - **Prompt tokens go through the single-token decode path one at a time.** There is no
//!   batched prefill; there is one code path to be correct instead of two.
//! - **A miss is a synchronous host-to-device copy with nothing overlapped behind it.** On a
//!   cold pool this is the largest single phase; on a warm one it nearly vanishes, which is why
//!   the profile example throws away its first tokens before reading the counters.
//! - **The router's choice is read back to the host once per block**, so the host can decide
//!   what to stage. It is a device round trip per block per token, and at ~13 us it is now
//!   under one percent of a step — the reason to remove it would be to drive the gather from
//!   the device, not the round trip itself.
//!
//! What is *no longer* true: the queue is asynchronous and in-order, so the kernels below
//! submit and return rather than waiting one at a time. That is what makes the ordering of
//! `stage` before compute a correctness property of this file rather than an accident of every
//! call synchronising.

use moearc_kernels::{Context, DeviceBuffer, KernelError, KvType, QuantType, RopeKind};
use moearc_model::gguf::Value;
use moearc_model::tensors::{ExpertBank, MappedModel, TensorView, names};
use moearc_model::{ModelError, ModelInfo};

use crate::cache::{CacheError, CacheStats, ExpertCache, Load, Slot, StepPlan};
use crate::kv::{KvError, PagedKvCache, SeqId};
use crate::memory::{self, PlanError};
use crate::profile;
use crate::residency::ExpertRef;

/// The one sequence a [`Model`] tracks. Batching is not implemented.
const SEQ: SeqId = 1;

/// Tokens per KV page. It only has to be small enough that a sequence's unfilled tail is cheap.
const PAGE_TOKENS: usize = 32;

/// The KV cache element type.
///
/// 🔴 f16 on purpose, matching llama.cpp's default `type_k`/`type_v`. f32 would be more
/// accurate, which is exactly why it is not used: this pass exists to be compared against
/// llama.cpp block by block, and a cache that rounds differently puts a difference into every
/// attention output that has nothing to do with the graph.
const KV: KvType = KvType::F16;

/// Why the engine could not load or run a model.
#[derive(Debug)]
pub enum EngineError {
    Model(ModelError),
    Kernel(KernelError),
    Kv(KvError),
    Cache(CacheError),
    Plan(PlanError),
    /// The file is structurally fine but this engine cannot run it.
    Unsupported(String),
}

impl From<ModelError> for EngineError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}
impl From<KernelError> for EngineError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}
impl From<KvError> for EngineError {
    fn from(e: KvError) -> Self {
        Self::Kv(e)
    }
}
impl From<CacheError> for EngineError {
    fn from(e: CacheError) -> Self {
        Self::Cache(e)
    }
}
impl From<PlanError> for EngineError {
    fn from(e: PlanError) -> Self {
        Self::Plan(e)
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(e) => write!(f, "model file: {e}"),
            Self::Kernel(e) => write!(f, "device: {e}"),
            Self::Kv(e) => write!(f, "kv cache: {e}"),
            Self::Cache(e) => write!(f, "expert cache: {e}"),
            Self::Plan(e) => write!(f, "memory plan: {e}"),
            Self::Unsupported(m) => write!(f, "unsupported model: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// The model's geometry, read from the GGUF header — never assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub arch: String,
    pub n_block: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// Channels of each head that rotate — `n_rot` in llama.cpp. Defaults to `head_dim`.
    pub n_rot: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
}

/// A float key, whatever width the writer chose.
fn f32_key(model: &MappedModel, key: &str) -> Option<f32> {
    match model.header().get(key)? {
        Value::F32(v) => Some(*v),
        Value::F64(v) => Some(*v as f32),
        _ => None,
    }
}

fn u64_key_opt(model: &MappedModel, key: &str) -> Option<u64> {
    model.header().get(key)?.as_u64()
}

impl Config {
    /// Read the geometry out of a mapped file.
    ///
    /// 🔴 Only `olmoe` is accepted. Every other architecture differs somewhere the graph below
    /// would get silently wrong — a shared expert, a per-head QK-norm, another RoPE convention —
    /// and producing fluent nonsense is worse than refusing.
    pub fn from_model(model: &MappedModel) -> Result<Self, EngineError> {
        let arch = model.architecture()?.to_string();
        if arch != "olmoe" {
            return Err(EngineError::Unsupported(format!(
                "this forward pass implements `olmoe` only; the file declares `{arch}`"
            )));
        }
        let h = model.header();
        let need = |k: &str| -> Result<u64, EngineError> {
            h.u64_key(&format!("{arch}.{k}")).map_err(EngineError::Model)
        };

        let n_block = need("block_count")? as usize;
        let n_embd = need("embedding_length")? as usize;
        let n_ff = need("feed_forward_length")? as usize;
        let n_head = need("attention.head_count")? as usize;
        let n_head_kv = need("attention.head_count_kv")? as usize;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(EngineError::Unsupported(format!(
                "embedding length {n_embd} is not divisible by {n_head} heads"
            )));
        }
        let head_dim = n_embd / n_head;
        let n_rot = u64_key_opt(model, &format!("{arch}.rope.dimension_count"))
            .map_or(head_dim, |v| v as usize);

        let n_expert = need("expert_count")? as usize;
        let n_expert_used = need("expert_used_count")? as usize;
        if n_expert_used == 0 || n_expert_used > n_expert {
            return Err(EngineError::Unsupported(format!(
                "{n_expert_used} experts used of {n_expert}"
            )));
        }

        // The vocabulary comes from the embedding table's own shape rather than from the
        // tokeniser array: the two can disagree in a padded model, and the table's row count is
        // what the output matmul actually produces.
        let embd = model.tensor(names::TOKEN_EMBD)?;
        let n_vocab = *embd.dims.get(1).ok_or_else(|| {
            EngineError::Unsupported("token_embd.weight is not a matrix".to_string())
        })? as usize;

        let rms_eps = f32_key(model, &format!("{arch}.attention.layer_norm_rms_epsilon"))
            .ok_or_else(|| {
                EngineError::Unsupported("no attention.layer_norm_rms_epsilon".to_string())
            })?;

        Ok(Self {
            n_block,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            n_rot,
            n_expert,
            n_expert_used,
            n_vocab,
            n_ctx_train: need("context_length")? as usize,
            rms_eps,
            rope_freq_base: f32_key(model, &format!("{arch}.rope.freq_base")).unwrap_or(10_000.0),
            bos: u64_key_opt(model, "tokenizer.ggml.bos_token_id").map(|v| v as u32),
            eos: u64_key_opt(model, "tokenizer.ggml.eos_token_id").map(|v| v as u32),
            arch,
        })
    }

    /// `1/sqrt(head_dim)`, the scale llama.cpp passes to `build_attn`.
    fn kq_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// A weight matrix on the device, with the shape a matvec needs.
struct QTensor<'c> {
    buf: DeviceBuffer<'c>,
    ty: QuantType,
    /// Output length — the product of every dimension after the first.
    n_rows: usize,
    /// Reduction length — `dims[0]`, the contiguous axis.
    n_cols: usize,
}

/// The quantisation type of a view, refused rather than guessed if this build cannot expand it.
fn quant_of(v: &TensorView<'_>) -> Result<QuantType, EngineError> {
    QuantType::from_type_id(v.quant.id).ok_or_else(|| {
        EngineError::Unsupported(format!(
            "tensor `{}` is {}, which the kernels cannot expand",
            v.name, v.quant.name
        ))
    })
}

fn upload<'c>(ctx: &'c Context, v: &TensorView<'_>) -> Result<DeviceBuffer<'c>, EngineError> {
    let buf = ctx.alloc(v.data.len())?;
    ctx.upload(&buf, v.data)?;
    Ok(buf)
}

fn upload_matrix<'c>(ctx: &'c Context, v: &TensorView<'_>) -> Result<QTensor<'c>, EngineError> {
    let ty = quant_of(v)?;
    let (&n_cols, rest) = v.dims.split_first().ok_or_else(|| {
        EngineError::Unsupported(format!("tensor `{}` has no dimensions", v.name))
    })?;
    let n_rows = rest.iter().product::<u64>() as usize;
    Ok(QTensor { buf: upload(ctx, v)?, ty, n_rows, n_cols: n_cols as usize })
}

/// The geometry of one bank of one block's experts.
///
/// Every expert in a block is a slice of the same stacked tensor, so within a block they share
/// a type and a shape. Across blocks they do not — this file quantises `ffn_down_exps` at Q6_K
/// in 8 blocks and Q4_K in the other 8 — so this is per block, not per model.
#[derive(Debug, Clone, Copy)]
struct BankShape {
    ty: QuantType,
    n_rows: usize,
    n_cols: usize,
    /// Bytes one expert of this bank occupies in this block.
    bytes: usize,
}

/// One block's weights, resident on the device.
struct Block<'c> {
    attn_norm: DeviceBuffer<'c>,
    attn_q: QTensor<'c>,
    attn_k: QTensor<'c>,
    attn_v: QTensor<'c>,
    attn_output: QTensor<'c>,
    attn_q_norm: DeviceBuffer<'c>,
    attn_k_norm: DeviceBuffer<'c>,
    ffn_norm: DeviceBuffer<'c>,
    /// `[n_embd, n_expert]`. Uploaded as a matrix rather than assumed f32, so a quantised
    /// router in some other build would still be read through the right kernel.
    ffn_gate_inp: QTensor<'c>,
    /// The shape of an expert in each bank. The weights themselves are **not** here: they are
    /// staged into pool slots on demand, straight out of the mapping.
    gate: BankShape,
    up: BankShape,
    down: BankShape,
}

/// Every weight the graph reads, on the device.
pub struct Weights<'c> {
    pub cfg: Config,
    token_embd: QTensor<'c>,
    output_norm: DeviceBuffer<'c>,
    output: QTensor<'c>,
    blocks: Vec<Block<'c>>,
    /// Bytes copied to the card at load — the always-resident half. Summed from what was
    /// actually uploaded, not estimated.
    pub dense_bytes: u64,
    /// Bytes of expert bank in the file, summed from the tensor index. None of it is uploaded
    /// at load; it is staged slot by slot as the router asks for it.
    pub expert_bytes: u64,
    /// The largest an expert of each bank reaches in any block: gate, up, down. A pool slot is
    /// sized to this, so it can hold whichever block the router lands in.
    pub slot_bank_bytes: [usize; 3],
}

impl<'c> Weights<'c> {
    /// Upload the always-resident half of the model and measure the rest.
    ///
    /// Embeddings, attention, norms, the routers and the output head go to the card here and
    /// stay. The expert banks do not: they are read from the mapping on demand. On this model
    /// that split is roughly 360 MiB resident against 3.6 GiB pageable.
    pub fn upload(ctx: &'c Context, model: &MappedModel) -> Result<Self, EngineError> {
        let cfg = Config::from_model(model)?;
        let mut bytes = 0u64;
        let mut expert_bytes = 0u64;
        let mut slot_bank_bytes = [0usize; 3];

        let embd_view = model.tensor(names::TOKEN_EMBD)?;
        bytes += embd_view.data.len() as u64;
        let token_embd = upload_matrix(ctx, &embd_view)?;

        let onorm = model.tensor(names::OUTPUT_NORM)?;
        bytes += onorm.data.len() as u64;
        let output_norm = upload(ctx, &onorm)?;

        // 🔴 Read from the index, not assumed tied to the embedding: this file carries a
        // separate `output.weight`, and it is Q6_K where the embedding is Q4_K.
        let out_view = model.tensor(names::OUTPUT)?;
        bytes += out_view.data.len() as u64;
        let output = upload_matrix(ctx, &out_view)?;

        let mut blocks = Vec::with_capacity(cfg.n_block);
        for b in 0..cfg.n_block as u32 {
            for s in [
                names::ATTN_NORM,
                names::ATTN_Q,
                names::ATTN_K,
                names::ATTN_V,
                names::ATTN_OUTPUT,
                names::ATTN_Q_NORM,
                names::ATTN_K_NORM,
                names::FFN_NORM,
                names::FFN_GATE_INP,
            ] {
                bytes += model.block_tensor(b, s)?.data.len() as u64;
            }
            for s in [names::FFN_GATE_EXPS, names::FFN_UP_EXPS, names::FFN_DOWN_EXPS] {
                expert_bytes += model.block_tensor(b, s)?.data.len() as u64;
            }

            let simple = |suffix: &str| -> Result<DeviceBuffer<'c>, EngineError> {
                upload(ctx, &model.block_tensor(b, suffix)?)
            };
            let matrix = |suffix: &str| -> Result<QTensor<'c>, EngineError> {
                upload_matrix(ctx, &model.block_tensor(b, suffix)?)
            };
            // Expert 0 stands for the bank: `MappedModel::expert` refuses a bank whose last
            // dimension is not the expert count, so a shape read from one slice is a shape that
            // holds for all of them.
            let shape = |kind: ExpertBank| -> Result<BankShape, EngineError> {
                let v = model.expert(b, kind, 0)?;
                let ty = quant_of(&v)?;
                let (&n_cols, rest) = v.dims.split_first().ok_or_else(|| {
                    EngineError::Unsupported(format!("expert bank `{}` has no dimensions", v.name))
                })?;
                Ok(BankShape {
                    ty,
                    n_rows: rest.iter().product::<u64>() as usize,
                    n_cols: n_cols as usize,
                    bytes: v.data.len(),
                })
            };
            let (g, u, d) =
                (shape(ExpertBank::Gate)?, shape(ExpertBank::Up)?, shape(ExpertBank::Down)?);
            for (i, sh) in [g, u, d].iter().enumerate() {
                slot_bank_bytes[i] = slot_bank_bytes[i].max(sh.bytes);
            }

            blocks.push(Block {
                attn_norm: simple(names::ATTN_NORM)?,
                attn_q: matrix(names::ATTN_Q)?,
                attn_k: matrix(names::ATTN_K)?,
                attn_v: matrix(names::ATTN_V)?,
                attn_output: matrix(names::ATTN_OUTPUT)?,
                attn_q_norm: simple(names::ATTN_Q_NORM)?,
                attn_k_norm: simple(names::ATTN_K_NORM)?,
                ffn_norm: simple(names::FFN_NORM)?,
                ffn_gate_inp: matrix(names::FFN_GATE_INP)?,
                gate: g,
                up: u,
                down: d,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            output_norm,
            output,
            blocks,
            dense_bytes: bytes,
            expert_bytes,
            slot_bank_bytes,
        })
    }

    /// Every *(block, expert)* pair in the model — the number of residency slots a fully
    /// resident engine would need.
    ///
    /// 🔴 Not the expert count. The file says 64 experts; the model has 16 x 64 = 1024 slots,
    /// and conflating the two is a factor-of-16 error on this model.
    pub fn n_slots(&self) -> u32 {
        self.cfg.n_block as u32 * self.cfg.n_expert as u32
    }
}

// =======================================================================================
// Expert residency
// =======================================================================================

/// How much of the expert bank lives in VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Residency {
    /// Every *(block, expert)* slot. Nothing is ever fetched after the first token that needs
    /// it, and the cache degenerates to a warm-up. This is the baseline the rest is measured
    /// against.
    #[default]
    All,
    /// Exactly this many slots, managed by [`ExpertCache`]'s LRU.
    Slots(u32),
    /// Let [`crate::memory::plan`] decide, from a device measurement the caller supplies.
    ///
    /// 🔴 The measurement has to come from outside: querying the card lives in `moearc-device`,
    /// and this crate deliberately does not depend on it — being buildable and testable with no
    /// GPU present is the property that split buys. `moearc-cli` already holds both and is the
    /// natural caller.
    Planned(memory::DeviceMemory),
    /// The incumbent: the first `resident_blocks` blocks pinned in VRAM forever, everything
    /// above streamed through a ring just wide enough for one step.
    ///
    /// Mirrors `residency::Policy::StaticSplit`, which is how llama.cpp's `--n-cpu-moe` is
    /// modelled offline, so the two are comparable. ⚠️ One asymmetry to know about: the ring is
    /// real memory, so if only one block streams, its experts survive in the ring from one
    /// token to the next and hit. The offline model counts them as misses. The two therefore
    /// agree except at `resident_blocks == n_block - 1` — which is also the only setting under
    /// which `StaticSplit::next_victim`'s pinning matters, and the one that caught it.
    StaticSplit { resident_blocks: u16 },
}

/// The resident expert pool: `capacity` slots, each able to hold any one *(block, expert)*.
///
/// Three parallel arrays rather than one buffer per slot, because `matvec_q` reads a weight
/// matrix from the start of a `DeviceBuffer` and there is no way to offset into one. A slot is
/// therefore the same index in all three.
struct ExpertPool<'c> {
    gate: Vec<DeviceBuffer<'c>>,
    up: Vec<DeviceBuffer<'c>>,
    down: Vec<DeviceBuffer<'c>>,
    /// Bytes one slot commits across all three banks.
    slot_bytes: u64,
}

impl<'c> ExpertPool<'c> {
    fn new(ctx: &'c Context, per_bank: [usize; 3], capacity: u32) -> Result<Self, EngineError> {
        let alloc = |n: usize| -> Result<Vec<DeviceBuffer<'c>>, EngineError> {
            (0..capacity).map(|_| ctx.alloc(n).map_err(EngineError::Kernel)).collect()
        };
        Ok(Self {
            gate: alloc(per_bank[0])?,
            up: alloc(per_bank[1])?,
            down: alloc(per_bank[2])?,
            slot_bytes: per_bank.iter().map(|b| *b as u64).sum(),
        })
    }

    fn capacity(&self) -> u32 {
        self.gate.len() as u32
    }
}

/// The incumbent policy, implemented rather than simulated.
///
/// Blocks below the split own a slot each, permanently; blocks above it share a ring exactly
/// one step wide, so an expert fetched for block *n* is gone by the time block *n+1* has been
/// served. There is no recency to track and nothing to tune — which is the point. It exists so
/// the claim "dynamic residency beats a static split" can be a measurement in this engine
/// rather than a simulation result.
struct StaticSplit {
    n_expert: u32,
    resident_blocks: u16,
    /// Whether each pinned slot has been filled yet. A pinned expert still costs one transfer
    /// the first time it is asked for.
    pinned_filled: Vec<bool>,
    ring: Vec<Option<ExpertRef>>,
    ring_base: Slot,
    ring_next: usize,
    stats: CacheStats,
}

impl StaticSplit {
    fn new(n_expert: u32, resident_blocks: u16, ring_slots: u32) -> Self {
        let pinned = resident_blocks as u32 * n_expert;
        Self {
            n_expert,
            resident_blocks,
            pinned_filled: vec![false; pinned as usize],
            ring: vec![None; ring_slots as usize],
            ring_base: pinned,
            ring_next: 0,
            stats: CacheStats::default(),
        }
    }

    fn capacity(&self) -> u32 {
        self.pinned_filled.len() as u32 + self.ring.len() as u32
    }

    fn clear(&mut self) {
        self.pinned_filled.iter_mut().for_each(|f| *f = false);
        self.ring.iter_mut().for_each(|r| *r = None);
        self.ring_next = 0;
        self.stats = CacheStats::default();
    }

    /// The next ring position to overwrite, skipping any that holds an expert **this step
    /// still needs**.
    ///
    /// 🔴 This skip is not a refinement, it is the correctness of the whole policy, and leaving
    /// it out shipped wrong tokens. Plain round-robin will hand back a position holding an
    /// expert that was already reported as a hit earlier in the same step; staging then
    /// overwrites it before any matmul runs, and two different experts read the same slot. The
    /// output stays finite and fluent — `static:15` on this model diverged from llama.cpp at
    /// token 18 and nowhere earlier.
    ///
    /// It only ever bites when a ring entry survives from one visit to the next, which needs
    /// the ring to be at least as wide as the streaming demand *and* the same block to be
    /// visited twice without an intervening flush. With two or more streaming blocks the ring
    /// is flushed in between, so every position is stale and round-robin looks correct. One
    /// streaming block is the case that exposes it, and it was the last one swept.
    ///
    /// A victim always exists when one is wanted: reaching here means at least one demand
    /// missed, so at most `streaming - 1` positions can be pinned out of `ring.len() >=
    /// streaming`.
    fn next_victim(&mut self, needed: &[ExpertRef]) -> Option<usize> {
        for _ in 0..self.ring.len() {
            let pos = self.ring_next;
            self.ring_next = (self.ring_next + 1) % self.ring.len();
            match self.ring[pos] {
                Some(occupant) if needed.contains(&occupant) => continue,
                _ => return Some(pos),
            }
        }
        None
    }

    fn admit(&mut self, needed: &[ExpertRef]) -> Result<StepPlan, CacheError> {
        let streaming = needed.iter().filter(|e| e.layer >= self.resident_blocks).count();
        if streaming > self.ring.len() {
            return Err(CacheError::StepExceedsCapacity {
                needed: streaming,
                capacity: self.ring.len() as u32,
            });
        }
        self.stats.steps += 1;
        let mut plan = StepPlan::default();
        for &e in needed {
            self.stats.demands += 1;
            if e.layer < self.resident_blocks {
                let slot = u32::from(e.layer) * self.n_expert + u32::from(e.expert);
                if self.pinned_filled[slot as usize] {
                    self.stats.hits += 1;
                    plan.hits.push((e, slot));
                } else {
                    self.stats.misses += 1;
                    self.pinned_filled[slot as usize] = true;
                    plan.loads.push(Load { expert: e, into_slot: slot, evicted: None });
                }
                continue;
            }
            if let Some(pos) = self.ring.iter().position(|o| *o == Some(e)) {
                self.stats.hits += 1;
                plan.hits.push((e, self.ring_base + pos as u32));
                continue;
            }
            self.stats.misses += 1;
            let pos = self.next_victim(needed).ok_or(CacheError::StepExceedsCapacity {
                needed: streaming,
                capacity: self.ring.len() as u32,
            })?;
            let evicted = self.ring[pos].replace(e);
            if evicted.is_some() {
                self.stats.evictions += 1;
            }
            plan.loads.push(Load { expert: e, into_slot: self.ring_base + pos as u32, evicted });
        }
        Ok(plan)
    }
}

/// Which admission policy is deciding what stays resident.
enum Admission {
    Lru(ExpertCache),
    Static(StaticSplit),
}

impl Admission {
    fn admit(&mut self, needed: &[ExpertRef]) -> Result<StepPlan, CacheError> {
        match self {
            Self::Lru(c) => c.admit(needed),
            Self::Static(s) => s.admit(needed),
        }
    }

    fn stats(&self) -> CacheStats {
        match self {
            Self::Lru(c) => c.stats(),
            Self::Static(s) => s.stats,
        }
    }

    fn capacity(&self) -> u32 {
        match self {
            Self::Lru(c) => c.capacity(),
            Self::Static(s) => s.capacity(),
        }
    }

    fn policy_name(&self) -> &'static str {
        match self {
            Self::Lru(_) => "lru",
            Self::Static(_) => "static split",
        }
    }

    /// Forget everything resident, so the next token pays a cold cache.
    fn clear(&mut self) -> Result<(), CacheError> {
        match self {
            Self::Lru(c) => *c = ExpertCache::new(c.capacity())?,
            Self::Static(s) => s.clear(),
        }
        Ok(())
    }
}

/// Intermediate activations copied back to the host, for comparison against another
/// implementation.
///
/// 🔴 This exists because a forward pass that emits garbage cannot be debugged end to end. The
/// residual stream after each block is llama.cpp's `l_out-<il>`, and the vector fed to the
/// output head is its `result_norm`; capturing both by the same names is what makes a
/// difference attributable to one block instead of to two hundred operations.
#[derive(Debug, Default, Clone)]
pub struct Tap {
    pub items: Vec<(String, Vec<f32>)>,
}

impl Tap {
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.items.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
    }
}

fn tap_record(
    tap: &mut Option<Tap>,
    name: String,
    ctx: &Context,
    src: &DeviceBuffer<'_>,
    n: usize,
) -> Result<(), KernelError> {
    if let Some(t) = tap {
        let mut v = vec![0.0f32; n];
        ctx.download_slice(&mut v, src)?;
        t.items.push((name, v));
    }
    Ok(())
}

/// Everything that changes as a sequence runs: activations, the KV pool, the page table.
pub struct State<'c> {
    tok: DeviceBuffer<'c>,
    pos: DeviceBuffer<'c>,
    h: DeviceBuffer<'c>,
    x: DeviceBuffer<'c>,
    q: DeviceBuffer<'c>,
    k: DeviceBuffer<'c>,
    v: DeviceBuffer<'c>,
    q_normed: DeviceBuffer<'c>,
    k_normed: DeviceBuffer<'c>,
    q_roped: DeviceBuffer<'c>,
    k_roped: DeviceBuffer<'c>,
    attn: DeviceBuffer<'c>,
    proj: DeviceBuffer<'c>,
    router: DeviceBuffer<'c>,
    idx: DeviceBuffer<'c>,
    weights: DeviceBuffer<'c>,
    gate: DeviceBuffer<'c>,
    up: DeviceBuffer<'c>,
    act: DeviceBuffer<'c>,
    expert_out: DeviceBuffer<'c>,
    ffn: DeviceBuffer<'c>,
    logits_dev: DeviceBuffer<'c>,
    block_table: DeviceBuffer<'c>,

    /// One K and one V pool per block, paged and indexed through `block_table`.
    k_pages: Vec<DeviceBuffer<'c>>,
    v_pages: Vec<DeviceBuffer<'c>>,

    kv: PagedKvCache,
    begun: bool,
    /// Tokens written to the cache so far, which is `attn_decode`'s `n_kv`.
    n_kv: usize,
    /// The logical-to-physical page table, mirrored on the host so it can be re-uploaded when
    /// it grows.
    table_host: Vec<u32>,

    idx_host: Vec<u32>,
    w_host: Vec<f32>,
    /// Expert bytes copied host-to-device since the last `reset_traffic`. Counted from the
    /// slices actually uploaded, not from a per-expert constant times a miss count.
    bytes_staged: u64,
    /// Scratch for the experts one block wants, reused so the hot loop does not allocate.
    wanted: Vec<ExpertRef>,
    /// The last token's logits.
    pub logits: Vec<f32>,
    /// Set to `Some(Tap::default())` to capture per-block activations.
    pub tap: Option<Tap>,
    n_ctx: usize,
}

impl<'c> State<'c> {
    /// Allocate scratch and a KV pool for `n_ctx` tokens.
    pub fn new(ctx: &'c Context, cfg: &Config, n_ctx: usize) -> Result<Self, EngineError> {
        let n_embd = cfg.n_embd;
        let n_embd_kv = cfg.n_head_kv * cfg.head_dim;
        let pages = n_ctx.div_ceil(PAGE_TOKENS).max(1);
        let page_elems = PAGE_TOKENS * n_embd_kv;

        let mut k_pages = Vec::with_capacity(cfg.n_block);
        let mut v_pages = Vec::with_capacity(cfg.n_block);
        for _ in 0..cfg.n_block {
            k_pages.push(ctx.alloc(pages * page_elems * KV.elem_bytes())?);
            v_pages.push(ctx.alloc(pages * page_elems * KV.elem_bytes())?);
        }

        Ok(Self {
            tok: ctx.alloc_n::<u32>(1)?,
            pos: ctx.alloc_n::<i32>(1)?,
            h: ctx.alloc_n::<f32>(n_embd)?,
            x: ctx.alloc_n::<f32>(n_embd)?,
            q: ctx.alloc_n::<f32>(n_embd)?,
            k: ctx.alloc_n::<f32>(n_embd_kv)?,
            v: ctx.alloc_n::<f32>(n_embd_kv)?,
            q_normed: ctx.alloc_n::<f32>(n_embd)?,
            k_normed: ctx.alloc_n::<f32>(n_embd_kv)?,
            q_roped: ctx.alloc_n::<f32>(n_embd)?,
            k_roped: ctx.alloc_n::<f32>(n_embd_kv)?,
            attn: ctx.alloc_n::<f32>(cfg.n_head * cfg.head_dim)?,
            proj: ctx.alloc_n::<f32>(n_embd)?,
            router: ctx.alloc_n::<f32>(cfg.n_expert)?,
            idx: ctx.alloc_n::<u32>(cfg.n_expert_used)?,
            weights: ctx.alloc_n::<f32>(cfg.n_expert_used)?,
            gate: ctx.alloc_n::<f32>(cfg.n_ff)?,
            up: ctx.alloc_n::<f32>(cfg.n_ff)?,
            act: ctx.alloc_n::<f32>(cfg.n_ff)?,
            expert_out: ctx.alloc_n::<f32>(n_embd)?,
            ffn: ctx.alloc_n::<f32>(n_embd)?,
            logits_dev: ctx.alloc_n::<f32>(cfg.n_vocab)?,
            block_table: ctx.alloc_n::<u32>(pages)?,
            k_pages,
            v_pages,
            kv: PagedKvCache::new(pages as u32, PAGE_TOKENS as u32, n_ctx as u32)?,
            begun: false,
            n_kv: 0,
            table_host: Vec::new(),
            idx_host: vec![0; cfg.n_expert_used],
            w_host: vec![0.0; cfg.n_expert_used],
            bytes_staged: 0,
            wanted: Vec::with_capacity(cfg.n_expert_used),
            logits: vec![0.0; cfg.n_vocab],
            tap: None,
            n_ctx,
        })
    }

    /// Forget the sequence. The next token starts at position 0 with an empty cache.
    ///
    /// The pooled memory is not cleared: every read of the cache is bounded by `n_kv`, so stale
    /// bytes past the write head are unreachable rather than merely unlikely to matter.
    pub fn reset(&mut self) -> Result<(), EngineError> {
        if self.begun {
            self.kv.end(SEQ)?;
            self.begun = false;
        }
        self.n_kv = 0;
        self.table_host.clear();
        Ok(())
    }
}

/// A model on the device: resident weights, an expert pool, and the state of one sequence.
///
/// It keeps a borrow of the mapping, because that is where the experts come from on a miss.
pub struct Model<'c, 'm> {
    ctx: &'c Context,
    mapped: &'m MappedModel,
    pub weights: Weights<'c>,
    pub state: State<'c>,
    pool: ExpertPool<'c>,
    admission: Admission,
}

impl<'c, 'm> Model<'c, 'm> {
    pub fn new(
        ctx: &'c Context,
        model: &'m MappedModel,
        n_ctx: usize,
        residency: Residency,
    ) -> Result<Self, EngineError> {
        let weights = Weights::upload(ctx, model)?;
        let cfg = &weights.cfg;
        let n_slots = weights.n_slots();

        // A step activates `n_expert_used` distinct experts of one block, so no policy can serve
        // a pool smaller than that — `ExpertCache::admit` refuses it rather than thrashing, and
        // clamping here turns a confusing runtime refusal into a quiet, documented floor.
        let floor = cfg.n_expert_used as u32;

        let admission = match residency {
            Residency::All => Admission::Lru(ExpertCache::new(n_slots)?),
            Residency::Slots(n) => Admission::Lru(ExpertCache::new(n.clamp(floor, n_slots))?),
            Residency::Planned(device) => {
                let info = ModelInfo::from_header(model.header())?;
                let footprint = memory::ModelFootprint {
                    dense_weights_bytes: info.dense_weights_bytes,
                    per_expert_bytes: info.per_expert_bytes,
                    // 🔴 Slots, not experts. `moe_block_count * total_experts`: the file says 64
                    // experts and the planner must be told about 1024 places to put one.
                    total_experts: info.moe_block_count * info.total_experts,
                    active_experts: info.moe_block_count * info.active_experts,
                    kv_bytes_per_token: info.kv_bytes_per_token,
                };
                // 🔴 The planner reserves `min_context_tokens` off the top before it places
                // a single expert, and its default is 2048 — a product judgement about the
                // shortest context worth serving, made before a context length existed. Here
                // one does, and it is not negotiable: the KV cache is already allocated for
                // exactly `n_ctx`. Leaving the default in place makes the planner refuse any
                // session shorter than 2048 tokens, which is a policy floor being reported as
                // a capacity failure.
                let policy = memory::Policy {
                    min_context_tokens: n_ctx as u32,
                    ..memory::Policy::default()
                };
                let allocation = memory::plan(
                    device,
                    &footprint,
                    &policy,
                    memory::Context::Tokens(n_ctx as u32),
                )?;
                Admission::Lru(ExpertCache::new(allocation.resident_experts.clamp(floor, n_slots))?)
            }
            Residency::StaticSplit { resident_blocks } => Admission::Static(StaticSplit::new(
                cfg.n_expert as u32,
                resident_blocks.min(cfg.n_block as u16),
                floor,
            )),
        };

        let pool = ExpertPool::new(ctx, weights.slot_bank_bytes, admission.capacity())?;
        let state = State::new(ctx, &weights.cfg, n_ctx)?;
        Ok(Self { ctx, mapped: model, weights, state, pool, admission })
    }

    pub fn cfg(&self) -> &Config {
        &self.weights.cfg
    }

    /// Slots the expert pool holds, and what they cost.
    pub fn residency(&self) -> ResidencyReport {
        let stats = self.admission.stats();
        ResidencyReport {
            policy: self.admission.policy_name(),
            resident_slots: self.pool.capacity(),
            total_slots: self.weights.n_slots(),
            slot_bytes: self.pool.slot_bytes,
            pool_bytes: self.pool.slot_bytes * u64::from(self.pool.capacity()),
            dense_bytes: self.weights.dense_bytes,
            expert_bytes: self.weights.expert_bytes,
            stats,
            bytes_staged: self.state.bytes_staged,
        }
    }

    /// Zero the cache counters without disturbing what is resident.
    ///
    /// Kept separate from [`Model::clear_residency`] on purpose: measuring a warm cache means
    /// resetting the counters and *not* the contents, and one call that did both would make
    /// that measurement impossible to express.
    pub fn reset_cache_stats(&mut self) {
        match &mut self.admission {
            Admission::Lru(c) => c.reset_stats(),
            Admission::Static(s) => s.stats = CacheStats::default(),
        }
        self.state.bytes_staged = 0;
    }

    /// Forget everything resident, so the next token pays a cold cache.
    pub fn clear_residency(&mut self) -> Result<(), EngineError> {
        self.admission.clear()?;
        self.state.bytes_staged = 0;
        Ok(())
    }

    /// Forget the sequence: the next `decode` starts at position 0 with an empty cache.
    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.state.reset()
    }

    /// Tokens currently in the KV cache — the position the next `decode` will occupy.
    pub fn position(&self) -> usize {
        self.state.n_kv
    }

    /// The context length this model was built for.
    pub fn n_ctx(&self) -> usize {
        self.state.n_ctx
    }

    /// Run one token through every block and return its logits.
    ///
    /// The token is appended to the KV cache at the next free position, so calling this
    /// repeatedly walks a sequence. Prompt tokens and generated tokens take exactly this path —
    /// there is no separate prefill, which costs speed on a long prompt and buys one code path
    /// to be correct instead of two.
    pub fn decode(&mut self, token: u32) -> Result<&[f32], EngineError> {
        let ctx = self.ctx;
        let mapped = self.mapped;
        let w = &self.weights;
        let cfg = &w.cfg;
        let pool = &self.pool;
        let admission = &mut self.admission;
        let st = &mut self.state;

        if token as usize >= cfg.n_vocab {
            return Err(EngineError::Unsupported(format!(
                "token id {token} is outside the {}-entry vocabulary",
                cfg.n_vocab
            )));
        }

        let _p_step = profile::scope("decode.total");
        if !st.begun {
            st.kv.begin(SEQ, 0)?;
            st.begun = true;
        }
        let (page, slot) = st.kv.append(SEQ)?;
        let pos = st.n_kv as i32;
        st.n_kv += 1;

        // The page table only changes when a page is added, and re-uploading it then is the
        // entire cost, so it is compared rather than tracked.
        let pages = &st.kv.pages_of(SEQ)?.pages;
        if pages[..] != st.table_host[..] {
            st.table_host.clear();
            st.table_host.extend_from_slice(pages);
            ctx.upload_slice(&st.block_table, &st.table_host)?;
        }

        {
            let _p = profile::scope("setup.upload");
            ctx.upload_slice(&st.tok, &[token])?;
            ctx.upload_slice(&st.pos, &[pos])?;
        }

        let n_embd = cfg.n_embd;
        let n_embd_kv = cfg.n_head_kv * cfg.head_dim;
        let eps = cfg.rms_eps;
        let scale = cfg.kq_scale();

        // h = token_embd[token]
        {
            let _p = profile::scope("embed");
            ctx.embed_rows(w.token_embd.ty, &st.h, &w.token_embd.buf, &st.tok, 1, n_embd)?;
        }

        for (bi, b) in w.blocks.iter().enumerate() {
            // ---- attention -------------------------------------------------------------
            {
                let _p = profile::scope("attn.norm");
                ctx.rmsnorm(&st.x, &st.h, Some(&b.attn_norm), 1, n_embd, eps)?;
            }
            {
                let _p = profile::scope("attn.qkv");
                matvec(ctx, &st.q, &b.attn_q, &st.x)?;
                matvec(ctx, &st.k, &b.attn_k, &st.x)?;
                matvec(ctx, &st.v, &b.attn_v, &st.x)?;
            }

            // QK-norm over the whole vector — before the head reshape, before RoPE.
            {
                let _p = profile::scope("attn.qk_norm");
                ctx.rmsnorm(&st.q_normed, &st.q, Some(&b.attn_q_norm), 1, n_embd, eps)?;
                ctx.rmsnorm(&st.k_normed, &st.k, Some(&b.attn_k_norm), 1, n_embd_kv, eps)?;
            }

            {
                let _p = profile::scope("attn.rope");
                ctx.rope(
                    &st.q_roped,
                    &st.q_normed,
                    &st.pos,
                    1,
                    cfg.n_head,
                    cfg.head_dim,
                    cfg.n_rot,
                    cfg.rope_freq_base,
                    RopeKind::Neox,
                )?;
                ctx.rope(
                    &st.k_roped,
                    &st.k_normed,
                    &st.pos,
                    1,
                    cfg.n_head_kv,
                    cfg.head_dim,
                    cfg.n_rot,
                    cfg.rope_freq_base,
                    RopeKind::Neox,
                )?;
            }

            {
                let _p = profile::scope("attn.kv_append");
                ctx.kv_append(
                    &st.k_pages[bi],
                    &st.v_pages[bi],
                    &st.k_roped,
                    &st.v,
                    page,
                    slot,
                    cfg.n_head_kv,
                    cfg.head_dim,
                    PAGE_TOKENS,
                    KV,
                )?;
            }
            {
                let _p = profile::scope("attn.attend");
                ctx.attn_decode(
                    &st.attn,
                    &st.q_roped,
                    &st.k_pages[bi],
                    &st.v_pages[bi],
                    &st.block_table,
                    cfg.n_head,
                    cfg.n_head_kv,
                    cfg.head_dim,
                    st.n_kv,
                    PAGE_TOKENS,
                    scale,
                    KV,
                )?;
            }
            {
                let _p = profile::scope("attn.proj");
                matvec(ctx, &st.proj, &b.attn_output, &st.attn)?;
                // `add` writes each index from the same index of both inputs, so aliasing the
                // accumulator into the left operand is well defined.
                ctx.add(&st.h, &st.h, &st.proj, n_embd)?;
            }
            tap_record(&mut st.tap, format!("ffn_inp-{bi}"), ctx, &st.h, n_embd)?;

            // ---- MoE FFN ---------------------------------------------------------------
            {
                let _p = profile::scope("moe.norm");
                ctx.rmsnorm(&st.x, &st.h, Some(&b.ffn_norm), 1, n_embd, eps)?;
            }
            {
                let _p = profile::scope("moe.router");
                matvec(ctx, &st.router, &b.ffn_gate_inp, &st.x)?;
            }
            tap_record(&mut st.tap, format!("ffn_moe_logits-{bi}"), ctx, &st.router, cfg.n_expert)?;
            // `normalize = false`: llama.cpp calls `build_moe_ffn` with `norm_w = false` for
            // OLMoE, so the weights are raw softmax probabilities and do not sum to one.
            {
                let _p = profile::scope("moe.topk");
                ctx.topk_router(
                    &st.idx,
                    &st.weights,
                    &st.router,
                    1,
                    cfg.n_expert,
                    cfg.n_expert_used,
                    false,
                )?;
            }
            {
                let _p = profile::scope("moe.readback");
                ctx.download_slice(&mut st.idx_host, &st.idx)?;
                ctx.download_slice(&mut st.w_host, &st.weights)?;
            }
            if let Some(t) = st.tap.as_mut() {
                t.items.push((
                    format!("ffn_moe_topk-{bi}"),
                    st.idx_host.iter().map(|i| *i as f32).collect(),
                ));
                t.items.push((format!("ffn_moe_weights-{bi}"), st.w_host.clone()));
            }

            // Ask the cache what is resident and what must move.
            let plan = {
                let _p = profile::scope("moe.admit");
                st.wanted.clear();
                for e in &st.idx_host {
                    if *e as usize >= cfg.n_expert {
                        return Err(EngineError::Unsupported(format!(
                            "the router named expert {e}, past the {} in block {bi}",
                            cfg.n_expert
                        )));
                    }
                    st.wanted.push(ExpertRef::new(bi as u16, *e as u16));
                }
                admission.admit(&st.wanted)?
            };

            // 🔴 Every miss is staged BEFORE anything computes. A matmul against a slot still
            // being filled does not fail — it returns plausible wrong output — so this ordering
            // is the difference between a bug that is caught and one that is never found.
            {
                let _p = profile::scope("moe.stage");
                for load in &plan.loads {
                    st.bytes_staged += stage(ctx, mapped, pool, load.expert, load.into_slot)?;
                }
            }
            let slots = plan.slots_for(&st.wanted);

            {
                let _p = profile::scope("moe.zero");
                ctx.zero(&st.ffn, n_embd)?;
            }
            for (j, &slot) in slots.iter().enumerate() {
                let weight = st.w_host[j];
                let sl = slot as usize;
                {
                    let _p = profile::scope("moe.expert_matvec");
                    matvec_bank(ctx, &st.gate, &pool.gate[sl], &b.gate, &st.x)?;
                    matvec_bank(ctx, &st.up, &pool.up[sl], &b.up, &st.x)?;
                }
                {
                    let _p = profile::scope("moe.swiglu");
                    ctx.swiglu(&st.act, &st.gate, &st.up, cfg.n_ff)?;
                }
                {
                    let _p = profile::scope("moe.expert_down");
                    matvec_bank(ctx, &st.expert_out, &pool.down[sl], &b.down, &st.act)?;
                }
                {
                    let _p = profile::scope("moe.axpy");
                    ctx.axpy(&st.ffn, &st.expert_out, weight, n_embd)?;
                }
            }
            tap_record(&mut st.tap, format!("ffn_moe_out-{bi}"), ctx, &st.ffn, n_embd)?;
            {
                let _p = profile::scope("moe.add");
                ctx.add(&st.h, &st.h, &st.ffn, n_embd)?;
            }
            tap_record(&mut st.tap, format!("l_out-{bi}"), ctx, &st.h, n_embd)?;
        }

        {
            let _p = profile::scope("out.norm");
            ctx.rmsnorm(&st.x, &st.h, Some(&w.output_norm), 1, n_embd, eps)?;
        }
        tap_record(&mut st.tap, "result_norm".to_string(), ctx, &st.x, n_embd)?;
        {
            let _p = profile::scope("out.matvec");
            matvec(ctx, &st.logits_dev, &w.output, &st.x)?;
        }
        {
            let _p = profile::scope("out.readback");
            ctx.download_slice(&mut st.logits, &st.logits_dev)?;
        }
        Ok(&st.logits)
    }
}

/// What the pool is holding and what it has cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyReport {
    pub policy: &'static str,
    pub resident_slots: u32,
    pub total_slots: u32,
    /// Bytes one slot commits, across all three banks.
    pub slot_bytes: u64,
    /// Device bytes the pool commits in total.
    pub pool_bytes: u64,
    /// Always-resident weights.
    pub dense_bytes: u64,
    /// The whole expert bank as it sits in the file.
    pub expert_bytes: u64,
    pub stats: CacheStats,
    /// Expert bytes actually copied across the bus.
    pub bytes_staged: u64,
}

impl ResidencyReport {
    /// Fraction of the model's slots that fit.
    pub fn resident_fraction(&self) -> f64 {
        if self.total_slots == 0 {
            0.0
        } else {
            f64::from(self.resident_slots) / f64::from(self.total_slots)
        }
    }
}

/// Copy one expert's three banks out of the mapping and into a pool slot.
///
/// The bytes come straight from `MappedModel::expert`, which is a borrow of the mapping — the
/// only copy in the path is the one to the device.
fn stage(
    ctx: &Context,
    mapped: &MappedModel,
    pool: &ExpertPool<'_>,
    expert: ExpertRef,
    slot: Slot,
) -> Result<u64, EngineError> {
    let (b, e) = (u32::from(expert.layer), u32::from(expert.expert));
    let s = slot as usize;
    let mut moved = 0u64;
    for (kind, dst) in [
        (ExpertBank::Gate, &pool.gate[s]),
        (ExpertBank::Up, &pool.up[s]),
        (ExpertBank::Down, &pool.down[s]),
    ] {
        let v = mapped.expert(b, kind, e)?;
        // 🔴 `Context::upload` copies `min(src, dst)` and reports success either way, so a slot
        // sized for the wrong block would silently truncate an expert and produce fluent
        // nonsense. Checked here rather than trusted to the pool's sizing arithmetic.
        if dst.len() < v.data.len() {
            return Err(EngineError::Kernel(KernelError::BufferTooSmall {
                what: "expert slot",
                need: v.data.len(),
                have: dst.len(),
            }));
        }
        ctx.upload(dst, v.data)?;
        moved += v.data.len() as u64;
    }
    Ok(moved)
}

/// `out = W . x` for an expert staged into a pool slot.
///
/// The slot may be larger than this block's expert — it is sized for the largest block — which
/// `matvec_q` tolerates: it checks the buffer is big enough, not that it is exact.
fn matvec_bank(
    ctx: &Context,
    out: &DeviceBuffer<'_>,
    w: &DeviceBuffer<'_>,
    shape: &BankShape,
    x: &DeviceBuffer<'_>,
) -> Result<(), KernelError> {
    ctx.matvec_q(shape.ty, out, w, x, shape.n_rows, shape.n_cols)
}

/// `out = W . x`, dispatching on how the weight is stored.
///
/// f32 weights go through `matvec_f32` rather than `matvec_q` with a one-element "block": the
/// two compute the same product, but the dedicated kernel is the one the f32 path is tested on.
fn matvec(
    ctx: &Context,
    out: &DeviceBuffer<'_>,
    w: &QTensor<'_>,
    x: &DeviceBuffer<'_>,
) -> Result<(), KernelError> {
    if w.ty == QuantType::F32 {
        ctx.matvec_f32(out, &w.buf, x, w.n_rows, w.n_cols)
    } else {
        ctx.matvec_q(w.ty, out, &w.buf, x, w.n_rows, w.n_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(layer: u16, expert: u16) -> ExpertRef {
        ExpertRef::new(layer, expert)
    }

    /// Every expert a step names must end up in a slot of its own once the plan is executed.
    /// Two of them sharing a slot is the silent-wrong-output failure, and it is invisible in
    /// the plan's own fields — it only shows in the mapping `slots_for` produces.
    fn assert_no_slot_collision(plan: &StepPlan, needed: &[ExpertRef]) {
        let slots = plan.slots_for(needed);
        for i in 0..needed.len() {
            for j in i + 1..needed.len() {
                if needed[i] != needed[j] {
                    assert_ne!(
                        slots[i], slots[j],
                        "{:?} and {:?} were both given slot {}",
                        needed[i], needed[j], slots[i]
                    );
                }
            }
        }
    }

    #[test]
    fn a_pinned_block_costs_one_transfer_and_then_hits_forever() {
        let mut s = StaticSplit::new(64, 2, 8);
        let need: Vec<ExpertRef> = (0..8).map(|x| e(1, x)).collect();
        let plan = s.admit(&need).unwrap();
        assert_eq!(plan.loads.len(), 8, "a cold pinned block still has to be fetched");
        let plan = s.admit(&need).unwrap();
        assert!(plan.loads.is_empty(), "a pinned block must not be refetched");
        assert_eq!(s.stats.hits, 8);
    }

    #[test]
    fn a_streaming_block_never_hits_across_other_blocks() {
        // The incumbent's defining property: above the split there is no residency to speak of.
        let mut s = StaticSplit::new(64, 0, 8);
        let a: Vec<ExpertRef> = (0..8).map(|x| e(0, x)).collect();
        let b: Vec<ExpertRef> = (0..8).map(|x| e(1, x)).collect();
        s.admit(&a).unwrap();
        s.admit(&b).unwrap();
        let plan = s.admit(&a).unwrap();
        assert_eq!(plan.loads.len(), 8, "the ring should have been flushed by the other block");
        assert_eq!(s.stats.hits, 0);
    }

    #[test]
    fn the_ring_never_evicts_an_expert_the_same_step_still_needs() {
        // 🔴 The regression. With one streaming block the ring survives from one token to the
        // next, so a step can hit on some entries and miss on others — and a round-robin victim
        // will then land on a slot this step already handed out. On the real model this showed
        // up as `static:15` diverging from llama.cpp at token 18; here it is three lines.
        let mut s = StaticSplit::new(64, 15, 8);
        let first: Vec<ExpertRef> = (0..8).map(|x| e(15, x)).collect();
        s.admit(&first).unwrap();

        // Three of the previous step's experts again, five new ones.
        let second: Vec<ExpertRef> = vec![
            e(15, 0),
            e(15, 1),
            e(15, 2),
            e(15, 90),
            e(15, 91),
            e(15, 92),
            e(15, 93),
            e(15, 94),
        ];
        let plan = s.admit(&second).unwrap();
        assert_eq!(plan.hits.len(), 3);
        assert_eq!(plan.loads.len(), 5);
        for load in &plan.loads {
            if let Some(v) = load.evicted {
                assert!(!second.contains(&v), "{v:?} was evicted while this step still needed it");
            }
        }
        assert_no_slot_collision(&plan, &second);
    }

    #[test]
    fn a_step_wider_than_the_ring_is_refused_rather_than_corrupted() {
        let mut s = StaticSplit::new(64, 0, 4);
        let need: Vec<ExpertRef> = (0..8).map(|x| e(0, x)).collect();
        assert!(matches!(
            s.admit(&need),
            Err(CacheError::StepExceedsCapacity { needed: 8, capacity: 4 })
        ));
    }

    #[test]
    fn clearing_forgets_both_halves() {
        let mut s = StaticSplit::new(64, 2, 8);
        s.admit(&(0..8).map(|x| e(1, x)).collect::<Vec<_>>()).unwrap();
        s.admit(&(0..8).map(|x| e(5, x)).collect::<Vec<_>>()).unwrap();
        s.clear();
        assert_eq!(s.stats, CacheStats::default());
        let plan = s.admit(&(0..8).map(|x| e(1, x)).collect::<Vec<_>>()).unwrap();
        assert_eq!(plan.loads.len(), 8, "a cleared pool must refetch even its pinned half");
    }

    #[test]
    fn the_lru_cache_also_keeps_a_steps_experts_in_distinct_slots() {
        // The same property, asserted against the policy that is actually shipped. `ExpertCache`
        // pins them by construction; this is the check that says so from the outside.
        let mut c = ExpertCache::new(10).unwrap();
        let first: Vec<ExpertRef> = (0..8).map(|x| e(3, x)).collect();
        c.admit(&first).unwrap();
        let second: Vec<ExpertRef> =
            vec![e(3, 0), e(3, 1), e(3, 2), e(3, 20), e(3, 21), e(3, 22), e(3, 23), e(3, 24)];
        let plan = c.admit(&second).unwrap();
        assert_no_slot_collision(&plan, &second);
    }
}
