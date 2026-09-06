//! One decode step of a routed-MoE transformer, on the device.
//!
//! This is the integration layer: it reads a GGUF with `moearc-model`, uploads every tensor to
//! the card, and issues the `moearc-kernels` calls that turn one token id into one logit
//! vector. It contains no arithmetic of its own — every operation below is a kernel that was
//! already checked against a CPU twin and, for the dequantisers, against llama.cpp itself.
//!
//! # Two architectures, one graph
//!
//! [`Arch`] lists what this pass implements: `olmoe` and `qwen3moe`. They are the *same* graph —
//! RMSNorm, QK-normalised attention with NeoX RoPE, a softmax router, a SwiGLU expert FFN, no
//! shared expert, no biases anywhere — differing in four scalars and two switches, every one of
//! which is read or derived in [`Config::from_model`] and named there. That is why this is one
//! module and not two: the half of the file that matters to this project — the expert pool, the
//! admission policies, `stage`, the ordering rule — is architecture-independent, and a second
//! copy of it would drift from this one within a week.
//!
//! 🔴 [`Config::from_model`] still refuses every other architecture by name. Each MoE family
//! differs somewhere this graph would get silently wrong — a shared expert, a sigmoid router, a
//! partial-rotary RoPE — and fluent nonsense is worse than a refusal.
//!
//! # Where the graph came from
//!
//! Transcribed from llama.cpp's `llama_model_olmoe::graph` (`src/models/olmoe.cpp`),
//! `llama_model_qwen3moe::graph` (`src/models/qwen3moe.cpp`) and
//! `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`), not from the tensor names. None of
//! the following is guessable from the names, and each is wrong in an interesting way if assumed:
//!
//! - **QK-norm spans a different vector in the two.** OLMoE's `attn_q_norm.weight` is `n_embd`
//!   long and normalises the whole projection before the reshape into heads. Qwen3's is
//!   `head_dim` long — `create_tensor(..., {n_embd_head_k})` in `load_arch_tensors` — and
//!   normalises **each head separately**, because `build_qkv` has already returned a 3D tensor by
//!   the time `build_norm` runs. Either choice raises no error on the other model and degrades
//!   the output subtly. [`Config`] takes the answer from the architecture and then **checks it
//!   against the tensor's own length**.
//! - **`head_dim` is not `n_embd / n_head`.** Qwen3-30B-A3B is 2048 wide with 32 heads and a head
//!   dimension of **128**; the quotient is 64. llama.cpp reads `attention.key_length` and falls
//!   back to the quotient only when the file is silent, and so does this. It also means the Q
//!   projection is *wider* than the residual stream (4096 against 2048), which is why the Q-side
//!   activation buffers below are sized `n_head * head_dim` and not `n_embd`.
//! - **The expert FFN's width is `expert_feed_forward_length` for `qwen3moe`** (768), not
//!   `feed_forward_length` (6144). The latter describes a dense FFN this architecture does not
//!   have; `load_arch_tensors` builds every expert bank from `hparams.n_ff_exp()`. OLMoE has no
//!   such key and uses `feed_forward_length` (1024). Whichever key is used, [`Config`]
//!   cross-checks it against the expert bank's own middle dimension.
//! - **RoPE is NeoX-style in both.** `llama_model_rope_type` puts `LLM_ARCH_OLMOE` and
//!   `LLM_ARCH_QWEN3MOE` in the same `LLAMA_ROPE_TYPE_NEOX` arm. The pairs are
//!   `(i, i + n_dims/2)`, not `(2i, 2i+1)`.
//! - **The router softmaxes over all experts before the top-k — and then the two differ.**
//!   `build_moe_ffn` is called with `norm_w = false` for OLMoE, so its selected weights are raw
//!   probabilities that do not sum to one, and with `norm_w = true` for Qwen3, which divides them
//!   by their sum (clamped up to `6.103515625e-5`, f16's smallest normal). Getting this backwards
//!   rescales every expert's contribution by one factor per block: finite, fluent, wrong.
//! - **`w_scale` is `hparams.expert_weights_scale`, which neither architecture sets**, so it
//!   keeps its `0.0f` default and `build_moe_ffn`'s `if (w_scale != 0.0f && w_scale != 1.0f)`
//!   guard skips the scaling entirely.
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
//! sized to the largest that bank reaches in any block — both files quantise `ffn_down_exps` at
//! Q6_K in half their blocks and Q4_K in the rest, and a slot has to hold either. 🔴 That makes
//! a full pool *larger than the bank it holds*: on Qwen3-30B-A3B, 6144 slots of 2.92 MiB commit
//! 17.51 GiB to store 16.35 GiB of experts. The 1.16 GiB difference is not waste to be tuned
//! away — it is the price of a slot that can hold any block — but it must be counted, because a
//! budget built from `expert_bytes` rather than `slot_bytes` will over-promise by 7%.
//!
//! # The expert FFN, in three launches
//!
//! The router names `n_expert_used` experts and each has three weight matrices, so the obvious
//! shape is a launch per expert per bank — 384 of them a token on OLMoE, plus a SwiGLU and
//! an axpy each — 656 launches a token in all. That is now **four a block**, 64 a token: the
//! gate and up projections of every active expert in one, a SwiGLU over all of them, the down
//! projections in a second matvec, and a weighted combine that replaces both the zeroing pass
//! and the run of axpys. (Five, in a block whose gate and up banks are quantised differently and
//! so cannot share a launch. No block of this file is.)
//!
//! 🔴 The reason is **parallelism, not launch count**. A submission costs 1.6 us, which against a
//! 13 ms token is a rounding error. What matters is that one expert's gate matvec is 1024 rows —
//! roughly one pass over a B580's resident threads — and a kernel one wave deep has no second
//! wave to run while the first waits on memory. Measured on OLMoE: the expert FFN fell from
//! 8.39 ms a token to 3.83, and decode went from 63.8 tok/s to 76.7, with every greedy token id
//! unchanged. ⚠️ Qwen3's experts are **768** rows, narrower still, so the same argument applies
//! harder there and the numbers above are not transferable to it.
//!
//! What batching did **not** do is saturate the card. The expert matvecs move 465 MiB a token,
//! at 68 GB/s before and 133 GB/s after, against a peak of 456 GB/s — so they are still under a
//! third of what the memory system will give. Two experiments that did *not* help are worth
//! knowing about before anyone repeats them: carrying two or four rows per lane so a lane loads
//! the activation once and spends it several times (neutral at two, 4% worse at four — the lost
//! waves cost exactly what the saved loads bought), and widening or narrowing the rows a
//! work-group covers (within 4% across 2, 4, 8 and 16).
//!
//! # What is still slow, and known to be
//!
//! Three things, all deliberate and all measured — run the `profile_decode` example for the
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

use std::sync::Arc;

use moearc_kernels::{
    Context, DeviceBuffer, Gating, KernelError, KvType, MAX_BATCHED_MATS, QuantType, RopeKind,
    RopeScaling,
};
use moearc_model::gguf::Value;
use moearc_model::tensors::{ExpertBank, MappedModel, TensorView, names};
use moearc_model::{ModelError, ModelInfo};

use crate::cache::{CacheError, CacheStats, ExpertCache, Load, Slot, StepPlan};
use crate::host_experts::{
    self, BankSpec, BlockSpec, Geometry, HostError, HostExecutor, HostPolicy, HostStats,
};
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
    Host(HostError),
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
impl From<HostError> for EngineError {
    fn from(e: HostError) -> Self {
        Self::Host(e)
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
            Self::Host(e) => write!(f, "{e}"),
            Self::Unsupported(m) => write!(f, "unsupported model: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Which architecture's graph a file gets.
///
/// 🔴 Adding an arm is a deliberate act, not a formality. Every entry here has been read out of
/// llama.cpp's `src/models/<arch>.cpp` and its `build_moe_ffn` call, and the differences that
/// matter are the ones no tensor name reveals — see the module header. An architecture that is
/// *nearly* one of these still belongs in a new arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// `olmoe` — OLMoE-1B-7B: 16 blocks, 64 experts, whole-vector QK-norm, unnormalised router.
    Olmoe,
    /// `qwen3moe` — Qwen3-30B-A3B and siblings: real GQA, per-head QK-norm, normalised router.
    Qwen3Moe,
    /// `gpt-oss` — GPT-OSS-20B/120B. The same skeleton as the other two and different in six
    /// places, every one of which runs and is wrong if left out: **biases on every projection
    /// and every expert bank**, a **per-head attention sink**, **no QK-norm**, an
    /// **alpha-scaled, clamped SwiGLU with a `+1` on the up branch**, a router that
    /// **softmaxes after the top-k rather than before**, and **YaRN RoPE that engages from
    /// position 0**. Its experts are **MXFP4**, which is not a K-quant.
    ///
    /// ⚠️ It also declares `attention.sliding_window = 128`, which this pass does **not**
    /// implement — see [`Config::n_swa`].
    GptOss,
}

/// What an expert's gate and up projections feed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// `silu(gate) * up` — the ordinary gated FFN.
    Swiglu,
    /// `ggml_swiglu_oai`: `(min(g, limit) * sigmoid(alpha * min(g, limit))) *
    /// (clamp(u, -limit, limit) + 1)`.
    ///
    /// 🔴 `alpha` and `limit` are hard-coded constants at llama.cpp's `LLM_FFN_SWIGLU_OAI_MOE`
    /// call site, not GGUF keys, so they are carried here rather than read from the file.
    SwigluOai { alpha: f32, limit: f32 },
}

/// The model's geometry, read from the GGUF header — never assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `general.architecture`, verbatim.
    pub arch: String,
    /// Which graph [`Config::from_model`] selected for it.
    pub kind: Arch,
    pub n_block: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// 🔴 Read from `attention.key_length`, **not** computed as `n_embd / n_head`. On
    /// Qwen3-30B-A3B those are 128 and 64, and the computed one is fluent nonsense.
    pub head_dim: usize,
    /// Channels of each head that rotate — `n_rot` in llama.cpp. Defaults to `head_dim`.
    pub n_rot: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// Whether QK-norm normalises each head on its own (`qwen3moe`) or the whole projection at
    /// once (`olmoe`). Checked against the length of `attn_q_norm.weight`. Meaningless when
    /// [`Config::has_qk_norm`] is false.
    pub qk_norm_per_head: bool,
    /// Whether the architecture normalises Q and K at all. `gpt-oss` does not.
    pub has_qk_norm: bool,
    /// How the router turns logits into weights — llama.cpp's `norm_w` and `gating_op`
    /// together.
    ///
    /// 🔴 Not derivable from the file. It is a property of llama.cpp's `build_moe_ffn` call
    /// site, and nothing in a GGUF records it — which is exactly why architectures are
    /// allowlisted above.
    pub gating: Gating,
    /// The expert activation. Also a call-site property, also unrecorded in the file.
    pub act: Activation,
    /// YaRN, when the file declares `rope.scaling.type = yarn`.
    ///
    /// 🔴 `Some` means it applies at **every** position, including the first. llama.cpp's
    /// `rope_yarn` has no position gate; see [`RopeScaling`].
    pub rope_scaling: Option<RopeScaling>,
    /// `attention.sliding_window`, when the file declares one.
    ///
    /// 🔴 This pass does **not** implement sliding-window attention. The value is carried so
    /// that a session longer than the window can be refused by name rather than silently
    /// attending to keys llama.cpp would have masked. Below the window an SWA mask and a plain
    /// causal mask are the same mask — `is_masked_swa` masks on `p1 - p0 >= n_swa`, which no
    /// pair of positions inside one window satisfies — so a short context is exact, not
    /// approximate.
    pub n_swa: Option<usize>,
    /// The per-block suffix of the pre-FFN norm. `gpt-oss` spells it
    /// `post_attention_norm.weight`; everything else here spells it `ffn_norm.weight`.
    pub ffn_norm: &'static str,
    /// Whether the Q, K, V and output projections carry biases.
    pub has_attn_bias: bool,
    /// Whether each block carries a per-head attention sink.
    pub has_sinks: bool,
    /// Whether the router logits carry a bias, added **before** the top-k.
    pub has_router_bias: bool,
    /// Whether each expert bank carries a per-expert bias.
    pub has_expert_bias: bool,
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
    /// 🔴 Only the architectures in [`Arch`] are accepted. Every other one differs somewhere the
    /// graph below would get silently wrong — a shared expert, a sigmoid router, another RoPE
    /// convention — and producing fluent nonsense is worse than refusing.
    ///
    /// Where a value can be taken two ways, this takes the one the file states and then checks it
    /// against a tensor that must agree. `head_dim`, the expert FFN width and the QK-norm span
    /// are all like that, and all three differ between the two supported architectures.
    pub fn from_model(model: &MappedModel) -> Result<Self, EngineError> {
        let arch = model.architecture()?.to_string();
        let kind = match arch.as_str() {
            "olmoe" => Arch::Olmoe,
            "qwen3moe" => Arch::Qwen3Moe,
            "gpt-oss" => Arch::GptOss,
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "this forward pass implements `olmoe`, `qwen3moe` and `gpt-oss`; the file \
                     declares `{arch}`"
                )));
            }
        };
        let h = model.header();
        let need = |k: &str| -> Result<u64, EngineError> {
            h.u64_key(&format!("{arch}.{k}")).map_err(EngineError::Model)
        };

        let n_block = need("block_count")? as usize;
        let n_embd = need("embedding_length")? as usize;
        let n_head = need("attention.head_count")? as usize;
        let n_head_kv = need("attention.head_count_kv")? as usize;
        if n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 {
            return Err(EngineError::Unsupported(format!(
                "{n_head} query heads over {n_head_kv} key/value heads is not a grouping"
            )));
        }

        // 🔴 `attention.key_length` first, the quotient only as llama.cpp's own fallback.
        // Qwen3-30B-A3B states 128 where the quotient is 64: computing it would halve every
        // head, and the model would still emit fluent text.
        let head_dim = match u64_key_opt(model, &format!("{arch}.attention.key_length")) {
            Some(v) => v as usize,
            None if n_embd % n_head == 0 => n_embd / n_head,
            None => {
                return Err(EngineError::Unsupported(format!(
                    "no attention.key_length, and embedding length {n_embd} is not divisible by \
                     {n_head} heads"
                )));
            }
        };
        if head_dim == 0 {
            return Err(EngineError::Unsupported("attention.key_length is zero".to_string()));
        }
        // One `head_dim` runs through the KV pool's layout, `attn_decode` and `kq_scale`. A file
        // whose values are a different width than its keys is not a tuning difference; it is a
        // model this pass has not been written for.
        let v_len = u64_key_opt(model, &format!("{arch}.attention.value_length"))
            .map_or(head_dim, |v| v as usize);
        if v_len != head_dim {
            return Err(EngineError::Unsupported(format!(
                "key length {head_dim} and value length {v_len} differ; this pass assumes one \
                 head dimension"
            )));
        }
        let n_rot = u64_key_opt(model, &format!("{arch}.rope.dimension_count"))
            .map_or(head_dim, |v| v as usize);

        // The width of one expert's hidden layer. `qwen3moe` states it separately and its
        // `feed_forward_length` (6144) describes a dense FFN it does not have; using that would
        // size every scratch buffer eight times too large and read past every expert's rows.
        let n_ff = match kind {
            Arch::Olmoe => need("feed_forward_length")? as usize,
            // 🔴 `gpt-oss` states both keys and they happen to be equal (2880) on the 120B, so
            // this file would run either way. `load_arch_hparams` reads `n_ff_exp`, so this
            // does too: an equality that holds in one checkpoint is not a reason to read the
            // other key.
            Arch::Qwen3Moe | Arch::GptOss => need("expert_feed_forward_length")? as usize,
        };
        // The bank's own middle dimension is the same number, so disagreement means the key was
        // read wrongly — checked rather than trusted, because the failure is silent.
        let gate0 = model.block_tensor(0, names::FFN_GATE_EXPS)?;
        let banked = gate0.dims.get(1).copied().unwrap_or(0) as usize;
        if banked != n_ff {
            return Err(EngineError::Unsupported(format!(
                "the header says an expert is {n_ff} wide, but `{}` is {banked}",
                gate0.name
            )));
        }

        let n_expert = need("expert_count")? as usize;
        let n_expert_used = need("expert_used_count")? as usize;
        // The expert FFN issues one batched launch per bank, and the kernel carries its weight
        // pointers as an argument. Refusing here is the alternative to silently computing the
        // first `MAX_BATCHED_MATS` experts and dropping the rest.
        if n_expert_used > MAX_BATCHED_MATS {
            return Err(EngineError::Unsupported(format!(
                "{n_expert_used} active experts; a batched expert matvec covers at most \
                 {MAX_BATCHED_MATS}"
            )));
        }
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

        // 🔴 The graph switches that no GGUF key records. Every one comes from llama.cpp's
        // source — `load_arch_tensors`' norm shapes, and `build_moe_ffn`'s `norm_w`,
        // `gating_op` and `type_op` arguments — and they are the reason `Arch` is an allowlist
        // rather than a hint. A file that is *nearly* one of these still needs a new arm.
        let (qk_norm_per_head, has_qk_norm, gating, act, ffn_norm) = match kind {
            Arch::Olmoe => (false, true, Gating::Softmax, Activation::Swiglu, names::FFN_NORM),
            Arch::Qwen3Moe => {
                (true, true, Gating::SoftmaxNormalised, Activation::Swiglu, names::FFN_NORM)
            }
            // `alpha` and `limit` are `constexpr` at llama.cpp's `LLM_FFN_SWIGLU_OAI_MOE` case,
            // not hparams, so they are transcribed rather than read.
            Arch::GptOss => (
                false,
                false,
                Gating::SoftmaxAfterTopK,
                Activation::SwigluOai { alpha: 1.702, limit: 7.0 },
                names::POST_ATTENTION_NORM,
            ),
        };
        let n_embd_q = n_head * head_dim;
        let n_embd_kv = n_head_kv * head_dim;
        // The QK-norm span is checkable, so it is checked. `attn_q_norm.weight` is `head_dim`
        // long under per-head normalisation and as wide as the projection otherwise; a file that
        // disagrees would be normalised over the wrong axis and never say so.
        if has_qk_norm {
            for (suffix, want) in [
                (names::ATTN_Q_NORM, if qk_norm_per_head { head_dim } else { n_embd_q }),
                (names::ATTN_K_NORM, if qk_norm_per_head { head_dim } else { n_embd_kv }),
            ] {
                let t = model.block_tensor(0, suffix)?;
                let got = t.dims.iter().product::<u64>() as usize;
                if got != want {
                    return Err(EngineError::Unsupported(format!(
                        "`{}` is {got} long; `{arch}` normalises {} and needs {want}",
                        t.name,
                        if qk_norm_per_head { "each head" } else { "the whole projection" },
                    )));
                }
            }
        } else if model.optional_block_tensor(0, names::ATTN_Q_NORM)?.is_some() {
            // The switch says this architecture has no QK-norm and the file disagrees. Rather
            // than silently ignore a weight the model was trained with, say so.
            return Err(EngineError::Unsupported(format!(
                "`{arch}` is implemented without QK-norm, but the file carries \
                 `blk.0.{}`",
                names::ATTN_Q_NORM
            )));
        }

        // 🔴 The optional tensors are discovered, not assumed — and then required to be
        // *consistent across blocks*, because a bias that exists in block 0 and not in block 12
        // would be applied to a third of the model and skipped for the rest.
        let has = |suffix: &str| -> Result<bool, EngineError> {
            Ok(model.optional_block_tensor(0, suffix)?.is_some())
        };
        let has_attn_bias = has(names::ATTN_Q_BIAS)?;
        let has_sinks = has(names::ATTN_SINKS)?;
        let has_router_bias = has(names::FFN_GATE_INP_BIAS)?;
        let has_expert_bias = has(names::FFN_GATE_EXPS_BIAS)?;
        if has_attn_bias {
            for suffix in [names::ATTN_K_BIAS, names::ATTN_V_BIAS, names::ATTN_OUTPUT_BIAS] {
                if !has(suffix)? {
                    return Err(EngineError::Unsupported(format!(
                        "`{arch}` carries `{}` but not `{suffix}`; this pass applies the four \
                         attention biases together or not at all",
                        names::ATTN_Q_BIAS
                    )));
                }
            }
        }
        if has_expert_bias {
            for suffix in [names::FFN_UP_EXPS_BIAS, names::FFN_DOWN_EXPS_BIAS] {
                if !has(suffix)? {
                    return Err(EngineError::Unsupported(format!(
                        "`{arch}` carries `{}` but not `{suffix}`",
                        names::FFN_GATE_EXPS_BIAS
                    )));
                }
            }
        }

        // 🔴 YaRN is not a long-context-only correction. `rope_yarn` interpolates every
        // frequency and rescales every magnitude from position 0; the only thing its ramp
        // consults is the channel index. So this is read whenever the file declares it, and a
        // model that declares it and is run without it is wrong on its first token.
        let rope_freq_base = f32_key(model, &format!("{arch}.rope.freq_base")).unwrap_or(10_000.0);
        let n_ctx_orig =
            u64_key_opt(model, &format!("{arch}.rope.scaling.original_context_length"));
        let scaling_type = match model.header().get(&format!("{arch}.rope.scaling.type")) {
            Some(Value::String(v)) => Some(v.as_str()),
            _ => None,
        };
        let rope_scaling = match (scaling_type, n_ctx_orig) {
            (None, _) => None,
            (Some("none"), _) | (Some("linear"), None) => None,
            (Some("yarn"), Some(orig)) => {
                let factor =
                    f32_key(model, &format!("{arch}.rope.scaling.factor")).ok_or_else(|| {
                        EngineError::Unsupported(
                            "rope.scaling.type is yarn with no rope.scaling.factor".to_string(),
                        )
                    })?;
                // llama.cpp's own defaults when the keys are absent
                // (`llama_model_default_params`): beta_fast 32, beta_slow 1.
                let beta_fast =
                    f32_key(model, &format!("{arch}.rope.scaling.yarn_beta_fast")).unwrap_or(32.0);
                let beta_slow =
                    f32_key(model, &format!("{arch}.rope.scaling.yarn_beta_slow")).unwrap_or(1.0);
                Some(RopeScaling::yarn(
                    n_rot,
                    orig as usize,
                    rope_freq_base,
                    factor,
                    beta_fast,
                    beta_slow,
                ))
            }
            (Some(other), _) => {
                return Err(EngineError::Unsupported(format!(
                    "`{arch}` declares rope.scaling.type = `{other}`; this pass implements \
                     plain RoPE and YaRN"
                )));
            }
        };

        let n_swa = u64_key_opt(model, &format!("{arch}.attention.sliding_window"))
            .filter(|w| *w > 0)
            .map(|w| w as usize);

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
            rope_freq_base,
            qk_norm_per_head,
            has_qk_norm,
            gating,
            act,
            rope_scaling,
            n_swa,
            ffn_norm,
            has_attn_bias,
            has_sinks,
            has_router_bias,
            has_expert_bias,
            bos: u64_key_opt(model, "tokenizer.ggml.bos_token_id").map(|v| v as u32),
            eos: u64_key_opt(model, "tokenizer.ggml.eos_token_id").map(|v| v as u32),
            arch,
            kind,
        })
    }

    /// Width of the Q projection — `n_head * head_dim`, which is **not** `n_embd` when the
    /// architecture states its own `head_dim`. Qwen3-30B-A3B projects 2048 up to 4096.
    fn n_embd_q(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Width of each of the K and V projections.
    fn n_embd_kv(&self) -> usize {
        self.n_head_kv * self.head_dim
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
    /// QK-norm, where the architecture has it.
    attn_q_norm: Option<DeviceBuffer<'c>>,
    attn_k_norm: Option<DeviceBuffer<'c>>,
    /// The four projection biases, applied before RoPE on Q and K and after the output matmul.
    attn_q_bias: Option<DeviceBuffer<'c>>,
    attn_k_bias: Option<DeviceBuffer<'c>>,
    attn_v_bias: Option<DeviceBuffer<'c>>,
    attn_output_bias: Option<DeviceBuffer<'c>>,
    /// One sink logit per **query** head.
    attn_sinks: Option<DeviceBuffer<'c>>,
    ffn_norm: DeviceBuffer<'c>,
    /// `[n_embd, n_expert]`. Uploaded as a matrix rather than assumed f32, so a quantised
    /// router in some other build would still be read through the right kernel.
    ffn_gate_inp: QTensor<'c>,
    /// The router's bias, added to the logits **before** the top-k.
    ffn_gate_inp_bias: Option<DeviceBuffer<'c>>,
    /// The gate and up expert biases, **concatenated** into one buffer of
    /// `[2 * n_expert, n_ff]` f32 rows: gate's experts first, then up's.
    ///
    /// 🔴 One buffer rather than two because the gate and up projections are computed in a
    /// *single* batched launch whose output is one buffer of `2k` matrices, and
    /// [`Context::add_bias_id`] indexes one bank. Concatenating lets the same call bias both
    /// halves — expert `e`'s gate row is `e`, its up row is `n_expert + e` — and keeps the
    /// fusion that the batched matvec exists for.
    gate_up_bias: Option<DeviceBuffer<'c>>,
    /// The down expert bias, `[n_expert, n_embd]`. Applied to each expert's output **before**
    /// the router's weight multiplies it, which is where `ggml_add_id` sits in `build_moe_ffn`.
    down_bias: Option<DeviceBuffer<'c>>,
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
    /// stay. The expert banks do not: they are read from the mapping on demand. That split is
    /// roughly 360 MiB resident against 3.6 GiB pageable on OLMoE-1B-7B, and 951 MiB against
    /// 16.35 GiB on Qwen3-30B-A3B — where the pageable half alone exceeds the card.
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
                cfg.ffn_norm,
                names::FFN_GATE_INP,
            ] {
                bytes += model.block_tensor(b, s)?.data.len() as u64;
            }
            // 🔴 Counted from what the file actually carries, not from the architecture's
            // switches. gpt-oss's expert biases alone are 159 MiB across 36 blocks — f32, and
            // therefore resident rather than streamed — and a `dense_bytes` that omitted them
            // would under-report the always-resident half by that much, which is exactly the
            // number a residency plan divides the card by.
            for s in [
                names::ATTN_Q_NORM,
                names::ATTN_K_NORM,
                names::ATTN_Q_BIAS,
                names::ATTN_K_BIAS,
                names::ATTN_V_BIAS,
                names::ATTN_OUTPUT_BIAS,
                names::ATTN_SINKS,
                names::FFN_GATE_INP_BIAS,
                names::FFN_GATE_EXPS_BIAS,
                names::FFN_UP_EXPS_BIAS,
                names::FFN_DOWN_EXPS_BIAS,
            ] {
                if let Some(v) = model.optional_block_tensor(b, s)? {
                    bytes += v.data.len() as u64;
                }
            }
            for s in [names::FFN_GATE_EXPS, names::FFN_UP_EXPS, names::FFN_DOWN_EXPS] {
                expert_bytes += model.block_tensor(b, s)?.data.len() as u64;
            }

            let simple = |suffix: &str| -> Result<DeviceBuffer<'c>, EngineError> {
                upload(ctx, &model.block_tensor(b, suffix)?)
            };
            // Present-or-absent is a fact about the file; `Config` has already refused a file
            // whose blocks disagree with the architecture about which of these exist.
            let opt = |suffix: &str| -> Result<Option<DeviceBuffer<'c>>, EngineError> {
                match model.optional_block_tensor(b, suffix)? {
                    Some(v) => Ok(Some(upload(ctx, &v)?)),
                    None => Ok(None),
                }
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

            // The gate and up expert biases, laid end to end so one `add_bias_id` covers the
            // fused launch. Concatenated on the host because a `DeviceBuffer` is written from
            // its start and this crate has no offset upload; it is 2.9 MiB per block, once, at
            // load.
            let gate_up_bias = match (
                model.optional_block_tensor(b, names::FFN_GATE_EXPS_BIAS)?,
                model.optional_block_tensor(b, names::FFN_UP_EXPS_BIAS)?,
            ) {
                (Some(gb), Some(ub)) => {
                    let mut joined = Vec::with_capacity(gb.data.len() + ub.data.len());
                    joined.extend_from_slice(gb.data);
                    joined.extend_from_slice(ub.data);
                    let buf = ctx.alloc(joined.len())?;
                    ctx.upload(&buf, &joined)?;
                    Some(buf)
                }
                _ => None,
            };

            blocks.push(Block {
                attn_norm: simple(names::ATTN_NORM)?,
                attn_q: matrix(names::ATTN_Q)?,
                attn_k: matrix(names::ATTN_K)?,
                attn_v: matrix(names::ATTN_V)?,
                attn_output: matrix(names::ATTN_OUTPUT)?,
                attn_q_norm: opt(names::ATTN_Q_NORM)?,
                attn_k_norm: opt(names::ATTN_K_NORM)?,
                attn_q_bias: opt(names::ATTN_Q_BIAS)?,
                attn_k_bias: opt(names::ATTN_K_BIAS)?,
                attn_v_bias: opt(names::ATTN_V_BIAS)?,
                attn_output_bias: opt(names::ATTN_OUTPUT_BIAS)?,
                attn_sinks: opt(names::ATTN_SINKS)?,
                ffn_norm: simple(cfg.ffn_norm)?,
                ffn_gate_inp: matrix(names::FFN_GATE_INP)?,
                ffn_gate_inp_bias: opt(names::FFN_GATE_INP_BIAS)?,
                gate_up_bias,
                down_bias: opt(names::FFN_DOWN_EXPS_BIAS)?,
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
    ///
    /// ⚠️ It is also the **default, and it is not always achievable**. A full pool is
    /// `n_block * n_expert * slot_bytes`, which on Qwen3-30B-A3B is 17.51 GiB against a B580's
    /// 11.33 GiB. Asking for it on such a model fails, which is the honest outcome — the
    /// alternative would be silently choosing a budget the caller did not ask for. Use
    /// [`Residency::Planned`], or [`Residency::Slots`], on a model that does not fit.
    ///
    /// 🔴 **But it does not fail where you would expect, and that is a property of the runtime,
    /// not of this code.** `ExpertPool::new` calls `malloc_device` once per slot per bank — 9,300
    /// allocations at 3,100 slots — and on the Level Zero runtime on this box **every one of them
    /// returns a valid pointer well past the point where the memory exists**. The load reports
    /// success and prints a pool size; the failure arrives on the first token, as a
    /// host-to-device copy or a kernel launch that fails, and near the boundary the driver can
    /// spin at 100% of a core for minutes before it says so. Measured on a B580 that reports
    /// 11.33 GiB, at `n_ctx = 512` and 951 MiB of dense weights:
    ///
    /// ```text
    ///   3050 slots   8899 MiB pool   9.67 GiB committed   runs
    ///   3100 slots   9045 MiB pool   9.81 GiB committed   "embedding lookup failed"
    ///   3157 slots   9212 MiB pool   9.97 GiB committed   "host-to-device copy failed"
    /// ```
    ///
    /// So the usable ceiling is about **85% of what the device reports free**, and no allocation
    /// return value discovers it. See [`crate::memory::Headroom::PROVISIONAL`], whose 12% is the
    /// only thing standing between a plan and this cliff — and which lands 3157 slots, one row
    /// into it.
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

/// A [`Residency`] written the way a command line writes one.
///
/// One spelling, shared by every tool, so a sweep row and a probe run cannot mean different
/// things by the same word:
///
/// | Spec | Meaning |
/// |---|---|
/// | `all` | every slot resident — see the warning on [`Residency::All`] |
/// | `<n>` | `n` slots, LRU |
/// | `plan:<bytes>` | whatever [`crate::memory::plan`] decides for a device with that much free |
/// | `static:<blocks>` | the incumbent: `blocks` blocks pinned, the rest streamed |
impl std::str::FromStr for Residency {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "all" {
            return Ok(Self::All);
        }
        if let Some(rest) = s.strip_prefix("plan:") {
            let free: u64 = rest.parse().map_err(|_| format!("`{rest}` is not a byte count"))?;
            return Ok(Self::Planned(memory::DeviceMemory { total_bytes: free, free_bytes: free }));
        }
        if let Some(rest) = s.strip_prefix("static:") {
            let resident_blocks =
                rest.parse().map_err(|_| format!("`{rest}` is not a block count"))?;
            return Ok(Self::StaticSplit { resident_blocks });
        }
        s.parse().map(Self::Slots).map_err(|_| {
            format!("`{s}` is not a residency: expected `all`, `<slots>`, `plan:<bytes>` or `static:<blocks>`")
        })
    }
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

    /// Whether this expert is in VRAM right now — the peek [`crate::cache::ExpertCache::resident`]
    /// provides for the LRU, so the host policy can ask both the same question.
    fn resident(&self, e: ExpertRef) -> bool {
        if e.layer < self.resident_blocks {
            let slot = u32::from(e.layer) * self.n_expert + u32::from(e.expert);
            return self.pinned_filled[slot as usize];
        }
        self.ring.contains(&Some(e))
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

    /// Whether `e` is resident, asked without committing anything.
    fn resident(&self, e: ExpertRef) -> bool {
        match self {
            Self::Lru(c) => c.resident(e),
            Self::Static(s) => s.resident(e),
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
    /// The expert FFN's intermediates, `n_expert_used` of each laid end to end: every active
    /// expert's gate, up, activation and output vector, produced by one launch apiece rather
    /// than one launch per expert.
    gate: DeviceBuffer<'c>,
    up: DeviceBuffer<'c>,
    act: DeviceBuffer<'c>,
    expert_out: DeviceBuffer<'c>,
    ffn: DeviceBuffer<'c>,
    /// Where the host executor's contribution lands before it is added to `ffn`.
    cpu_ffn: DeviceBuffer<'c>,
    /// The compacted router weights for a device batch that is a subset of the router's choice.
    weights_sub: DeviceBuffer<'c>,
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
    /// The block's post-norm activation, on the host, because the CPU experts need it. Only
    /// filled when a block actually routes something host-side.
    x_host: Vec<f32>,
    /// What the host executor returns: the router-weighted sum of its experts' outputs.
    cpu_out: Vec<f32>,
    /// `(expert id, router weight)` for the experts this block sends to the CPU, in router
    /// order.
    cpu_pick: Vec<(u16, f32)>,
    /// The experts left for the device, in router order — a *subset* of `wanted` when the host
    /// executor is on.
    gpu_wanted: Vec<ExpertRef>,
    /// Those experts' router weights, compacted to match. 🔴 `moe_combine` reads `weights[m]`
    /// for the `m`-th matrix it was handed, so a device batch that is a non-prefix subset of the
    /// router's choice cannot use the router's own weight vector: the scalars would be paired
    /// with the wrong experts and the output would stay finite and fluent.
    w_sub: Vec<f32>,
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
        // 🔴 The Q side is `n_head * head_dim`, not `n_embd`. They are equal on OLMoE and differ
        // by 2x on Qwen3-30B-A3B, where a buffer sized `n_embd` would have the QK-norm, the RoPE
        // and the attention output all reading half a projection.
        let n_embd_q = cfg.n_embd_q();
        let n_embd_kv = cfg.n_embd_kv();
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
            q: ctx.alloc_n::<f32>(n_embd_q)?,
            k: ctx.alloc_n::<f32>(n_embd_kv)?,
            v: ctx.alloc_n::<f32>(n_embd_kv)?,
            q_normed: ctx.alloc_n::<f32>(n_embd_q)?,
            k_normed: ctx.alloc_n::<f32>(n_embd_kv)?,
            q_roped: ctx.alloc_n::<f32>(n_embd_q)?,
            k_roped: ctx.alloc_n::<f32>(n_embd_kv)?,
            attn: ctx.alloc_n::<f32>(n_embd_q)?,
            proj: ctx.alloc_n::<f32>(n_embd)?,
            router: ctx.alloc_n::<f32>(cfg.n_expert)?,
            idx: ctx.alloc_n::<u32>(cfg.n_expert_used)?,
            weights: ctx.alloc_n::<f32>(cfg.n_expert_used)?,
            gate: ctx.alloc_n::<f32>(2 * cfg.n_expert_used * cfg.n_ff)?,
            up: ctx.alloc_n::<f32>(cfg.n_expert_used * cfg.n_ff)?,
            act: ctx.alloc_n::<f32>(cfg.n_expert_used * cfg.n_ff)?,
            expert_out: ctx.alloc_n::<f32>(cfg.n_expert_used * n_embd)?,
            ffn: ctx.alloc_n::<f32>(n_embd)?,
            cpu_ffn: ctx.alloc_n::<f32>(n_embd)?,
            weights_sub: ctx.alloc_n::<f32>(cfg.n_expert_used)?,
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
            x_host: vec![0.0; n_embd],
            cpu_out: vec![0.0; n_embd],
            cpu_pick: Vec::with_capacity(cfg.n_expert_used),
            gpu_wanted: Vec::with_capacity(cfg.n_expert_used),
            w_sub: Vec::with_capacity(cfg.n_expert_used),
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
    /// The host-side expert executor, present only when a policy asked for one.
    host: Option<HostExecutor>,
    host_policy: HostPolicy,
    /// Each block's expert geometry, in the shape the executor wants. Built once because it is
    /// per block and does not change.
    host_specs: Vec<BlockSpec>,
}

impl<'c, 'm> Model<'c, 'm> {
    /// Load with no host executor: every miss is streamed. This is the engine as it was.
    pub fn new(
        ctx: &'c Context,
        model: &'m MappedModel,
        n_ctx: usize,
        residency: Residency,
    ) -> Result<Self, EngineError> {
        Self::build(ctx, model, None, n_ctx, residency, HostPolicy::Off)
    }

    /// Load with a host-side expert executor.
    ///
    /// 🔴 The `Arc` is not decoration. The executor's worker threads read expert weights straight
    /// out of the mapping, so the mapping has to outlive them — and a `&MappedModel` cannot
    /// express that to threads whose lifetime the borrow checker never sees. Sharing ownership
    /// does, and [`HostExecutor::drop`] joins the workers before its clone is released.
    pub fn new_hybrid(
        ctx: &'c Context,
        model: &'m Arc<MappedModel>,
        n_ctx: usize,
        residency: Residency,
        host: HostPolicy,
    ) -> Result<Self, EngineError> {
        Self::build(ctx, model, Some(Arc::clone(model)), n_ctx, residency, host)
    }

    fn build(
        ctx: &'c Context,
        model: &'m MappedModel,
        owned: Option<Arc<MappedModel>>,
        n_ctx: usize,
        residency: Residency,
        host_policy: HostPolicy,
    ) -> Result<Self, EngineError> {
        let weights = Weights::upload(ctx, model)?;
        let cfg = &weights.cfg;
        let n_slots = weights.n_slots();

        // 🔴 Sliding-window attention is declared by `gpt-oss` and **not implemented here**.
        //
        // Below the window it does not have to be: llama.cpp masks a key when
        // `p1 - p0 >= n_swa`, and no pair of positions inside a context of `n_swa` tokens
        // satisfies that, so an SWA mask and a plain causal mask are the same mask and this
        // pass is exact rather than approximate. Above it they diverge, silently and
        // progressively — the first token past the window attends to one key llama.cpp has
        // dropped, and the divergence grows from there.
        //
        // ⚠️ It is also **alternating**: `set_swa_pattern(2)` makes even blocks windowed and
        // odd blocks full causal, so implementing it means two masks and two KV caches, not
        // one shorter cache.
        if cfg.n_swa.is_some_and(|w| n_ctx > w) {
            let w = cfg.n_swa.unwrap_or_default();
            return Err(EngineError::Unsupported(format!(
                "`{}` uses sliding-window attention with a {w}-token window on alternating \
                 blocks, which this forward pass does not implement; a context of {n_ctx} \
                 tokens would diverge from llama.cpp past position {w}. Load with n_ctx <= {w}.",
                cfg.arch
            )));
        }

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

        let bank = |b: &BankShape| BankSpec { ty: b.ty, n_rows: b.n_rows, n_cols: b.n_cols };
        let host_specs: Vec<BlockSpec> = weights
            .blocks
            .iter()
            .map(|b| BlockSpec { gate: bank(&b.gate), up: bank(&b.up), down: bank(&b.down) })
            .collect();
        let host = match (owned, host_policy.is_off()) {
            (Some(mapped), false) => Some(HostExecutor::new(
                mapped,
                Geometry {
                    n_block: cfg.n_block,
                    n_expert: cfg.n_expert,
                    n_expert_used: cfg.n_expert_used,
                    n_embd: cfg.n_embd,
                    n_ff: cfg.n_ff,
                    // 🔴 The host must compute the same function as the device or the model's
                    // output would depend on the host policy, which is a performance knob.
                    expert_bias: cfg.has_expert_bias,
                    act: cfg.act,
                },
                &host_specs,
                host_experts::default_threads(),
            )?),
            _ => None,
        };

        Ok(Self {
            ctx,
            mapped: model,
            weights,
            state,
            pool,
            admission,
            host,
            host_policy,
            host_specs,
        })
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
            host_threads: self.host.as_ref().map_or(0, HostExecutor::n_threads),
            host: self.host.as_ref().map(HostExecutor::stats).unwrap_or_default(),
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
        if let Some(h) = &self.host {
            h.reset_stats();
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
        let host = self.host.as_ref();
        let host_policy = self.host_policy;
        let host_specs = &self.host_specs;
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
        let n_embd_q = cfg.n_embd_q();
        let n_embd_kv = cfg.n_embd_kv();
        let eps = cfg.rms_eps;
        let scale = cfg.kq_scale();

        // h = token_embd[token]
        {
            let _p = profile::scope("embed");
            ctx.embed_rows(w.token_embd.ty, &st.h, &w.token_embd.buf, &st.tok, 1, n_embd)?;
        }

        // Reused by every block: `slots.len()` pool buffers, one per active expert.
        let mut batch: Vec<&DeviceBuffer<'c>> = Vec::with_capacity(cfg.n_expert_used);

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
                // 🔴 Before RoPE, and before QK-norm where there is one — `build_qkv` adds the
                // bias to the projection immediately and everything downstream sees the sum.
                // `add` writes each index from the same index of both inputs, so aliasing the
                // destination into the left operand is well defined.
                if let Some(bias) = &b.attn_q_bias {
                    ctx.add(&st.q, &st.q, bias, n_embd_q)?;
                }
                if let Some(bias) = &b.attn_k_bias {
                    ctx.add(&st.k, &st.k, bias, n_embd_kv)?;
                }
                if let Some(bias) = &b.attn_v_bias {
                    ctx.add(&st.v, &st.v, bias, n_embd_kv)?;
                }
            }

            // QK-norm, after the projections and before RoPE — where the architecture has it.
            //
            // 🔴 What one norm covers is architecture-specific, and both spellings are the same
            // kernel with different row counts: `qwen3moe` normalises each head over `head_dim`
            // channels with a `head_dim`-wide weight broadcast across heads, `olmoe` normalises
            // the whole projection in one row. Neither raises an error on the other's model —
            // and `gpt-oss` has neither, which is a third silent difference: normalising a
            // projection that was not trained normalised rescales every head.
            let (q_src, k_src) = if cfg.has_qk_norm {
                let _p = profile::scope("attn.qk_norm");
                let (q_rows, q_cols, k_rows, k_cols) = if cfg.qk_norm_per_head {
                    (cfg.n_head, cfg.head_dim, cfg.n_head_kv, cfg.head_dim)
                } else {
                    (1, n_embd_q, 1, n_embd_kv)
                };
                let (qn, kn) = (
                    b.attn_q_norm.as_ref().expect("has_qk_norm implies the weight was uploaded"),
                    b.attn_k_norm.as_ref().expect("has_qk_norm implies the weight was uploaded"),
                );
                ctx.rmsnorm(&st.q_normed, &st.q, Some(qn), q_rows, q_cols, eps)?;
                ctx.rmsnorm(&st.k_normed, &st.k, Some(kn), k_rows, k_cols, eps)?;
                (&st.q_normed, &st.k_normed)
            } else {
                (&st.q, &st.k)
            };

            {
                let _p = profile::scope("attn.rope");
                ctx.rope_ext(
                    &st.q_roped,
                    q_src,
                    &st.pos,
                    1,
                    cfg.n_head,
                    cfg.head_dim,
                    cfg.n_rot,
                    cfg.rope_freq_base,
                    cfg.rope_scaling,
                    RopeKind::Neox,
                )?;
                ctx.rope_ext(
                    &st.k_roped,
                    k_src,
                    &st.pos,
                    1,
                    cfg.n_head_kv,
                    cfg.head_dim,
                    cfg.n_rot,
                    cfg.rope_freq_base,
                    cfg.rope_scaling,
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
                ctx.attn_decode_ext(
                    &st.attn,
                    &st.q_roped,
                    &st.k_pages[bi],
                    &st.v_pages[bi],
                    &st.block_table,
                    b.attn_sinks.as_ref(),
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
                if let Some(bias) = &b.attn_output_bias {
                    ctx.add(&st.proj, &st.proj, bias, n_embd)?;
                }
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
            if let Some(bias) = &b.ffn_gate_inp_bias {
                // 🔴 Before the top-k, not after. `build_moe_ffn` adds `gate_inp_b` to the
                // logits and *then* selects, so the bias changes which experts run, not only how
                // much each contributes.
                let _p = profile::scope("moe.router");
                ctx.add(&st.router, &st.router, bias, cfg.n_expert)?;
                // ⚠️ Tapped under llama.cpp's name for it, which is **`ffn_moe_probs`** and not
                // `ffn_moe_logits_biased`. Both `cb()` calls name the same node — `probs =
                // logits` under `SOFTMAX_WEIGHT` is an assignment, not an operation — and the
                // second rename wins, so the earlier name never reaches a dump. This tap is
                // emitted only where a router bias exists, which is the only case in which
                // `ffn_moe_probs` means the biased logits rather than a softmax of them.
                tap_record(
                    &mut st.tap,
                    format!("ffn_moe_probs-{bi}"),
                    ctx,
                    &st.router,
                    cfg.n_expert,
                )?;
            }
            // [`Gating`] is llama.cpp's `norm_w` and `gating_op` together: raw softmax
            // probabilities that do not sum to one for OLMoE, the same divided by their sum for
            // Qwen3 (clamped up to `6.103515625e-5`, exactly as `ggml_clamp` does), and for
            // gpt-oss a softmax over the k selected logits taken *after* the top-k. All three
            // select the same experts and weight them differently.
            {
                let _p = profile::scope("moe.topk");
                ctx.topk_router(
                    &st.idx,
                    &st.weights,
                    &st.router,
                    1,
                    cfg.n_expert,
                    cfg.n_expert_used,
                    cfg.gating,
                )?;
            }
            {
                let _p = profile::scope("moe.readback");
                ctx.download_slice(&mut st.idx_host, &st.idx)?;
                // The router's weights are not read back in the serving path: they stay on
                // the device, where `moe_combine` reads them out of `st.weights`, and the only
                // consumer of a host copy is the tap. Downloading them unconditionally was
                // simply pointless work, so this is kept — but ⚠️ **it is not an optimisation
                // worth quoting.** Measured on Qwen3-30B-A3B at 2952 slots, 95 steady-state
                // tokens: `moe.readback` 19.01 -> 18.71 ms/token, 22.87 -> 23.17 tok/s. **+1.3%.**
                //
                // 🔴 That near-zero is the useful result. It says the 18.7 ms this phase costs is
                // **not** the copy — 48 calls moving 32 bytes each — it is the *drain*. The queue
                // is asynchronous and in-order, so the first download waits for everything
                // submitted before it, and the second then costs almost nothing because the
                // pipeline is already empty. Removing a second drain that was never happening
                // buys nothing. The 390 us/call is one pipeline stall per block per token, and
                // the only thing that removes it is not reading the router's choice back at all
                // — driving the expert gather from the device. See the note at the top of this
                // file, which called this out as "a device round trip per block" and, on OLMoE
                // with 16 blocks and a resident pool, correctly measured it at ~13 us. At 48
                // blocks with a streaming pool it is 30x that, and it is the largest phase in
                // the step.
                // 🔴 A second consumer, and it is not optional: when part of the block is
                // computed host-side the CPU has to apply the router's weights itself, and the
                // device batch that is left needs its own compacted copy of them.
                if st.tap.is_some() || host.is_some() {
                    ctx.download_slice(&mut st.w_host, &st.weights)?;
                }
            }
            if let Some(t) = st.tap.as_mut() {
                t.items.push((
                    format!("ffn_moe_topk-{bi}"),
                    st.idx_host.iter().map(|i| *i as f32).collect(),
                ));
                t.items.push((format!("ffn_moe_weights-{bi}"), st.w_host.clone()));
            }

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

            // ---- the split: which of this block's misses go to the CPU ---------------------
            //
            // 🔴 The cache is *asked* with `resident` and only then told with `admit`, and that
            // order is the correctness of the whole thing. `admit` plans and commits together —
            // deliberately, see `cache.rs` — so admitting an expert and then declining to stage
            // it would leave the cache certain of a slot that was never filled, and the next hit
            // on it would read whatever the slot held before. Only the experts the device is
            // going to keep are ever admitted.
            st.cpu_pick.clear();
            st.gpu_wanted.clear();
            st.w_sub.clear();
            if host.is_some() {
                let _p = profile::scope("moe.host_split");
                let misses = st.wanted.iter().filter(|e| !admission.resident(**e)).count();
                let mut budget = host_policy.host_count(misses);
                for (i, e) in st.wanted.iter().enumerate() {
                    if budget > 0 && !admission.resident(*e) {
                        st.cpu_pick.push((e.expert, st.w_host[i]));
                        budget -= 1;
                    } else {
                        st.gpu_wanted.push(*e);
                        st.w_sub.push(st.w_host[i]);
                    }
                }
            } else {
                st.gpu_wanted.extend_from_slice(&st.wanted);
            }

            // 🔴 Submitted BEFORE the staging copies and the expert matvecs are issued, which is
            // the entire hypothesis. `submit` returns as soon as the job is published; the host
            // thread then goes on to queue this block's GPU work, and the two run at once. A
            // `sync` here instead of below would make this substitution rather than overlap, and
            // `bench/baselines/qwen3-30b-a3b.md` already records what substitution is worth.
            let job = if st.cpu_pick.is_empty() {
                None
            } else {
                let _p = profile::scope("moe.host_submit");
                // The router readback a few lines up drained the queue, so this download costs
                // the copy and not a fence.
                ctx.download_slice(&mut st.x_host, &st.x)?;
                let h = host.expect("a non-empty pick implies an executor");
                Some(h.submit(bi, host_specs[bi], &st.cpu_pick, &st.x_host)?)
            };

            // The device batch is now a subset of the router's choice, so it needs its own
            // weight vector. Uploaded here, while the queue is still empty, rather than after
            // the staging copies this blocking copy would otherwise have to wait for.
            if !st.cpu_pick.is_empty() {
                ctx.upload_slice(&st.weights_sub, &st.w_sub)?;
            }

            // Ask the cache what is resident and what must move.
            let plan = {
                let _p = profile::scope("moe.admit");
                admission.admit(&st.gpu_wanted)?
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
            let slots = plan.slots_for(&st.gpu_wanted);
            let k = slots.len();

            // 🔴 All k experts in one launch per bank, not one launch per expert per bank.
            //
            // The reason is parallelism, not launch count. One expert's gate matvec is 1024
            // rows, which on a B580 is about one pass over the card's resident threads — a
            // kernel one wave deep, with no second wave to run while the first waits on memory.
            // Measured on this model it moved 1.18 MB in 12 us: 98 GB/s against a 456 GB/s
            // peak. Eight experts in one launch is eight waves deep.
            //
            // The weights come from `slots`, which is the cache's answer for the experts the
            // router named, in router order — so `st.weights`, still holding the router's own
            // probabilities on the device, lines up with them index for index and the combine
            // below needs nothing from the host.
            //
            // Gate and up read the *same* activation vector and differ only in their weights, so
            // when they share a shape and a type they are 2k matrices of one launch rather than
            // two launches of k. `fused` is checked per block because a GGUF may quantise the
            // two banks differently, and this file already sees a model whose blocks disagree
            // with each other about the down bank.
            let fused = b.gate.ty == b.up.ty
                && b.gate.n_rows == b.up.n_rows
                && b.gate.n_cols == b.up.n_cols
                && 2 * k <= MAX_BATCHED_MATS;
            // A block can now leave the device nothing to do — `frac:1.0` at a zero hit rate
            // does exactly that. Zeroing here rather than letting a batch of zero matrices fall
            // through four kernels keeps every one of them a shape they were tested on.
            if k == 0 {
                let _p = profile::scope("moe.combine");
                ctx.zero(&st.ffn, n_embd)?;
            } else {
                {
                    let _p = profile::scope("moe.expert_matvec");
                    bank_batch(&mut batch, &pool.gate, &slots);
                    if fused {
                        for &sl in &slots {
                            batch.push(&pool.up[sl as usize]);
                        }
                    }
                    ctx.matvec_q_batched(
                        b.gate.ty,
                        &st.gate,
                        &batch,
                        &st.x,
                        0,
                        b.gate.n_rows,
                        b.gate.n_cols,
                    )?;
                    if !fused {
                        bank_batch(&mut batch, &pool.up, &slots);
                        ctx.matvec_q_batched(
                            b.up.ty,
                            &st.up,
                            &batch,
                            &st.x,
                            0,
                            b.up.n_rows,
                            b.up.n_cols,
                        )?;
                    }
                    // 🔴 `ggml_add_id` immediately after each `mul_mat_id`, before the
                    // activation. `gate_up_bias` holds both banks end to end, so expert `e`'s
                    // gate row is `e` and its up row is `n_expert + e` — which is what lets one
                    // call bias a fused launch's two halves and the fusion survive.
                    if let Some(bias) = &b.gate_up_bias {
                        let mut bidx = [0u32; MAX_BATCHED_MATS];
                        for (i, e) in st.gpu_wanted.iter().enumerate() {
                            bidx[i] = u32::from(e.expert);
                            if fused {
                                bidx[k + i] = cfg.n_expert as u32 + u32::from(e.expert);
                            }
                        }
                        if fused {
                            ctx.add_bias_id(&st.gate, bias, &bidx[..2 * k], b.gate.n_rows)?;
                        } else {
                            ctx.add_bias_id(&st.gate, bias, &bidx[..k], b.gate.n_rows)?;
                            for (i, e) in st.gpu_wanted.iter().enumerate() {
                                bidx[i] = cfg.n_expert as u32 + u32::from(e.expert);
                            }
                            ctx.add_bias_id(&st.up, bias, &bidx[..k], b.up.n_rows)?;
                        }
                    }
                }
                {
                    let _p = profile::scope("moe.swiglu");
                    match cfg.act {
                        Activation::Swiglu if fused => {
                            ctx.swiglu_halves(&st.act, &st.gate, k * cfg.n_ff)?;
                        }
                        Activation::Swiglu => {
                            ctx.swiglu(&st.act, &st.gate, &st.up, k * cfg.n_ff)?;
                        }
                        Activation::SwigluOai { alpha, limit } if fused => {
                            ctx.swiglu_oai_halves(&st.act, &st.gate, k * cfg.n_ff, alpha, limit)?;
                        }
                        Activation::SwigluOai { alpha, limit } => {
                            ctx.swiglu_oai(&st.act, &st.gate, &st.up, k * cfg.n_ff, alpha, limit)?;
                        }
                    }
                }
                {
                    let _p = profile::scope("moe.expert_down");
                    bank_batch(&mut batch, &pool.down, &slots);
                    ctx.matvec_q_batched(
                        b.down.ty,
                        &st.expert_out,
                        &batch,
                        &st.act,
                        cfg.n_ff,
                        b.down.n_rows,
                        b.down.n_cols,
                    )?;
                    // 🔴 **Inside** the router's weighting, not outside it. `build_moe_ffn` adds
                    // this bias to each expert's output and only then multiplies by that
                    // expert's weight, so folding it into the combined result instead would
                    // scale it by the sum of the weights rather than by each expert's own.
                    if let Some(bias) = &b.down_bias {
                        let mut bidx = [0u32; MAX_BATCHED_MATS];
                        for (i, e) in st.gpu_wanted.iter().enumerate() {
                            bidx[i] = u32::from(e.expert);
                        }
                        ctx.add_bias_id(&st.expert_out, bias, &bidx[..k], b.down.n_rows)?;
                    }
                }
                {
                    // Writes rather than accumulates, so there is no zeroing pass ahead of it, and
                    // it sums the experts in router order — the order the axpy loop it replaces
                    // used, which the token-for-token assertion against llama.cpp depends on.
                    let _p = profile::scope("moe.combine");
                    // 🔴 `st.weights` is the router's own vector, in router order. It is only the
                    // right one when the device has every expert the router named; when some went
                    // to the CPU the device's batch is a non-prefix subset and `moe_combine` — which
                    // pairs `weights[m]` with the `m`-th matrix — would apply the wrong scalar to
                    // every expert after the first one the host took.
                    let w = if st.cpu_pick.is_empty() { &st.weights } else { &st.weights_sub };
                    ctx.moe_combine(&st.ffn, &st.expert_out, w, k, n_embd)?;
                }
            }

            // ---- collect the host's share ----------------------------------------------------
            //
            // Everything above was submitted, not executed: the queue is asynchronous, so by the
            // time this runs the CPU has had the whole of this block's staging and arithmetic to
            // work in. `host.wait` against `host.busy` in the residency report is the measurement
            // of how much of it was actually hidden.
            if let Some(job) = job {
                let h = host.expect("a job implies an executor");
                {
                    let _p = profile::scope("moe.host_sync");
                    h.sync(job, &mut st.cpu_out)?;
                }
                let _p = profile::scope("moe.host_add");
                // 🔴 Asynchronous, for the same reason `stage` is, and with the same argument.
                // A blocking copy here is a fence in the middle of every block: it waits for
                // this block's staging and matvecs, which are already submitted, and it gets
                // billed for them. Measured at 520 slots and `frac:0.5`, it was **10.96 ms a
                // token, 20% of the step** — device time that was going to be paid anyway,
                // charged to the wrong line and serialising the host behind it.
                //
                // The safety condition is that `cpu_out` stay alive and unmodified until the
                // copy runs. It lives in `State` for the session, and the only thing that
                // writes it is the *next* block's `sync` — which is downstream of that block's
                // router readback, a blocking download that drains this copy first. So the copy
                // has completed before its source is touched again, on the same in-order queue
                // the rest of this file rests on.
                let bytes = unsafe {
                    std::slice::from_raw_parts(st.cpu_out.as_ptr().cast::<u8>(), n_embd * 4)
                };
                unsafe { ctx.upload_async(&st.cpu_ffn, bytes)? };
                ctx.add(&st.ffn, &st.ffn, &st.cpu_ffn, n_embd)?;
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

/// 🔴 Wait for the queue before any of this model's host memory goes away.
///
/// `moe.host_add` submits an **asynchronous** copy whose source is `State::cpu_out`, a plain
/// `Vec` on the host. `session.rs` solves the same problem for the memory-mapped weights by
/// declaring the mapping before the `Context`, so the queue's destructor drains before the pages
/// are unmapped — but `State` lives *inside* this struct, which is declared **after** the
/// `Context` and therefore drops **before** it. Without this, a copy still in flight would be
/// reading a freed buffer, and nothing anywhere would report it.
///
/// In practice every `decode` ends with a blocking logit readback, so the queue is already empty
/// by the time this runs. That is exactly the incidental property `session.rs` refuses to rely
/// on for the same reason, and this does not rely on it either.
impl Drop for Model<'_, '_> {
    fn drop(&mut self) {
        let _ = self.ctx.sync();
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
    /// Worker threads the host executor runs on, or zero when there is none.
    pub host_threads: usize,
    /// What it actually did.
    pub host: HostStats,
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
        // 🔴 Asynchronous, and the safety argument is the whole reason this is fast.
        //
        // `v.data` is a borrow of the memory-mapped GGUF — `MappedModel` outlives every buffer
        // in this engine and the file is never written — so the source stays alive and
        // unmodified for as long as the copy needs it, which is what `upload_async` requires.
        // Ordering is the in-order queue's: this memcpy is submitted before the matvec that
        // reads the slot, so the matvec runs after it. That is the same guarantee the rest of
        // the decode path already relies on, and it is why `stage` before compute remains a
        // correctness property of this file rather than an accident of every call waiting.
        //
        // ⚠️ If this ever stages from anything but the mapping — a decompressed buffer, a
        // reordered scratch, a temporary — it must go back to the blocking `ctx.upload`. The
        // failure mode is a slot filled with whatever the memory became: finite, fluent, wrong.
        unsafe { ctx.upload_async(dst, v.data)? };
        moved += v.data.len() as u64;
    }
    Ok(moved)
}

/// Point `dst` at the pool buffers `slots` names.
///
/// The scratch is reused rather than collected fresh because this runs three times per block
/// per token. It cannot live in [`State`]: its elements borrow the expert pool, which is a
/// sibling field, so it is a local of `decode` instead. A slot may be larger than the block's expert — it is sized for the largest block
/// — which the batched matvec tolerates: it checks each buffer is big enough, not that it is
/// exact.
fn bank_batch<'a, 'c>(
    dst: &mut Vec<&'a DeviceBuffer<'c>>,
    bank: &'a [DeviceBuffer<'c>],
    slots: &[Slot],
) {
    dst.clear();
    dst.extend(slots.iter().map(|&sl| &bank[sl as usize]));
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
    fn a_residency_spec_round_trips_through_the_one_parser_every_tool_uses() {
        // Two tools spelling the same sweep row differently is how a table ends up comparing
        // two policies under one heading.
        assert_eq!("all".parse::<Residency>().unwrap(), Residency::All);
        assert_eq!("3172".parse::<Residency>().unwrap(), Residency::Slots(3172));
        assert_eq!(
            "static:24".parse::<Residency>().unwrap(),
            Residency::StaticSplit { resident_blocks: 24 }
        );
        let free = 12_166_012_928u64;
        assert_eq!(
            format!("plan:{free}").parse::<Residency>().unwrap(),
            Residency::Planned(memory::DeviceMemory { total_bytes: free, free_bytes: free })
        );
        // A typo has to be a refusal, not a silent `Slots(0)` or a fallback to `All`.
        assert!("half".parse::<Residency>().is_err());
        assert!("plan:lots".parse::<Residency>().is_err());
        assert!("static:".parse::<Residency>().is_err());
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
