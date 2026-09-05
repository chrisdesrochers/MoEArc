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
//! # What this is not
//!
//! Not fast, and not trying to be. Every expert gets its own device allocation, so no offset
//! arithmetic stands between `MappedModel::expert` and the matmul; the router's choice is read
//! back to the host once per block, which costs a device round trip per block per token; and
//! prompt tokens go through the single-token decode path one at a time rather than as a batched
//! prefill. All three are deliberate. `cache::ExpertCache` and `runtime::Runtime` exist to make
//! this fast, and they can only be trusted once there is a correct answer to compare against.

use moearc_kernels::{Context, DeviceBuffer, KernelError, KvType, QuantType, RopeKind};
use moearc_model::ModelError;
use moearc_model::gguf::Value;
use moearc_model::tensors::{ExpertBank, MappedModel, TensorView, names};

use crate::kv::{KvError, PagedKvCache, SeqId};

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

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(e) => write!(f, "model file: {e}"),
            Self::Kernel(e) => write!(f, "device: {e}"),
            Self::Kv(e) => write!(f, "kv cache: {e}"),
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
    /// One entry per expert, in expert order, sliced out of the stacked bank by
    /// `MappedModel::expert` — the helper that is cross-checked against independent Python. No
    /// offset arithmetic is done here.
    gate: Vec<QTensor<'c>>,
    up: Vec<QTensor<'c>>,
    down: Vec<QTensor<'c>>,
}

/// Every weight the graph reads, on the device.
pub struct Weights<'c> {
    pub cfg: Config,
    token_embd: QTensor<'c>,
    output_norm: DeviceBuffer<'c>,
    output: QTensor<'c>,
    blocks: Vec<Block<'c>>,
    /// Bytes copied to the card. Summed from what was actually uploaded, not estimated.
    pub bytes_uploaded: u64,
}

impl<'c> Weights<'c> {
    /// Upload every tensor the graph reads.
    ///
    /// The whole model is made resident. That is the simplest thing that can be correct, and it
    /// fits: OLMoE-1B-7B at Q4_K_M is 3.9 GiB against a B580's 11.3 GiB usable.
    pub fn upload(ctx: &'c Context, model: &MappedModel) -> Result<Self, EngineError> {
        let cfg = Config::from_model(model)?;
        let mut bytes = 0u64;

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
                names::FFN_GATE_EXPS,
                names::FFN_UP_EXPS,
                names::FFN_DOWN_EXPS,
            ] {
                bytes += model.block_tensor(b, s)?.data.len() as u64;
            }

            let simple = |suffix: &str| -> Result<DeviceBuffer<'c>, EngineError> {
                upload(ctx, &model.block_tensor(b, suffix)?)
            };
            let matrix = |suffix: &str| -> Result<QTensor<'c>, EngineError> {
                upload_matrix(ctx, &model.block_tensor(b, suffix)?)
            };
            let bank = |kind: ExpertBank| -> Result<Vec<QTensor<'c>>, EngineError> {
                (0..cfg.n_expert as u32)
                    .map(|e| upload_matrix(ctx, &model.expert(b, kind, e)?))
                    .collect()
            };

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
                gate: bank(ExpertBank::Gate)?,
                up: bank(ExpertBank::Up)?,
                down: bank(ExpertBank::Down)?,
            });
        }

        Ok(Self { cfg, token_embd, output_norm, output, blocks, bytes_uploaded: bytes })
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

    zeros: Vec<f32>,
    idx_host: Vec<u32>,
    w_host: Vec<f32>,
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
            zeros: vec![0.0; n_embd],
            idx_host: vec![0; cfg.n_expert_used],
            w_host: vec![0.0; cfg.n_expert_used],
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

/// A model on the device: weights plus the state of one sequence.
pub struct Model<'c> {
    ctx: &'c Context,
    pub weights: Weights<'c>,
    pub state: State<'c>,
}

impl<'c> Model<'c> {
    pub fn new(ctx: &'c Context, model: &MappedModel, n_ctx: usize) -> Result<Self, EngineError> {
        let weights = Weights::upload(ctx, model)?;
        let state = State::new(ctx, &weights.cfg, n_ctx)?;
        Ok(Self { ctx, weights, state })
    }

    pub fn cfg(&self) -> &Config {
        &self.weights.cfg
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
        let w = &self.weights;
        let cfg = &w.cfg;
        let st = &mut self.state;

        if token as usize >= cfg.n_vocab {
            return Err(EngineError::Unsupported(format!(
                "token id {token} is outside the {}-entry vocabulary",
                cfg.n_vocab
            )));
        }

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

        ctx.upload_slice(&st.tok, &[token])?;
        ctx.upload_slice(&st.pos, &[pos])?;

        let n_embd = cfg.n_embd;
        let n_embd_kv = cfg.n_head_kv * cfg.head_dim;
        let eps = cfg.rms_eps;
        let scale = cfg.kq_scale();

        // h = token_embd[token]
        ctx.embed_rows(w.token_embd.ty, &st.h, &w.token_embd.buf, &st.tok, 1, n_embd)?;

        for (bi, b) in w.blocks.iter().enumerate() {
            // ---- attention -------------------------------------------------------------
            ctx.rmsnorm(&st.x, &st.h, Some(&b.attn_norm), 1, n_embd, eps)?;
            matvec(ctx, &st.q, &b.attn_q, &st.x)?;
            matvec(ctx, &st.k, &b.attn_k, &st.x)?;
            matvec(ctx, &st.v, &b.attn_v, &st.x)?;

            // QK-norm over the whole vector — before the head reshape, before RoPE.
            ctx.rmsnorm(&st.q_normed, &st.q, Some(&b.attn_q_norm), 1, n_embd, eps)?;
            ctx.rmsnorm(&st.k_normed, &st.k, Some(&b.attn_k_norm), 1, n_embd_kv, eps)?;

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
            matvec(ctx, &st.proj, &b.attn_output, &st.attn)?;
            // `add` writes each index from the same index of both inputs, so aliasing the
            // accumulator into the left operand is well defined.
            ctx.add(&st.h, &st.h, &st.proj, n_embd)?;
            tap_record(&mut st.tap, format!("ffn_inp-{bi}"), ctx, &st.h, n_embd)?;

            // ---- MoE FFN ---------------------------------------------------------------
            ctx.rmsnorm(&st.x, &st.h, Some(&b.ffn_norm), 1, n_embd, eps)?;
            matvec(ctx, &st.router, &b.ffn_gate_inp, &st.x)?;
            tap_record(&mut st.tap, format!("ffn_moe_logits-{bi}"), ctx, &st.router, cfg.n_expert)?;
            // `normalize = false`: llama.cpp calls `build_moe_ffn` with `norm_w = false` for
            // OLMoE, so the weights are raw softmax probabilities and do not sum to one.
            ctx.topk_router(
                &st.idx,
                &st.weights,
                &st.router,
                1,
                cfg.n_expert,
                cfg.n_expert_used,
                false,
            )?;
            ctx.download_slice(&mut st.idx_host, &st.idx)?;
            ctx.download_slice(&mut st.w_host, &st.weights)?;
            if let Some(t) = st.tap.as_mut() {
                t.items.push((
                    format!("ffn_moe_topk-{bi}"),
                    st.idx_host.iter().map(|i| *i as f32).collect(),
                ));
                t.items.push((format!("ffn_moe_weights-{bi}"), st.w_host.clone()));
            }

            ctx.upload_slice(&st.ffn, &st.zeros)?;
            for j in 0..cfg.n_expert_used {
                let e = st.idx_host[j] as usize;
                let weight = st.w_host[j];
                let (Some(gate), Some(up), Some(down)) =
                    (b.gate.get(e), b.up.get(e), b.down.get(e))
                else {
                    return Err(EngineError::Unsupported(format!(
                        "the router named expert {e}, past the {} in block {bi}",
                        cfg.n_expert
                    )));
                };
                matvec(ctx, &st.gate, gate, &st.x)?;
                matvec(ctx, &st.up, up, &st.x)?;
                ctx.swiglu(&st.act, &st.gate, &st.up, cfg.n_ff)?;
                matvec(ctx, &st.expert_out, down, &st.act)?;
                ctx.axpy(&st.ffn, &st.expert_out, weight, n_embd)?;
            }
            tap_record(&mut st.tap, format!("ffn_moe_out-{bi}"), ctx, &st.ffn, n_embd)?;
            ctx.add(&st.h, &st.h, &st.ffn, n_embd)?;
            tap_record(&mut st.tap, format!("l_out-{bi}"), ctx, &st.h, n_embd)?;
        }

        ctx.rmsnorm(&st.x, &st.h, Some(&w.output_norm), 1, n_embd, eps)?;
        tap_record(&mut st.tap, "result_norm".to_string(), ctx, &st.x, n_embd)?;
        matvec(ctx, &st.logits_dev, &w.output, &st.x)?;
        ctx.download_slice(&mut st.logits, &st.logits_dev)?;
        Ok(&st.logits)
    }
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
