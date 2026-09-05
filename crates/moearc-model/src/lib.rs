//! Model metadata: the numbers the cache planner needs, read out of the model file.
//!
//! [`AutoCacheRequest`][auto] in `moearc-engine` needs four facts about a mixture-of-experts
//! model before it can size a GPU cache: how many experts fire per token, how many exist, how
//! big one resident expert is, and how big the weights are. Every one of them is already
//! written in the GGUF file. Asking a user to supply them — as most runtimes do — is asking
//! them to look up numbers the file already knows, and to be wrong occasionally.
//!
//! [`inspect`] answers all four from the file's header alone. It reads about 11 MB of a
//! 20.6 GiB model and never opens the tensor blob.
//!
//! [`pull`](pull::pull) is the other half of the job: getting the file in the first place. It
//! fetches a GGUF from the Hugging Face Hub — resumably, silently, and reporting progress
//! through a callback — and hands it to [`inspect`] before it says the download succeeded.
//!
//! ```no_run
//! # use std::path::Path;
//! let info = moearc_model::inspect(Path::new("model.gguf"))?;
//! println!("{} experts, {} active, {} B each", info.total_experts, info.active_experts, info.per_expert_bytes);
//! # Ok::<(), moearc_model::ModelError>(())
//! ```
//!
//! [auto]: https://github.com/chrisdesrochers/MoEArc

pub mod gguf;
pub mod pull;
pub mod quant;
pub mod tensors;

use std::collections::BTreeMap;
use std::path::Path;

use gguf::GgufHeader;

/// Why a model file could not be inspected.
///
/// Every variant is a fact about the file, not about this crate: a caller can act on all of
/// them. Nothing in the read path panics on malformed input — a GGUF header is untrusted data
/// and a corrupt one is an expected outcome, not a bug.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    #[error("could not read the model file: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a GGUF file: expected magic \"GGUF\", found {found:02x?}")]
    BadMagic { found: [u8; 4] },

    #[error("unsupported GGUF version {0}: this reader handles v2 and v3")]
    UnsupportedVersion(u32),

    #[error(
        "file ends inside the header: needed {needed} B at offset {offset}, file is {file_size} B"
    )]
    Truncated { offset: u64, needed: u64, file_size: u64 },

    #[error("header declares {count} {what}, which cannot fit in a {file_size} B file")]
    ImplausibleCount { what: &'static str, count: u64, file_size: u64 },

    #[error("key `{key}` has unknown GGUF value type {type_id}")]
    BadValueType { key: String, type_id: u32 },

    #[error("key `{key}` is an array of arrays, which GGUF does not define")]
    NestedArray { key: String },

    #[error("tensor `{tensor}` declares {n_dims} dimensions; ggml allows at most 4")]
    TooManyDims { tensor: String, n_dims: u32 },

    #[error("tensor `{tensor}` has unknown ggml type id {type_id}")]
    UnknownTensorType { tensor: String, type_id: u32 },

    #[error(
        "tensor `{tensor}` has {elements} elements, not a whole number of {block_size}-element blocks"
    )]
    ElementsNotBlockAligned { tensor: String, elements: u64, block_size: u64 },

    #[error("general.alignment is {alignment}; it must be a power of two")]
    BadAlignment { alignment: u64 },

    #[error(
        "tensor `{tensor}` ends at byte {end} but the file is only {file_size} B (truncated download?)"
    )]
    TensorDataOverrunsFile { tensor: String, end: u64, file_size: u64 },

    #[error("required metadata key `{0}` is missing")]
    MissingKey(String),

    #[error("metadata key `{key}` is not {want}")]
    WrongKeyType { key: String, want: &'static str },

    #[error("metadata key `{key}` is an array whose elements differ; this reader needs one value")]
    NonUniformKey { key: String },

    #[error("`{architecture}` is not a mixture-of-experts model: no expert count in its metadata")]
    NotMixtureOfExperts { architecture: String },

    #[error("model declares {experts} experts but has no `ffn_*_exps` tensors to size them from")]
    NoExpertTensors { experts: u32 },

    #[error(
        "block {block}'s expert tensors are {bytes} B, which {experts} experts do not divide evenly"
    )]
    ExpertBytesNotDivisible { block: u32, bytes: u64, experts: u32 },

    #[error(
        "expert ({expert} B) and dense ({dense} B) weights sum to {} B, not the {total} B in the \
         tensor index; a tensor is being counted twice or not at all",
        expert + dense
    )]
    WeightsPartitionMismatch { dense: u64, expert: u64, total: u64 },

    #[error("the model has no tensor named `{name}`")]
    TensorNotFound { name: String },

    #[error(
        "tensor `{tensor}` has rank {rank}; slicing one index out of it needs at least 2 dimensions"
    )]
    NotSliceable { tensor: String, rank: usize },

    #[error(
        "tensor `{tensor}` has {count} slices along its last dimension; {index} is out of range"
    )]
    SliceOutOfRange { tensor: String, index: u64, count: u64 },

    #[error(
        "tensor `{tensor}` slices into {count} x {slice_bytes} B, which is not the {total_bytes} B \
         it occupies; the slice arithmetic and the tensor size disagree"
    )]
    SliceStrideMismatch { tensor: String, slice_bytes: u64, count: u64, total_bytes: u64 },

    #[error(
        "expert bank `{tensor}` stacks {last_dim} matrices but the model declares {expert_count} \
         experts; this tensor is not the expert bank it was addressed as"
    )]
    ExpertBankShape { tensor: String, last_dim: u64, expert_count: u32 },
}

/// What a model file says about itself.
///
/// All byte counts are exact sums over the tensor index, not estimates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// `general.architecture`, e.g. `qwen35moe`. Every other key below is namespaced under it.
    pub architecture: String,
    /// `general.name`, if the file carries one. Many quantised repacks do not.
    pub name: Option<String>,
    /// Transformer blocks in the model.
    pub block_count: u32,
    /// Blocks that carry an expert bank, counted from the tensor index.
    ///
    /// A residency slot is a *(block, expert)* pair, so the model's full slot count is
    /// `moe_block_count * total_experts` — **not** `block_count * total_experts`. The two
    /// coincide only when every block is MoE, which is common but not guaranteed: several
    /// architectures keep the first block or two dense, and a hybrid can interleave.
    pub moe_block_count: u32,
    /// Model hidden size.
    pub embedding_length: u32,
    /// The longest context the model was trained for.
    pub context_length: u32,
    /// Experts the model has — `AutoCacheRequest::total_experts`.
    pub total_experts: u32,
    /// Experts routed to per token — `AutoCacheRequest::num_experts`.
    pub active_experts: u32,
    /// Bytes one resident expert slot occupies — `AutoCacheRequest::per_expert_bytes`.
    ///
    /// See [`expert_geometry`] for how this is derived and why it is a maximum.
    pub per_expert_bytes: u64,
    /// Whether every MoE block agreed on [`Self::per_expert_bytes`].
    ///
    /// `false` means the file mixes quantisation types across blocks and the figure above is a
    /// conservative maximum rather than an exact size. Worth surfacing: a caller that wants
    /// tight packing needs to know it is leaving slots on the table.
    pub per_expert_bytes_uniform: bool,
    /// Total bytes of tensor data — `AutoCacheRequest::weights_bytes`.
    ///
    /// Exactly [`Self::dense_weights_bytes`] + [`Self::expert_weights_bytes`]; the partition is
    /// checked, not assumed. See [`ModelError::WeightsPartitionMismatch`].
    pub weights_bytes: u64,
    /// Bytes of every `ffn_*_exps` tensor — the weights the residency cache pages.
    ///
    /// Summed from the tensor index, not estimated. 🔴 It is deliberately **not**
    /// `per_expert_bytes * total_experts * moe_block_count`: that product uses the per-block
    /// *maximum* and so overstates any file that mixes quantisation types across blocks — which
    /// real files do. On the Qwen3.5 build this was developed against the estimate is 5.4% high.
    pub expert_weights_bytes: u64,
    /// Bytes of everything that is not an expert bank: embeddings, attention, norms, shared
    /// experts, the output head. These are always resident, so this is the figure a memory
    /// planner subtracts before it has anything left to page.
    pub dense_weights_bytes: u64,
    /// Bytes of KV cache one token needs across the whole model, at f16.
    ///
    /// Multiply by a context length to get the KV footprint; divide by a page size to get
    /// `cache_per_page`. See [`kv_bytes_per_token`] for the assumptions.
    pub kv_bytes_per_token: u64,
    /// Blocks that actually hold a per-token KV cache. Equals [`Self::block_count`] for a plain
    /// transformer; smaller for a hybrid, where most blocks are recurrent. Reported because
    /// [`Self::kv_bytes_per_token`] is not interpretable without it.
    pub kv_layers: u32,
    /// Size of the file on disk.
    pub file_size: u64,
}

/// Read a GGUF model file and report what the cache planner needs.
///
/// Reads the header only; the tensor data blob is never touched.
pub fn inspect(path: &Path) -> Result<ModelInfo, ModelError> {
    ModelInfo::from_header(&gguf::read(path)?)
}

impl ModelInfo {
    /// Derive the model facts from an already-parsed header.
    ///
    /// Split out from [`inspect`] so the derivation can be tested against a header built in
    /// memory, with no file involved.
    pub fn from_header(h: &GgufHeader) -> Result<Self, ModelError> {
        let architecture = h.str_key("general.architecture")?.to_string();
        let key = |suffix: &str| format!("{architecture}.{suffix}");

        // `expert_count` is what distinguishes a MoE file. A dense model omits it entirely, so
        // its absence is a clear "wrong kind of model" rather than a parse failure.
        let total_experts = match h.get(&key("expert_count")).and_then(gguf::Value::as_u64) {
            Some(n) if n > 0 => u32::try_from(n).map_err(|_| ModelError::WrongKeyType {
                key: key("expert_count"),
                want: "a value that fits in u32",
            })?,
            _ => return Err(ModelError::NotMixtureOfExperts { architecture }),
        };
        let active_experts = h.u32_key(&key("expert_used_count"))?;

        let block_count = h.u32_key(&key("block_count"))?;
        let embedding_length = h.u32_key(&key("embedding_length"))?;
        let context_length = h.u32_key(&key("context_length"))?;

        let experts = expert_geometry(h, total_experts)?;
        let (kv_bytes_per_token, kv_layers) =
            kv_bytes_per_token(h, &architecture, embedding_length)?;

        // One pass, two buckets, decided by the same predicate `expert_geometry` uses — so the
        // split cannot drift from the per-expert arithmetic above.
        let mut dense_weights_bytes = 0u64;
        let mut expert_weights_bytes = 0u64;
        for t in &h.tensors {
            let n = t.nbytes()?;
            if is_expert_tensor(&t.name) {
                expert_weights_bytes += n;
            } else {
                dense_weights_bytes += n;
            }
        }
        let weights_bytes = dense_weights_bytes + expert_weights_bytes;
        // A genuine partition check, not a tautology: `expert_geometry` walks the same tensors
        // through its own name filter and block grouping, so if that classification and this one
        // ever disagree — a new `ffn_*_exps` spelling, a tensor outside any `blk.N.` prefix — the
        // totals part company here. Loud beats quietly-off: an unnoticed few hundred megabytes
        // on the wrong side of this line is a plan that OOMs on a device, far from the cause.
        if experts.expert_bytes != expert_weights_bytes {
            return Err(ModelError::WeightsPartitionMismatch {
                dense: dense_weights_bytes,
                expert: expert_weights_bytes,
                total: weights_bytes,
            });
        }

        Ok(Self {
            name: h.get("general.name").and_then(|v| v.as_str()).map(str::to_string),
            architecture,
            block_count,
            moe_block_count: experts.moe_block_count,
            embedding_length,
            context_length,
            total_experts,
            active_experts,
            per_expert_bytes: experts.per_expert_bytes,
            per_expert_bytes_uniform: experts.uniform,
            weights_bytes,
            expert_weights_bytes,
            dense_weights_bytes,
            kv_bytes_per_token,
            kv_layers,
            file_size: h.file_size,
        })
    }
}

/// The three tensors that make up a block's expert bank, in llama.cpp's naming.
///
/// A slot in the MoE cache holds one expert's share of all three. Architectures vary: most
/// carry gate/up/down, a few omit the gate (no GLU), so this sums whichever are present rather
/// than requiring all three.
pub(crate) const EXPERT_TENSORS: [&str; 3] =
    ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];

/// Whether a tensor belongs to a block's expert bank.
///
/// The single predicate behind both the per-expert arithmetic and the dense/expert weight split,
/// so the two can never classify a tensor differently. Note what it excludes: `ffn_*_shexp`
/// (shared experts, which every token uses and which are therefore always resident) and
/// `ffn_gate_inp` (the router). Those are dense weights, not pageable ones.
fn is_expert_tensor(name: &str) -> bool {
    match split_block_tensor(name) {
        Some((_, suffix)) => EXPERT_TENSORS.contains(&suffix),
        None => false,
    }
}

/// What the tensor index says about the model's expert banks.
struct ExpertGeometry {
    /// Bytes one resident expert slot occupies; see [`expert_geometry`].
    per_expert_bytes: u64,
    /// Whether every MoE block agreed on that figure.
    uniform: bool,
    /// Blocks carrying an expert bank.
    moe_block_count: u32,
    /// Total bytes across every expert bank.
    expert_bytes: u64,
}

/// Measure the model's expert banks: slot size, block count and total bytes.
///
/// **Derived from the tensor index, never from a rule of thumb.** Each `ffn_*_exps` tensor is
/// the whole bank for its block — all `total_experts` experts stacked along the last dimension,
/// stored in one quantisation type. So one expert's share of one block is the summed byte size
/// of that block's expert tensors divided by the expert count, and the division must be exact:
/// the stack is contiguous and equally sized, so a remainder means the shape or the expert
/// count is not what we think it is, and quietly rounding would put the whole cache plan out.
///
/// 🔴 **Judgement call: the result is the maximum across blocks, not the mean.** Real files mix
/// types — the Qwen3.5 MoE build this was developed against quantises `ffn_down_exps` at Q5_K in
/// 37 blocks and Q6_K in the other 3, a 7.3% spread in slot size. A cache slot must be able to
/// hold *any* expert, so it has to be sized for the largest. Erring the other way would fit
/// more slots and then overrun VRAM on the first block that routed to a heavy expert, which is
/// exactly the class of late, unattributable failure the planner exists to avoid. The cost is
/// visible and bounded — [`ModelInfo::per_expert_bytes_uniform`] reports when it applies. It is
/// also why [`ModelInfo::expert_weights_bytes`] is summed here rather than reconstructed as
/// `per_expert_bytes * total_experts * moe_block_count`: that product applies the maximum to
/// every block and overstates a mixed-quant file.
fn expert_geometry(h: &GgufHeader, total_experts: u32) -> Result<ExpertGeometry, ModelError> {
    let mut per_block: BTreeMap<u32, u64> = BTreeMap::new();
    for t in &h.tensors {
        let Some((block, suffix)) = split_block_tensor(&t.name) else { continue };
        if EXPERT_TENSORS.contains(&suffix) {
            *per_block.entry(block).or_default() += t.nbytes()?;
        }
    }
    if per_block.is_empty() {
        return Err(ModelError::NoExpertTensors { experts: total_experts });
    }
    let moe_block_count = per_block.len() as u32;
    let expert_bytes = per_block.values().sum();

    let mut sizes = Vec::with_capacity(per_block.len());
    for (block, bytes) in per_block {
        if bytes % u64::from(total_experts) != 0 {
            return Err(ModelError::ExpertBytesNotDivisible {
                block,
                bytes,
                experts: total_experts,
            });
        }
        sizes.push(bytes / u64::from(total_experts));
    }
    let max = sizes.iter().copied().max().unwrap_or(0);
    let uniform = sizes.iter().all(|&s| s == max);
    Ok(ExpertGeometry { per_expert_bytes: max, uniform, moe_block_count, expert_bytes })
}

/// Tensor-name suffixes that prove a block keeps a per-token KV cache.
///
/// `attn_k`/`attn_v` are the ordinary projections. `attn_k_b`/`attn_v_b` are DeepSeek-style
/// MLA, where the cached thing is the latent, but the block is still a KV-caching block.
pub(crate) const KV_TENSORS: [&str; 4] =
    ["attn_k.weight", "attn_v.weight", "attn_k_b.weight", "attn_v_b.weight"];

/// Bytes of KV cache one token needs, and how many blocks contribute.
///
/// `2 * kv_layers * head_count_kv * head_dim` elements per token — K and V, per attention block
/// — at 2 bytes each.
///
/// Two things here are assumptions, and both are stated rather than hidden:
///
/// - **f16.** The cache element type is a runtime choice, not a model property; f16 is
///   llama.cpp's default for both K and V and the only one we can infer from the file. A caller
///   running a quantised KV cache must scale this.
/// - **`kv_layers` is counted from the tensor index, not taken as `block_count`.** 🔴 This
///   matters and it is not a micro-optimisation. The Qwen3.5 MoE model this was developed
///   against is a hybrid: only every 4th block is full attention (10 of 40 — the rest are
///   recurrent and carry `ssm_*` tensors and a fused `attn_qkv` that is *not* an attention KV
///   projection). Assuming all 40 blocks cache would overstate the KV footprint **4×**, and the
///   planner would hand back a cache a quarter the size it could have been, for every context
///   length, with nothing in the output to show why. Counting blocks that actually carry
///   `attn_k`/`attn_v` gets a plain transformer right too, where the count is every block.
///
/// Recurrent blocks do hold state, but it is per *sequence* and constant in context length, so
/// it does not belong in a per-token figure. It is not accounted for anywhere in this crate.
///
/// 🔴 **The hybrid layout is not only a sizing concern — it is a kernel-coverage one.** An
/// attempt to run FreeToken against this model hung in `causal_conv1d_varlen`, a recurrent/SSM
/// kernel. That looked mysterious while the model was assumed to be a plain MoE transformer; it
/// is entirely expected once 30 of its 40 blocks are known to be recurrent. Any engine targeting
/// this family needs a working SSM path, not just attention and MoE — and this function's
/// `kv_layers < block_count` is the cheapest early signal that such a path is required.
fn kv_bytes_per_token(
    h: &GgufHeader,
    arch: &str,
    embedding_length: u32,
) -> Result<(u64, u32), ModelError> {
    let mut kv_blocks: BTreeMap<u32, ()> = BTreeMap::new();
    for t in &h.tensors {
        if let Some((block, suffix)) = split_block_tensor(&t.name) {
            if KV_TENSORS.contains(&suffix) {
                kv_blocks.insert(block, ());
            }
        }
    }
    // A file with no recognisable KV projections is either an architecture we have not seen or
    // one with no attention at all. Fall back to every block: that is the conservative
    // direction — it overstates the cache cost, so the planner reserves too much rather than
    // handing out memory that does not exist.
    let kv_layers = if kv_blocks.is_empty() {
        h.u32_key(&format!("{arch}.block_count"))?
    } else {
        kv_blocks.len() as u32
    };

    let head_count_kv = uniform_u64(h, &format!("{arch}.attention.head_count_kv"))?;
    let head_count = uniform_u64(h, &format!("{arch}.attention.head_count"))?;
    // `key_length`/`value_length` are optional; llama.cpp defaults both to n_embd/n_head, so we
    // do the same rather than failing on a file that simply did not restate the default.
    let default_head_dim = u64::from(embedding_length).checked_div(head_count).unwrap_or(0);
    let key_length = optional_uniform_u64(h, &format!("{arch}.attention.key_length"))?
        .unwrap_or(default_head_dim);
    let value_length = optional_uniform_u64(h, &format!("{arch}.attention.value_length"))?
        .unwrap_or(default_head_dim);

    /// Bytes per f16 KV element. See the note above on why this is an assumption.
    const KV_ELEMENT_BYTES: u64 = 2;
    let per_token =
        u64::from(kv_layers) * head_count_kv * (key_length + value_length) * KV_ELEMENT_BYTES;
    Ok((per_token, kv_layers))
}

/// Split `blk.<n>.<rest>` into its block index and the rest of the name.
///
/// Returns [`None`] for anything else, which is how non-block tensors (`token_embd.weight`,
/// `output.weight`) are filtered out without a second list to keep in sync.
pub(crate) fn split_block_tensor(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    Some((rest[..dot].parse().ok()?, &rest[dot + 1..]))
}

/// A required key that is either a scalar or an array whose elements all agree.
///
/// Per-layer arrays appear for `attention.head_count_kv` in some architectures. Where every
/// layer is the same the array is just a verbose scalar and is accepted; where layers genuinely
/// differ, a single number would be a fiction, so it is refused instead of averaged.
fn uniform_u64(h: &GgufHeader, key: &str) -> Result<u64, ModelError> {
    optional_uniform_u64(h, key)?.ok_or_else(|| ModelError::MissingKey(key.to_string()))
}

/// [`uniform_u64`] for a key that may legitimately be absent.
fn optional_uniform_u64(h: &GgufHeader, key: &str) -> Result<Option<u64>, ModelError> {
    let Some(v) = h.get(key) else { return Ok(None) };
    if let Some(n) = v.as_u64() {
        return Ok(Some(n));
    }
    let elems = v.as_array().ok_or_else(|| ModelError::WrongKeyType {
        key: key.to_string(),
        want: "an unsigned integer",
    })?;
    let mut it = elems.iter().map(|e| {
        e.as_u64().ok_or_else(|| ModelError::WrongKeyType {
            key: key.to_string(),
            want: "an array of unsigned integers",
        })
    });
    let first = it.next().ok_or_else(|| ModelError::NonUniformKey { key: key.to_string() })??;
    for e in it {
        if e? != first {
            return Err(ModelError::NonUniformKey { key: key.to_string() });
        }
    }
    Ok(Some(first))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ---- a GGUF writer, so the tests do not depend on a 20 GiB file -------------------------
    //
    // Small enough to be obviously correct by inspection, which matters: it is the only thing
    // asserting the reader's field order, and a bug shared between the two would cancel out.
    // The field order below is transcribed from the format comment in llama.cpp's `gguf.h`,
    // independently of the reader.

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn put_kv_u32(b: &mut Vec<u8>, key: &str, v: u32) {
        put_str(b, key);
        b.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
        b.extend_from_slice(&v.to_le_bytes());
    }

    fn put_kv_str(b: &mut Vec<u8>, key: &str, v: &str) {
        put_str(b, key);
        b.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
        put_str(b, v);
    }

    /// A tensor to write into the synthetic file: name, dims, ggml type id.
    struct T(&'static str, &'static [u64], u32);

    const Q4_K: u32 = 12;
    const Q6_K: u32 = 14;
    const F32: u32 = 0;

    fn nbytes(t: &T) -> u64 {
        let q = quant::lookup(t.2).unwrap();
        t.1.iter().product::<u64>() / q.block_size * q.type_size
    }

    /// Build a complete, valid GGUF v3 file in memory.
    ///
    /// Three blocks over 4 experts, of which only two are MoE. Every distinction the crate
    /// draws is present on purpose, so none of them can pass by accident:
    ///
    /// - Block 0's `ffn_down_exps` is Q6_K where block 1's is Q4_K — the mixed-quant shape found
    ///   in the real Qwen3.5 build, which is what exercises the maximum-not-mean rule.
    /// - Block 2 has no expert bank at all, so `moe_block_count` (2) must differ from
    ///   `block_count` (3).
    /// - Only block 1 carries `attn_k`/`attn_v`, mimicking a hybrid, so `kv_layers` is 1 of 3.
    /// - A router (`ffn_gate_inp`) and a shared expert (`ffn_down_shexp`) are present and must
    ///   land on the *dense* side of the weight split, not the pageable one.
    fn synthetic_gguf() -> (Vec<u8>, Vec<T>) {
        let tensors = vec![
            T("blk.0.ffn_gate_exps.weight", &[256, 8, 4], Q4_K),
            T("blk.0.ffn_up_exps.weight", &[256, 8, 4], Q4_K),
            T("blk.0.ffn_down_exps.weight", &[8, 256, 4], Q6_K),
            T("blk.1.ffn_gate_exps.weight", &[256, 8, 4], Q4_K),
            T("blk.1.ffn_up_exps.weight", &[256, 8, 4], Q4_K),
            T("blk.1.ffn_down_exps.weight", &[8, 256, 4], Q4_K),
            T("blk.1.attn_k.weight", &[8, 4], F32),
            T("blk.1.attn_v.weight", &[8, 4], F32),
            // Router and shared expert: MoE-adjacent names that are dense weights all the same.
            T("blk.0.ffn_gate_inp.weight", &[8, 4], F32),
            T("blk.1.ffn_down_shexp.weight", &[256, 8], Q4_K),
            // A wholly dense block.
            T("blk.2.ffn_gate.weight", &[256, 8], Q4_K),
            T("blk.2.ffn_down.weight", &[8, 256], Q4_K),
            T("output_norm.weight", &[8], F32),
        ];

        let mut kv = Vec::new();
        let mut n_kv = 0u64;
        put_kv_str(&mut kv, "general.architecture", "testmoe");
        put_kv_str(&mut kv, "general.name", "Synthetic MoE");
        n_kv += 2;
        for (k, v) in [
            ("testmoe.block_count", 3u32),
            ("testmoe.embedding_length", 8),
            ("testmoe.context_length", 128),
            ("testmoe.expert_count", 4),
            ("testmoe.expert_used_count", 2),
            ("testmoe.attention.head_count", 2),
            ("testmoe.attention.head_count_kv", 1),
            ("testmoe.attention.key_length", 4),
            ("testmoe.attention.value_length", 4),
        ] {
            put_kv_u32(&mut kv, k, v);
            n_kv += 1;
        }

        let mut index = Vec::new();
        let mut offset = 0u64;
        for t in &tensors {
            put_str(&mut index, t.0);
            index.extend_from_slice(&(t.1.len() as u32).to_le_bytes());
            for d in t.1 {
                index.extend_from_slice(&d.to_le_bytes());
            }
            index.extend_from_slice(&t.2.to_le_bytes());
            index.extend_from_slice(&offset.to_le_bytes());
            offset += nbytes(t).div_ceil(32) * 32;
        }
        let data_len = offset;

        let mut f = Vec::new();
        f.extend_from_slice(b"GGUF");
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        f.extend_from_slice(&n_kv.to_le_bytes());
        f.extend_from_slice(&kv);
        f.extend_from_slice(&index);
        while f.len() % 32 != 0 {
            f.push(0);
        }
        f.resize(f.len() + data_len as usize, 0);
        (f, tensors)
    }

    /// Write bytes to a uniquely named file under the temp dir and hand back the path.
    ///
    /// Deliberately not a `tempfile` dependency: this crate has exactly one dependency and a
    /// dozen lines of test scaffolding is not worth a second.
    struct TempGguf(std::path::PathBuf);

    impl TempGguf {
        fn new(tag: &str, bytes: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("moearc-model-{}-{tag}.gguf", std::process::id()));
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(bytes).unwrap();
            f.sync_all().unwrap();
            Self(path)
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    // ---- the happy path ---------------------------------------------------------------------

    #[test]
    fn reads_every_planner_input_from_a_synthetic_file() {
        let (bytes, tensors) = synthetic_gguf();
        let f = TempGguf::new("ok", &bytes);
        let info = inspect(&f.0).unwrap();

        assert_eq!(info.architecture, "testmoe");
        assert_eq!(info.name.as_deref(), Some("Synthetic MoE"));
        assert_eq!(info.block_count, 3);
        assert_eq!(info.embedding_length, 8);
        assert_eq!(info.context_length, 128);
        assert_eq!(info.total_experts, 4);
        assert_eq!(info.active_experts, 2);
        assert_eq!(info.file_size, bytes.len() as u64);

        // Recomputed here from the type table rather than copied from the reader: 8192 elements
        // is 32 blocks, so Q4_K is 32*144 = 4608 B and Q6_K is 32*210 = 6720 B.
        let gate = 8192 / 256 * 144;
        let down_q6 = 8192 / 256 * 210;
        let block0 = (gate + gate + down_q6) / 4;
        let block1 = (gate + gate + gate) / 4;
        assert_eq!(block0, 3984);
        assert_eq!(block1, 3456);
        // The maximum, not the mean (3720) and not block 1's.
        assert_eq!(info.per_expert_bytes, block0);
        assert!(!info.per_expert_bytes_uniform);

        // Only block 1 has attn_k/attn_v, so 1 of 3 blocks caches:
        // 1 layer * 1 kv head * (4 + 4) * 2 B = 16 B/token.
        assert_eq!(info.kv_layers, 1);
        assert_eq!(info.kv_bytes_per_token, 16);

        // Blocks 0 and 1 have expert banks; block 2 does not. A residency slot is a
        // (block, expert) pair, so this model has 2*4 = 8 slots, not 3*4 = 12.
        assert_eq!(info.moe_block_count, 2);
        assert!(info.moe_block_count < info.block_count);

        let expected: u64 = tensors.iter().map(nbytes).sum();
        assert_eq!(info.weights_bytes, expected);
        // Weights cannot exceed the file that holds them, and padding keeps them just under.
        assert!(info.weights_bytes < info.file_size);

        // The split, recomputed here from the tensor list rather than trusted from the reader.
        let expert: u64 =
            tensors.iter().filter(|t| t.0.ends_with("_exps.weight")).map(nbytes).sum();
        assert_eq!(expert, 15936 + 13824);
        assert_eq!(info.expert_weights_bytes, expert);
        assert_eq!(info.dense_weights_bytes, expected - expert);
        assert_eq!(info.dense_weights_bytes + info.expert_weights_bytes, info.weights_bytes);

        // The router and the shared expert are dense, so the dense side is more than just the
        // attention and norm tensors. Pin that: 128 + 128 (attn) + 128 (router) + 1152 (shexp)
        // + 1152 + 1152 (dense block) + 32 (output norm).
        assert_eq!(info.dense_weights_bytes, 3872);

        // And the estimate the planner would otherwise have used is measurably wrong: applying
        // the per-block maximum uniformly overstates the expert weights on a mixed-quant file.
        let naive =
            info.per_expert_bytes * u64::from(info.total_experts) * u64::from(info.moe_block_count);
        assert!(naive > info.expert_weights_bytes);
        assert_eq!(naive, 31872);
    }

    #[test]
    fn a_uniform_model_reports_uniform() {
        // Rewrite block 0's down-projection to Q4_K so both blocks match.
        let (mut bytes, _) = synthetic_gguf();
        let needle: Vec<u8> = {
            let mut v = Vec::new();
            put_str(&mut v, "blk.0.ffn_down_exps.weight");
            v
        };
        let at = bytes.windows(needle.len()).position(|w| w == needle.as_slice()).unwrap();
        // name, then rank (4 B) + 3 dims (24 B), then the type id.
        let type_at = at + needle.len() + 4 + 24;
        assert_eq!(u32::from_le_bytes(bytes[type_at..type_at + 4].try_into().unwrap()), Q6_K);
        bytes[type_at..type_at + 4].copy_from_slice(&Q4_K.to_le_bytes());

        let f = TempGguf::new("uniform", &bytes);
        let info = inspect(&f.0).unwrap();
        assert!(info.per_expert_bytes_uniform);
        assert_eq!(info.per_expert_bytes, 3456);
        // With no quant spread the naive product is exact — which is precisely why it is
        // untrustworthy in general: it is right until the file mixes types, and then silently
        // is not.
        assert_eq!(
            info.per_expert_bytes * u64::from(info.total_experts) * u64::from(info.moe_block_count),
            info.expert_weights_bytes
        );
    }

    // ---- negative controls ------------------------------------------------------------------

    #[test]
    fn a_non_gguf_file_is_rejected_by_magic() {
        let f = TempGguf::new("magic", b"not a gguf file at all, just some bytes");
        assert!(matches!(inspect(&f.0), Err(ModelError::BadMagic { .. })));
    }

    #[test]
    fn an_empty_file_errors_rather_than_panicking() {
        let f = TempGguf::new("empty", b"");
        assert!(matches!(inspect(&f.0), Err(ModelError::Truncated { .. })));
    }

    #[test]
    fn gguf_v1_and_future_versions_are_refused() {
        for v in [1u32, 4, 0xdead_beef] {
            let (mut bytes, _) = synthetic_gguf();
            bytes[4..8].copy_from_slice(&v.to_le_bytes());
            let f = TempGguf::new("version", &bytes);
            match inspect(&f.0) {
                Err(ModelError::UnsupportedVersion(got)) => assert_eq!(got, v),
                other => panic!("version {v} should be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_header_cut_in_half_errors_rather_than_panicking() {
        let (bytes, _) = synthetic_gguf();
        // Cut inside the KV block, past the counts, so the parse is well underway when it runs
        // out. This is the shape of an interrupted download.
        //
        // The cut point is derived, not a magic number: it has to clear both up-front
        // plausibility floors (24 B per tensor entry, 13 B per KV pair) or the header is
        // rejected by *those* guards before a length is ever read, and this test would silently
        // stop testing what it names. That is not hypothetical — adding tensors to the synthetic
        // file pushed a hardcoded cut below the tensor floor and flipped the error.
        let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let cut = (n_tensors * 24).max(n_kv * 13) as usize + 64;
        assert!(cut < bytes.len(), "synthetic header is too small to cut meaningfully");
        let f = TempGguf::new("cut", &bytes[..cut]);
        assert!(matches!(inspect(&f.0), Err(ModelError::Truncated { .. })));
    }

    #[test]
    fn a_file_missing_its_tensor_data_is_caught_by_the_span_check() {
        // Keep the entire header and drop the data blob. Every key still parses; only the
        // tensor-index-versus-file-length check can catch this.
        let (bytes, tensors) = synthetic_gguf();
        let data_len: usize = tensors.iter().map(|t| nbytes(t).div_ceil(32) as usize * 32).sum();
        let f = TempGguf::new("nodata", &bytes[..bytes.len() - data_len]);
        match inspect(&f.0) {
            Err(ModelError::TensorDataOverrunsFile { .. }) => {}
            other => panic!("expected an overrun error, got {other:?}"),
        }
    }

    #[test]
    fn an_absurd_tensor_count_is_rejected_before_it_is_looped_on() {
        let (mut bytes, _) = synthetic_gguf();
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let f = TempGguf::new("count", &bytes);
        match inspect(&f.0) {
            Err(ModelError::ImplausibleCount { what, .. }) => assert_eq!(what, "tensors"),
            other => panic!("expected a count rejection, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_ggml_type_is_an_error_not_a_guess() {
        let (mut bytes, _) = synthetic_gguf();
        let needle: Vec<u8> = {
            let mut v = Vec::new();
            put_str(&mut v, "blk.0.ffn_gate_exps.weight");
            v
        };
        let at = bytes.windows(needle.len()).position(|w| w == needle.as_slice()).unwrap();
        let type_at = at + needle.len() + 4 + 24;
        // 4 is GGML_TYPE_Q4_2, retired years ago and a zero row in ggml's own table.
        bytes[type_at..type_at + 4].copy_from_slice(&4u32.to_le_bytes());
        let f = TempGguf::new("badtype", &bytes);
        match inspect(&f.0) {
            Err(ModelError::UnknownTensorType { type_id, .. }) => assert_eq!(type_id, 4),
            other => panic!("expected an unknown-type error, got {other:?}"),
        }
    }

    /// The synthetic file with `expert_count` renamed away, so it reads as a dense model.
    ///
    /// The key is replaced with one of exactly the same length, so every later offset in the
    /// header is untouched and the file stays structurally valid — the *only* difference is
    /// that the MoE metadata is missing. Shared with `pull`'s tests, where downloading a dense
    /// model has to come out as a success, not as a corrupt transfer.
    pub(crate) fn synthetic_gguf_without_experts() -> Vec<u8> {
        let (mut bytes, _) = synthetic_gguf();
        let needle: Vec<u8> = {
            let mut v = Vec::new();
            put_str(&mut v, "testmoe.expert_count");
            v
        };
        let at = bytes.windows(needle.len()).position(|w| w == needle.as_slice()).unwrap();
        bytes[at + 8..at + 8 + needle.len() - 8].copy_from_slice(b"testmoe.expert_counx");
        bytes
    }

    #[test]
    fn a_dense_model_is_named_as_such_rather_than_mis_sized() {
        let bytes = synthetic_gguf_without_experts();
        let f = TempGguf::new("dense", &bytes);
        match inspect(&f.0) {
            Err(ModelError::NotMixtureOfExperts { architecture }) => {
                assert_eq!(architecture, "testmoe")
            }
            other => panic!("expected a not-MoE error, got {other:?}"),
        }
    }

    // ---- helpers ---------------------------------------------------------------------------

    #[test]
    fn block_tensor_names_split_only_where_they_should() {
        assert_eq!(
            split_block_tensor("blk.12.ffn_up_exps.weight"),
            Some((12, "ffn_up_exps.weight"))
        );
        assert_eq!(split_block_tensor("blk.0.attn_k.weight"), Some((0, "attn_k.weight")));
        assert_eq!(split_block_tensor("token_embd.weight"), None);
        assert_eq!(split_block_tensor("blk.x.attn_k.weight"), None);
        assert_eq!(split_block_tensor("blk.7"), None);
    }
}
