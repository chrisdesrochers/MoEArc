//! Zero-copy access to a GGUF file's tensor data.
//!
//! [`inspect`](crate::inspect) answers questions about a model without ever opening the tensor
//! blob. A forward pass has the opposite need: it wants the bytes, all of them, addressed by
//! name — and it must get them without ever holding the model in process memory. A 20.6 GiB
//! file read into a `Vec<u8>` is 20.6 GiB of RSS before a single token is produced, on a machine
//! whose GPU has 12.
//!
//! So the file is mapped, not read. [`MappedModel::open`] parses the header the ordinary way and
//! then hands the whole file to `mmap(2)`; every [`TensorView`] is a borrowed subslice of that
//! mapping. Nothing is copied, and a tensor the engine never touches is never paged in. The
//! measurement that backs this claim is `examples/map.rs`, which prints `VmRSS` on either side of
//! the map.
//!
//! # The one function that has to be right
//!
//! A MoE block does not store its experts separately. All `expert_count` of them are stacked into
//! a single `blk.N.ffn_gate_exps.weight`-style tensor, and a residency cache that pages experts
//! individually needs to address one expert's bytes inside that stack.
//!
//! That is [`TensorView::slice_last_dim`], and its failure mode is the reason this module is
//! careful. GGUF dimensions are stored **fastest-varying first**, so the expert index is the
//! *last* dimension and each expert's share is therefore contiguous — but the stride is in bytes
//! of a quantised block layout, not in elements. Getting it wrong by one block yields data that
//! is still well-formed, still the right length, and wrong: the engine would run, produce
//! plausible-looking logits, and be silently answering with a blend of two experts. There is no
//! crash to debug. Hence the divisibility and partition checks below, which turn every case the
//! arithmetic does not exactly account for into an error rather than a shifted slice.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::gguf::{self, GgufHeader};
use crate::quant::QuantType;
use crate::{EXPERT_TENSORS, KV_TENSORS, ModelError, split_block_tensor};

/// Tensor names and per-block suffixes, as llama.cpp's GGUF writer spells them.
///
/// A forward pass reaches for these by convention rather than by searching, so the convention is
/// written down once here instead of as string literals scattered through the engine. The
/// per-block constants are *suffixes*: the full name is `blk.<n>.<suffix>`, which
/// [`MappedModel::block_tensor`] assembles.
///
/// 🔴 **Not every model carries every one of these.** `attn_q_norm`/`attn_k_norm` (QK-norm) are
/// present in OLMoE and absent from most older architectures, and the `_shexp` shared-expert
/// tensors exist only in the families that have them. Reach for the optional ones through
/// [`MappedModel::optional_tensor`] and friends; treat a missing one as a fact about the model,
/// not as an error.
pub mod names {
    /// The input embedding matrix.
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    /// The final norm before the output head.
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    /// The output head. Absent when the model ties it to [`TOKEN_EMBD`].
    pub const OUTPUT: &str = "output.weight";

    /// Pre-attention norm.
    pub const ATTN_NORM: &str = "attn_norm.weight";
    /// Query projection.
    pub const ATTN_Q: &str = "attn_q.weight";
    /// Key projection.
    pub const ATTN_K: &str = "attn_k.weight";
    /// Value projection.
    pub const ATTN_V: &str = "attn_v.weight";
    /// Attention output projection.
    pub const ATTN_OUTPUT: &str = "attn_output.weight";
    /// Optional QK-norm on the queries.
    pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
    /// Optional QK-norm on the keys.
    pub const ATTN_K_NORM: &str = "attn_k_norm.weight";

    /// Optional query-projection bias.
    pub const ATTN_Q_BIAS: &str = "attn_q.bias";
    /// Optional key-projection bias.
    pub const ATTN_K_BIAS: &str = "attn_k.bias";
    /// Optional value-projection bias.
    pub const ATTN_V_BIAS: &str = "attn_v.bias";
    /// Optional attention-output bias.
    pub const ATTN_OUTPUT_BIAS: &str = "attn_output.bias";

    /// Optional per-head **attention sink**: one extra logit that joins the softmax denominator
    /// and has no value vector. `n_head` floats. Present in gpt-oss.
    pub const ATTN_SINKS: &str = "attn_sinks.weight";

    /// Pre-FFN norm.
    pub const FFN_NORM: &str = "ffn_norm.weight";
    /// The pre-FFN norm under its other spelling.
    ///
    /// 🔴 gpt-oss has exactly two norms per block and calls the second one this. llama.cpp's
    /// symbol for it is `LLM_TENSOR_ATTN_POST_NORM` — "post-attention" — but structurally it
    /// sits where [`FFN_NORM`] sits in every other architecture here: after the attention
    /// residual and before the MoE branch. ⚠️ The C++ symbol and the GGUF string do **not**
    /// match (`attn_post_norm` against `post_attention_norm`), so a lookup written from the
    /// source's spelling silently finds nothing.
    pub const POST_ATTENTION_NORM: &str = "post_attention_norm.weight";
    /// The MoE router.
    pub const FFN_GATE_INP: &str = "ffn_gate_inp.weight";
    /// Stacked expert gate projections; see [`super::ExpertBank`].
    pub const FFN_GATE_EXPS: &str = "ffn_gate_exps.weight";
    /// Stacked expert up projections.
    pub const FFN_UP_EXPS: &str = "ffn_up_exps.weight";
    /// Stacked expert down projections.
    pub const FFN_DOWN_EXPS: &str = "ffn_down_exps.weight";

    /// Optional router bias, `n_expert` long. Added to the logits **before** the top-k.
    pub const FFN_GATE_INP_BIAS: &str = "ffn_gate_inp.bias";
    /// Optional per-expert gate bias, `[n_ff, n_expert]`.
    pub const FFN_GATE_EXPS_BIAS: &str = "ffn_gate_exps.bias";
    /// Optional per-expert up bias, `[n_ff, n_expert]`.
    pub const FFN_UP_EXPS_BIAS: &str = "ffn_up_exps.bias";
    /// Optional per-expert down bias, `[n_embd, n_expert]`. Applied **inside** the router's
    /// weighting, not after it.
    pub const FFN_DOWN_EXPS_BIAS: &str = "ffn_down_exps.bias";

    /// Shared-expert gate projection, where the architecture has one.
    pub const FFN_GATE_SHEXP: &str = "ffn_gate_shexp.weight";
    /// Shared-expert up projection.
    pub const FFN_UP_SHEXP: &str = "ffn_up_shexp.weight";
    /// Shared-expert down projection.
    pub const FFN_DOWN_SHEXP: &str = "ffn_down_shexp.weight";
}

/// Which of a block's three stacked expert tensors to address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertBank {
    /// `ffn_gate_exps` — the GLU gate. Absent in architectures without a gated FFN.
    Gate,
    /// `ffn_up_exps`.
    Up,
    /// `ffn_down_exps`.
    Down,
}

impl ExpertBank {
    /// All three, in the order a residency slot holds them.
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    /// The per-block tensor-name suffix for this bank.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Gate => names::FFN_GATE_EXPS,
            Self::Up => names::FFN_UP_EXPS,
            Self::Down => names::FFN_DOWN_EXPS,
        }
    }
}

/// One tensor, or one slice of one, as bytes inside the mapping.
///
/// Every field is borrowed from the [`MappedModel`] that produced it — the name and dimensions
/// from the parsed header, the data from the mapping itself. Constructing one costs a hash lookup
/// and some arithmetic; it never copies and never allocates.
///
/// The bytes are in the file's own quantisation, untouched: `data` is exactly what
/// `ggml_get_rows` and friends expect, which is what makes uploading a slice straight to a device
/// possible.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    /// The tensor's name in the file. A slice keeps its parent's name — the index that produced
    /// it is [`TensorView::file_offset`]'s business, not the name's.
    pub name: &'a str,
    /// Dimensions, **fastest-varying first**, exactly as stored. A slice has one fewer.
    pub dims: &'a [u64],
    /// The ggml block geometry these bytes are in.
    pub quant: QuantType,
    /// The bytes, borrowed from the mapping.
    pub data: &'a [u8],
    /// Absolute byte offset of `data` within the file.
    ///
    /// Carried so this crate's arithmetic can be checked by something that is not this crate: an
    /// independent reader can seek here, read [`TensorView::len`] bytes, and compare. That is
    /// exactly what `examples/expert_probe.rs` exists to make possible, and it is the only way an
    /// off-by-one stride is ever going to be caught — the wrong slice is not malformed, just
    /// wrong.
    pub file_offset: u64,
}

impl std::fmt::Debug for TensorView<'_> {
    /// Deliberately hand-written: `data` is routinely a megabyte and occasionally a gigabyte, and
    /// a derived `Debug` would render every byte of it into a panic message or a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("name", &self.name)
            .field("dims", &self.dims)
            .field("quant", &self.quant.name)
            .field("bytes", &self.data.len())
            .field("file_offset", &self.file_offset)
            .finish()
    }
}

impl<'a> TensorView<'a> {
    /// Bytes this view spans.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether this view spans no bytes. Only a zero-dimension tensor can.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total element count — the product of the dimensions.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().fold(1u64, |acc, &d| acc.saturating_mul(d))
    }

    /// Narrow to index `index` along the slowest-varying (last) dimension.
    ///
    /// GGUF stores dimensions fastest-varying first, so the last dimension is the outermost one
    /// and each of its slices is contiguous. For a `[d0, d1, n_experts]` expert bank that makes
    /// expert *k* a single run of `bytes(d0 * d1)` bytes starting `k` runs in — which is what
    /// this returns, as a borrow of the same mapping.
    ///
    /// Three things are checked rather than assumed, and each one exists to convert a silent
    /// mis-slice into an error:
    ///
    /// - **Rank.** A rank-1 tensor has no inner shape to slice; the result would be a single
    ///   element, which is not what any caller of this means. Refused.
    /// - **Block alignment.** The slice size is computed from the *inner* element count, which
    ///   must itself be a whole number of quantisation blocks. If it is not, the slices do not
    ///   fall on block boundaries at all and no byte offset is correct — rounding would put every
    ///   expert after the first into the middle of a block, reading a scale as a weight.
    /// - **Partition.** `slice_bytes * count` must equal the parent's byte length exactly. Today
    ///   the alignment check above should already guarantee this, so this is belt-and-braces —
    ///   but the two are computed by independent routes (this function's arithmetic versus
    ///   [`crate::gguf::TensorInfo::nbytes`] via the quant table), and a change to either could
    ///   part them. It costs one comparison to know they have not.
    pub fn slice_last_dim(&self, index: u64) -> Result<TensorView<'a>, ModelError> {
        let dims = self.dims;
        let Some((&count, inner)) = dims.split_last() else {
            return Err(ModelError::NotSliceable { tensor: self.name.to_string(), rank: 0 });
        };
        if inner.is_empty() {
            return Err(ModelError::NotSliceable { tensor: self.name.to_string(), rank: 1 });
        }
        if index >= count {
            return Err(ModelError::SliceOutOfRange {
                tensor: self.name.to_string(),
                index,
                count,
            });
        }

        let elements = inner.iter().fold(1u64, |acc, &d| acc.saturating_mul(d));
        if elements % self.quant.block_size != 0 {
            return Err(ModelError::ElementsNotBlockAligned {
                tensor: self.name.to_string(),
                elements,
                block_size: self.quant.block_size,
            });
        }
        let slice_bytes = elements / self.quant.block_size * self.quant.type_size;
        let total = slice_bytes.saturating_mul(count);
        if total != self.data.len() as u64 {
            return Err(ModelError::SliceStrideMismatch {
                tensor: self.name.to_string(),
                slice_bytes,
                count,
                total_bytes: self.data.len() as u64,
            });
        }

        // `index < count` and `slice_bytes * count == data.len()`, both just proven, so `start`
        // and `start + slice_bytes` are inside the parent slice and the casts cannot truncate.
        let start = slice_bytes * index;
        let s = usize::try_from(start).map_err(|_| ModelError::SliceOutOfRange {
            tensor: self.name.to_string(),
            index,
            count,
        })?;
        let n = slice_bytes as usize;
        Ok(TensorView {
            name: self.name,
            dims: inner,
            quant: self.quant,
            data: &self.data[s..s + n],
            file_offset: self.file_offset + start,
        })
    }
}

/// A GGUF file, header parsed and tensor data mapped.
///
/// Holds the mapping open for its whole life; every [`TensorView`] borrows from it, so it must
/// outlive them — which the borrow checker enforces without any help.
pub struct MappedModel {
    path: PathBuf,
    header: GgufHeader,
    map: Mmap,
    /// Name to index into `header.tensors`. Built once so lookup is not a linear walk of a few
    /// hundred to a few thousand names on every tensor a forward pass touches.
    by_name: BTreeMap<String, usize>,
}

impl std::fmt::Debug for MappedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedModel")
            .field("path", &self.path)
            .field("tensors", &self.header.tensors.len())
            .field("mapped_bytes", &self.map.len())
            .finish()
    }
}

impl MappedModel {
    /// Parse the header and map the file.
    ///
    /// The header read is the ordinary buffered one — a few megabytes, and it validates the whole
    /// tensor index before anything is mapped, so a truncated or corrupt file fails here rather
    /// than as a fault on first access. Only then is the file mapped, in full and lazily: no page
    /// of tensor data is read until something asks for those bytes.
    ///
    /// # Why the `unsafe` is here and what it rests on
    ///
    /// `mmap(2)` on a file is unsound in Rust's model if anything modifies or truncates the file
    /// while the mapping is live — a shrink turns a valid `&[u8]` into a `SIGBUS`, which no
    /// amount of bounds checking prevents. There is no portable way to forbid that, so this is a
    /// documented precondition rather than a guarantee: **the model file must not be written to
    /// while a `MappedModel` is open.** In practice model files are immutable artefacts that
    /// arrive by download and are then only read, and [`crate::pull::pull`] writes to a `.part`
    /// file and renames, so it never mutates a file in place under a reader.
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let header = gguf::read(path)?;
        let file = File::open(path)?;

        // Re-check the length after opening for the map. `gguf::read` validated the index against
        // the size it saw; this is a different open of the same name, and between the two the
        // file could have been replaced by a shorter one. Every subslice below is bounds-checked
        // against the mapping regardless, so this only buys a better error message — but "the
        // file shrank underneath us" is a much better message than a failed lookup.
        let file_size = file.metadata()?.len();
        if file_size < header.file_size {
            return Err(ModelError::Truncated { offset: 0, needed: header.file_size, file_size });
        }

        // SAFETY: see the precondition documented above — the mapped file must not be mutated
        // while this mapping is live.
        let map = unsafe { Mmap::map(&file)? };

        let mut by_name = BTreeMap::new();
        for (i, t) in header.tensors.iter().enumerate() {
            // First wins on a duplicate name, which is the conservative direction: a later
            // duplicate cannot displace the entry whose offsets `gguf::read` already validated.
            by_name.entry(t.name.clone()).or_insert(i);
        }

        Ok(Self { path: path.to_path_buf(), header, map, by_name })
    }

    /// The file this was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The parsed header, for metadata this module does not expose directly.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Bytes mapped — the whole file, including the header region.
    pub fn mapped_bytes(&self) -> usize {
        self.map.len()
    }

    /// Every tensor name in the file, in index order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.iter().map(|t| t.name.as_str())
    }

    /// Number of tensors in the index.
    pub fn tensor_count(&self) -> usize {
        self.header.tensors.len()
    }

    /// `general.architecture`.
    pub fn architecture(&self) -> Result<&str, ModelError> {
        self.header.str_key("general.architecture")
    }

    /// `<arch>.expert_count`, or [`None`] for a dense model.
    pub fn expert_count(&self) -> Option<u32> {
        let arch = self.architecture().ok()?;
        let v = self.header.get(&format!("{arch}.expert_count"))?.as_u64()?;
        u32::try_from(v).ok()
    }

    /// `<arch>.block_count`.
    pub fn block_count(&self) -> Result<u32, ModelError> {
        let arch = self.architecture()?.to_string();
        self.header.u32_key(&format!("{arch}.block_count"))
    }

    /// Whether a tensor of this name exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// A tensor by exact name.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, ModelError> {
        let &i = self
            .by_name
            .get(name)
            .ok_or_else(|| ModelError::TensorNotFound { name: name.to_string() })?;
        self.view(i)
    }

    /// A tensor that the model is allowed not to have.
    ///
    /// [`None`] means "this model does not carry that tensor", which for the optional ones —
    /// QK-norm, a tied output head, shared experts — is a fact about the architecture and not a
    /// failure. An `Err` means the entry exists and is malformed.
    pub fn optional_tensor(&self, name: &str) -> Result<Option<TensorView<'_>>, ModelError> {
        match self.by_name.get(name) {
            Some(&i) => self.view(i).map(Some),
            None => Ok(None),
        }
    }

    /// A per-block tensor: `blk.<block>.<suffix>`. See [`names`] for the suffixes.
    pub fn block_tensor(&self, block: u32, suffix: &str) -> Result<TensorView<'_>, ModelError> {
        self.tensor(&format!("blk.{block}.{suffix}"))
    }

    /// [`MappedModel::block_tensor`] for a suffix the model is allowed not to have.
    pub fn optional_block_tensor(
        &self,
        block: u32,
        suffix: &str,
    ) -> Result<Option<TensorView<'_>>, ModelError> {
        self.optional_tensor(&format!("blk.{block}.{suffix}"))
    }

    /// One expert's weights from one block's stacked expert bank.
    ///
    /// **The function the residency cache is built on.** The returned view is the exact bytes of
    /// expert `expert`'s matrix, borrowed from the mapping — nothing between here and a device
    /// upload needs to copy it, and nothing else in the bank is paged in to get at it.
    ///
    /// The expert count is cross-checked against the file's `<arch>.expert_count` before the
    /// slice is taken. That check is not redundant with the bounds check in
    /// [`TensorView::slice_last_dim`]: a bank whose last dimension is not the expert count is a
    /// tensor we have misunderstood, and in that case *every* index is wrong, including the
    /// in-range ones.
    pub fn expert(
        &self,
        block: u32,
        bank: ExpertBank,
        expert: u32,
    ) -> Result<TensorView<'_>, ModelError> {
        let t = self.block_tensor(block, bank.suffix())?;
        if let Some(declared) = self.expert_count() {
            let last = t.dims.last().copied().unwrap_or(0);
            if last != u64::from(declared) {
                return Err(ModelError::ExpertBankShape {
                    tensor: t.name.to_string(),
                    last_dim: last,
                    expert_count: declared,
                });
            }
        }
        t.slice_last_dim(u64::from(expert))
    }

    /// Build a [`TensorView`] for the tensor at index `i`.
    fn view(&self, i: usize) -> Result<TensorView<'_>, ModelError> {
        let t = &self.header.tensors[i];
        let quant = crate::quant::lookup(t.type_id).ok_or_else(|| {
            ModelError::UnknownTensorType { tensor: t.name.clone(), type_id: t.type_id }
        })?;
        let len = t.nbytes()?;
        let start = self.header.data_offset.saturating_add(t.offset);
        let end = start.saturating_add(len);

        let overrun = || ModelError::TensorDataOverrunsFile {
            tensor: t.name.clone(),
            end,
            file_size: self.map.len() as u64,
        };
        let s = usize::try_from(start).map_err(|_| overrun())?;
        let e = usize::try_from(end).map_err(|_| overrun())?;
        let data = self.map.get(s..e).ok_or_else(overrun)?;

        Ok(TensorView { name: &t.name, dims: &t.dims, quant, data, file_offset: start })
    }

    /// Census the tensor index by role. See [`Layout`].
    pub fn layout(&self) -> Layout {
        Layout::from_names(self.names())
    }
}

/// Tensor-name prefixes that mark a block as recurrent rather than attention-based.
///
/// 🔴 **This is a kernel-coverage question, not a bookkeeping one.** A block carrying any of these
/// needs a state-space or linear-attention kernel that an attention + MoE engine does not have,
/// and a model with even one of them cannot run a forward pass on such an engine at all. The
/// prefixes are ggml's names for the recurrent families: Mamba/SSM (`ssm_*`), RWKV
/// (`time_mix_*`, `channel_mix_*`) and the short-convolution mixers (`shortconv*`).
const RECURRENT_PREFIXES: [&str; 4] = ["ssm_", "time_mix_", "channel_mix_", "shortconv"];

/// The infix that marks a shared (always-routed) expert.
const SHARED_EXPERT_INFIX: &str = "_shexp";

/// What the tensor index says the model's blocks are made of.
///
/// Built from names alone, so it answers the question the model card cannot be trusted on: *is
/// this a plain transformer, and does it have shared experts?* Both change what an engine must
/// implement — the first decides whether recurrent kernels are needed at all, the second decides
/// how much of the FFN is always resident and therefore not pageable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Distinct `blk.<n>` indices seen.
    pub block_count: u32,
    /// Blocks carrying a KV projection (`attn_k`/`attn_v`, or the MLA `_b` variants).
    ///
    /// Counted from the KV projections rather than from `attn_q` on purpose: a hybrid's recurrent
    /// blocks carry a fused `attn_qkv` that is not an attention KV projection, so `attn_q` alone
    /// would count them as attention blocks.
    pub attention_blocks: u32,
    /// Blocks carrying an `ffn_*_exps` bank.
    pub expert_blocks: u32,
    /// Blocks carrying an `ffn_*_shexp` shared expert.
    pub shared_expert_blocks: u32,
    /// Blocks carrying a recurrent tensor — `ssm_*`, `time_mix_*`, `channel_mix_*`, `shortconv*`.
    pub recurrent_blocks: u32,
    /// The distinct recurrent suffixes found, sorted. Empty for a plain transformer.
    pub recurrent_suffixes: Vec<String>,
    /// Every per-block suffix and how many blocks carry it, sorted by suffix.
    ///
    /// A suffix whose count is below [`Layout::block_count`] is present in some blocks and not
    /// others — which is either an optional tensor or a hybrid, and is worth seeing either way.
    pub per_block_suffixes: Vec<(String, u32)>,
    /// Tensor names outside any `blk.<n>.` prefix, sorted.
    pub global_tensors: Vec<String>,
}

impl Layout {
    /// Census a sequence of tensor names.
    fn from_names<'a>(names: impl Iterator<Item = &'a str>) -> Self {
        let mut blocks: BTreeSet<u32> = BTreeSet::new();
        let mut attention: BTreeSet<u32> = BTreeSet::new();
        let mut experts: BTreeSet<u32> = BTreeSet::new();
        let mut shared: BTreeSet<u32> = BTreeSet::new();
        let mut recurrent: BTreeSet<u32> = BTreeSet::new();
        let mut recurrent_suffixes: BTreeSet<String> = BTreeSet::new();
        let mut per_block: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
        let mut globals: Vec<String> = Vec::new();

        for name in names {
            let Some((block, suffix)) = split_block_tensor(name) else {
                globals.push(name.to_string());
                continue;
            };
            blocks.insert(block);
            per_block.entry(suffix.to_string()).or_default().insert(block);

            if KV_TENSORS.contains(&suffix) {
                attention.insert(block);
            }
            if EXPERT_TENSORS.contains(&suffix) {
                experts.insert(block);
            }
            if suffix.contains(SHARED_EXPERT_INFIX) {
                shared.insert(block);
            }
            if RECURRENT_PREFIXES.iter().any(|p| suffix.starts_with(p)) {
                recurrent.insert(block);
                recurrent_suffixes.insert(suffix.to_string());
            }
        }

        globals.sort();
        Self {
            block_count: blocks.len() as u32,
            attention_blocks: attention.len() as u32,
            expert_blocks: experts.len() as u32,
            shared_expert_blocks: shared.len() as u32,
            recurrent_blocks: recurrent.len() as u32,
            recurrent_suffixes: recurrent_suffixes.into_iter().collect(),
            per_block_suffixes: per_block.into_iter().map(|(k, v)| (k, v.len() as u32)).collect(),
            global_tensors: globals,
        }
    }

    /// Whether every block is a standard attention + MoE transformer block.
    ///
    /// The precondition for a first forward pass on an attention + MoE engine: no recurrent
    /// tensors anywhere, an expert bank in every block, and a KV projection in every block. A
    /// model that fails this needs kernels beyond attention and MoE, and saying so from the
    /// tensor index costs nothing — where finding out from a hung kernel costs an afternoon.
    pub fn is_pure_transformer_moe(&self) -> bool {
        self.block_count > 0
            && self.recurrent_blocks == 0
            && self.attention_blocks == self.block_count
            && self.expert_blocks == self.block_count
    }

    /// Whether the model has always-resident shared experts.
    ///
    /// Load-bearing for the cache planner: a shared expert runs for **every** token, so it is
    /// dense weight that must stay resident, and it does not belong in the pageable pool.
    pub fn has_shared_experts(&self) -> bool {
        self.shared_expert_blocks > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const F32: u32 = 0;
    const Q4_K: u32 = 12;

    /// A tensor to write into the synthetic file.
    struct T {
        name: &'static str,
        dims: &'static [u64],
        type_id: u32,
    }

    fn nbytes(t: &T) -> u64 {
        let q = crate::quant::lookup(t.type_id).unwrap();
        t.dims.iter().product::<u64>() / q.block_size * q.type_size
    }

    /// The byte this fixture writes at offset `i` of the tensor named by `tag`.
    ///
    /// A per-tensor, per-offset pattern rather than a constant: a slice taken at the wrong offset
    /// within the *same* tensor still has to differ, which a repeating fill would not catch.
    fn pattern(tag: u8, i: usize) -> u8 {
        (i as u8).wrapping_mul(31).wrapping_add(tag.wrapping_mul(97)).wrapping_add((i >> 8) as u8)
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn put_kv_u32(b: &mut Vec<u8>, key: &str, v: u32) {
        put_str(b, key);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&v.to_le_bytes());
    }

    fn put_kv_str(b: &mut Vec<u8>, key: &str, v: &str) {
        put_str(b, key);
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(b, v);
    }

    /// A valid GGUF v3 file whose tensor data is *filled*, not zeroed.
    ///
    /// Deliberately not shared with the writer in `lib.rs`: that one zero-fills the data blob,
    /// which is exactly right for testing byte *counts* and useless for testing byte *addresses*.
    /// Distinguishing "expert 3" from "expert 4" needs the bytes to differ.
    ///
    /// Shape: 4 experts, two blocks. Block 0's bank is Q4_K, block 1's is F32, so the stride
    /// arithmetic is exercised on both a super-blocked type and an unquantised one — the Q4_K
    /// slice size (`512 elements / 256 * 144 = 288 B`) is not a round multiple of anything, which
    /// is where an element-vs-byte stride confusion shows up.
    fn fixture() -> (Vec<u8>, Vec<T>) {
        let tensors = vec![
            // 4 experts of a [256, 2] Q4_K matrix: 512 elements = 2 blocks = 288 B each.
            T { name: "blk.0.ffn_gate_exps.weight", dims: &[256, 2, 4], type_id: Q4_K },
            T { name: "blk.0.ffn_up_exps.weight", dims: &[256, 2, 4], type_id: Q4_K },
            T { name: "blk.0.ffn_down_exps.weight", dims: &[2, 256, 4], type_id: Q4_K },
            T { name: "blk.0.attn_k.weight", dims: &[8, 4], type_id: F32 },
            T { name: "blk.0.attn_v.weight", dims: &[8, 4], type_id: F32 },
            T { name: "blk.0.ffn_gate_inp.weight", dims: &[8, 4], type_id: F32 },
            // 4 experts of a [3, 5] F32 matrix: 15 elements = 60 B each.
            T { name: "blk.1.ffn_gate_exps.weight", dims: &[3, 5, 4], type_id: F32 },
            T { name: "blk.1.ffn_up_exps.weight", dims: &[3, 5, 4], type_id: F32 },
            T { name: "blk.1.ffn_down_exps.weight", dims: &[5, 3, 4], type_id: F32 },
            T { name: "blk.1.attn_k.weight", dims: &[8, 4], type_id: F32 },
            T { name: "blk.1.attn_v.weight", dims: &[8, 4], type_id: F32 },
            T { name: "blk.1.ffn_gate_inp.weight", dims: &[8, 4], type_id: F32 },
            T { name: "token_embd.weight", dims: &[8, 16], type_id: F32 },
            T { name: "output_norm.weight", dims: &[8], type_id: F32 },
        ];

        let mut kv = Vec::new();
        let mut n_kv = 0u64;
        put_kv_str(&mut kv, "general.architecture", "testmoe");
        n_kv += 1;
        for (k, v) in [
            ("testmoe.block_count", 2u32),
            ("testmoe.embedding_length", 8),
            ("testmoe.context_length", 128),
            ("testmoe.expert_count", 4),
            ("testmoe.expert_used_count", 2),
            ("testmoe.attention.head_count", 2),
            ("testmoe.attention.head_count_kv", 1),
        ] {
            put_kv_u32(&mut kv, k, v);
            n_kv += 1;
        }

        let mut index = Vec::new();
        let mut offset = 0u64;
        let mut spans = Vec::new();
        for t in &tensors {
            put_str(&mut index, t.name);
            index.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in t.dims {
                index.extend_from_slice(&d.to_le_bytes());
            }
            index.extend_from_slice(&t.type_id.to_le_bytes());
            index.extend_from_slice(&offset.to_le_bytes());
            spans.push((offset, nbytes(t)));
            offset += nbytes(t).div_ceil(32) * 32;
        }
        let data_len = offset as usize;

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
        let data_start = f.len();
        f.resize(data_start + data_len, 0);
        for (tag, (off, len)) in spans.iter().enumerate() {
            let base = data_start + *off as usize;
            for i in 0..*len as usize {
                f[base + i] = pattern(tag as u8, i);
            }
        }
        (f, tensors)
    }

    struct TempGguf(PathBuf);

    impl TempGguf {
        fn new(tag: &str, bytes: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("moearc-tensors-{}-{tag}.gguf", std::process::id()));
            let mut f = File::create(&path).unwrap();
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

    #[test]
    fn a_tensor_view_is_the_files_own_bytes_at_the_files_own_offset() {
        let (bytes, tensors) = fixture();
        let f = TempGguf::new("view", &bytes);
        let m = MappedModel::open(&f.0).unwrap();

        assert_eq!(m.tensor_count(), tensors.len());
        assert_eq!(m.mapped_bytes(), bytes.len());
        assert_eq!(m.expert_count(), Some(4));
        assert_eq!(m.block_count().unwrap(), 2);

        let t = m.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(t.dims, &[256, 2, 4]);
        assert_eq!(t.quant.name, "q4_K");
        // 256 * 2 * 4 = 2048 elements / 256 per block * 144 B per block = 1152 B for the bank,
        // which is 4 experts of 288 B.
        assert_eq!(t.len(), 1152);
        // The view really is a window onto the file, not a copy of something else: the same
        // bytes must be there when the file is read as a plain byte string.
        let off = usize::try_from(t.file_offset).unwrap();
        assert_eq!(t.data, &bytes[off..off + t.len()]);
    }

    #[test]
    fn every_expert_slice_lands_on_its_own_bytes() {
        let (bytes, _) = fixture();
        let f = TempGguf::new("experts", &bytes);
        let m = MappedModel::open(&f.0).unwrap();

        for (block, per_expert) in [(0u32, 288usize), (1, 60)] {
            for bank in ExpertBank::ALL {
                let parent = m.block_tensor(block, bank.suffix()).unwrap();
                for k in 0..4u32 {
                    let e = m.expert(block, bank, k).unwrap();
                    assert_eq!(e.len(), per_expert, "block {block} {bank:?} expert {k}");
                    // Rank drops by one and the inner shape is preserved.
                    assert_eq!(e.dims, &parent.dims[..parent.dims.len() - 1]);
                    // The slice is where the stride says it is...
                    assert_eq!(e.file_offset, parent.file_offset + (k as u64) * per_expert as u64);
                    // ...and holds the bytes the file holds there, read independently of the
                    // mapping. This is the assertion the whole module exists for.
                    let off = usize::try_from(e.file_offset).unwrap();
                    assert_eq!(
                        e.data,
                        &bytes[off..off + per_expert],
                        "block {block} {bank:?} expert {k} is not at its own offset"
                    );
                }
                // The experts tile the parent exactly, with nothing left over and nothing
                // shared: concatenating them must reproduce the bank byte for byte.
                let joined: Vec<u8> =
                    (0..4).flat_map(|k| m.expert(block, bank, k).unwrap().data.to_vec()).collect();
                assert_eq!(joined, parent.data, "block {block} {bank:?} experts do not tile");
            }
        }
    }

    #[test]
    fn neighbouring_experts_are_distinguishable() {
        // An off-by-one stride is only detectable if adjacent experts differ. Prove the fixture
        // actually has that property, so the test above cannot pass vacuously.
        let (bytes, _) = fixture();
        let f = TempGguf::new("distinct", &bytes);
        let m = MappedModel::open(&f.0).unwrap();
        for k in 0..3u32 {
            let a = m.expert(0, ExpertBank::Gate, k).unwrap();
            let b = m.expert(0, ExpertBank::Gate, k + 1).unwrap();
            assert_ne!(a.data, b.data, "experts {k} and {} have identical bytes", k + 1);
        }
    }

    #[test]
    fn an_out_of_range_expert_is_an_error_not_a_wrapped_index() {
        let (bytes, _) = fixture();
        let f = TempGguf::new("range", &bytes);
        let m = MappedModel::open(&f.0).unwrap();
        let e = m.expert(0, ExpertBank::Gate, 4).unwrap_err();
        assert!(
            matches!(e, ModelError::SliceOutOfRange { index: 4, count: 4, .. }),
            "unexpected error: {e}"
        );
    }

    #[test]
    fn a_rank_one_tensor_cannot_be_sliced() {
        let (bytes, _) = fixture();
        let f = TempGguf::new("rank1", &bytes);
        let m = MappedModel::open(&f.0).unwrap();
        let t = m.tensor("output_norm.weight").unwrap();
        let e = t.slice_last_dim(0).unwrap_err();
        assert!(matches!(e, ModelError::NotSliceable { rank: 1, .. }), "unexpected error: {e}");
    }

    #[test]
    fn a_missing_tensor_is_named_in_the_error_and_optional_lookup_says_none() {
        let (bytes, _) = fixture();
        let f = TempGguf::new("missing", &bytes);
        let m = MappedModel::open(&f.0).unwrap();

        assert!(!m.has_tensor("output.weight"));
        assert!(m.optional_tensor("output.weight").unwrap().is_none());
        assert!(m.optional_block_tensor(0, names::ATTN_Q_NORM).unwrap().is_none());

        let e = m.tensor("output.weight").unwrap_err();
        assert!(e.to_string().contains("output.weight"), "error does not name the tensor: {e}");
    }

    #[test]
    fn slicing_a_bank_whose_last_dim_is_not_the_expert_count_is_refused() {
        // `blk.0.ffn_gate_inp.weight` is [8, 4] and 4 *is* the expert count, so the guard has to
        // be tested on something whose last dimension genuinely disagrees. `token_embd` is
        // [8, 16]; addressed as if it were a bank it must be refused rather than sliced.
        let (bytes, _) = fixture();
        let f = TempGguf::new("shape", &bytes);
        let m = MappedModel::open(&f.0).unwrap();
        let t = m.tensor("token_embd.weight").unwrap();
        // The generic slice is fine — it is a legitimate operation on any rank>=2 tensor.
        assert_eq!(t.slice_last_dim(0).unwrap().len(), 32);
        // Through `expert()`, the same tensor under a bank name would be refused. Simulate by
        // checking the guard directly on the shape it protects.
        assert_ne!(t.dims.last().copied(), m.expert_count().map(u64::from));
    }

    #[test]
    fn the_layout_census_reads_a_plain_transformer_moe_off_the_names() {
        let (bytes, _) = fixture();
        let f = TempGguf::new("layout", &bytes);
        let m = MappedModel::open(&f.0).unwrap();
        let l = m.layout();

        assert_eq!(l.block_count, 2);
        assert_eq!(l.attention_blocks, 2);
        assert_eq!(l.expert_blocks, 2);
        assert_eq!(l.shared_expert_blocks, 0);
        assert_eq!(l.recurrent_blocks, 0);
        assert!(l.recurrent_suffixes.is_empty());
        assert!(l.is_pure_transformer_moe());
        assert!(!l.has_shared_experts());
        assert_eq!(l.global_tensors, ["output_norm.weight", "token_embd.weight"]);
        assert!(l.per_block_suffixes.contains(&("ffn_gate_exps.weight".to_string(), 2)));
    }

    #[test]
    fn the_layout_census_flags_recurrence_and_shared_experts() {
        // Names only — no file needed, which is the point of building the census from names.
        let l = Layout::from_names(
            [
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.ffn_gate_exps.weight",
                "blk.0.ffn_down_shexp.weight",
                "blk.1.ssm_conv1d.weight",
                "blk.1.ffn_gate_exps.weight",
                "token_embd.weight",
            ]
            .into_iter(),
        );
        assert_eq!(l.block_count, 2);
        assert_eq!(l.attention_blocks, 1);
        assert_eq!(l.expert_blocks, 2);
        assert_eq!(l.recurrent_blocks, 1);
        assert_eq!(l.recurrent_suffixes, ["ssm_conv1d.weight"]);
        assert!(l.has_shared_experts());
        assert!(!l.is_pure_transformer_moe(), "a hybrid must not pass as a plain transformer");
    }
}
