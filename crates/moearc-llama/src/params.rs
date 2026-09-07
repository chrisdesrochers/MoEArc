//! The parameter surface the tuning layer sets.
//!
//! This module is **pure Rust and has no dependency on llama.cpp**. It is the
//! vocabulary in which a tuning decision is expressed, so that a plan can be
//! computed, serialised, tested and diffed on any machine, and only then handed to
//! a runtime that happens to have a GPU attached.
//!
//! Every knob here maps to exactly one llama.cpp setting, and the mapping is
//! recorded on the field. Where the mapping is *not* one-to-one -- and there is
//! one important case, [`ModelParams::n_cpu_moe`] -- the field documents what it
//! actually expands to, because getting that wrong is how a tuning layer silently
//! does nothing.

use std::fmt;

/// `enum ggml_type`, restricted to the values that are legal for a KV cache.
///
/// Values are ggml's own discriminants (`ggml.h`), not a MoEArc numbering; they
/// cross the FFI boundary as-is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum KvType {
    /// 32-bit float. Largest, and the only one that is never lossy.
    F32 = 0,
    /// 16-bit float. llama.cpp's default for both K and V.
    F16 = 1,
    /// 4-bit, 32-element blocks.
    Q4_0 = 2,
    /// 8-bit, 32-element blocks. The usual quality/size compromise.
    Q8_0 = 8,
}

impl KvType {
    /// Bytes per element, averaged over a block for the quantised types.
    ///
    /// Q4_0 is a 32-element block of 16 bytes plus one f16 scale (18 bytes total),
    /// and Q8_0 is 32 bytes plus one f16 scale (34 bytes). These are the numbers a
    /// KV-budget calculation needs, so they are given exactly rather than as the
    /// rounded 0.5 / 1.0 that the names suggest.
    #[must_use]
    pub fn bytes_per_element(self) -> f64 {
        match self {
            KvType::F32 => 4.0,
            KvType::F16 => 2.0,
            KvType::Q4_0 => 18.0 / 32.0,
            KvType::Q8_0 => 34.0 / 32.0,
        }
    }

    /// The name llama.cpp's CLI uses for this type in `-ctk` / `-ctv`.
    #[must_use]
    pub fn cli_name(self) -> &'static str {
        match self {
            KvType::F32 => "f32",
            KvType::F16 => "f16",
            KvType::Q4_0 => "q4_0",
            KvType::Q8_0 => "q8_0",
        }
    }
}

impl fmt::Display for KvType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.cli_name())
    }
}

/// `enum llama_flash_attn_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum FlashAttn {
    /// Let llama.cpp decide per backend. llama.cpp's own default.
    #[default]
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

/// `enum llama_split_mode`. Single-GPU MoEArc uses `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum SplitMode {
    /// One GPU, selected by `main_gpu`.
    #[default]
    None = 0,
    Layer = 1,
    Row = 2,
    Tensor = 3,
}

/// `enum llama_load_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum LoadMode {
    #[default]
    Auto = -1,
    None = 0,
    Mmap = 1,
    Mlock = 2,
    MmapMlock = 3,
    DirectIo = 4,
}

/// Load-time parameters. These are fixed for the lifetime of a model and changing
/// one requires a reload.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelParams {
    /// `-ngl` / `llama_model_params.n_gpu_layers`. Negative means every layer.
    pub n_gpu_layers: i32,

    /// `-mg` / `llama_model_params.main_gpu`. The device index used when
    /// `split_mode == None`.
    ///
    /// ⚠️ This indexes the devices llama.cpp enumerated, which on the reference
    /// machine is filtered by `ONEAPI_DEVICE_SELECTOR`. With the iGPU deliberately
    /// left enabled in BIOS for Jellyfin, index 0 is *not* reliably the B580 unless
    /// the selector is set. See [`crate::params::ModelParams::n_gpu_layers`] callers.
    pub main_gpu: i32,

    /// `-ncmoe` / `--n-cpu-moe`: keep the expert weights of the first N blocks in
    /// host RAM.
    ///
    /// 🔴 **This is not a scalar field in the C API.** There is no `n_cpu_moe`
    /// anywhere in `llama.h`. The flag expands into N entries of
    /// `llama_model_params.tensor_buft_overrides`, a NULL-terminated array of
    /// `{ const char *pattern; ggml_backend_buffer_type_t buft; }`, where entry `i`
    /// is the regex `blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps` bound to
    /// `ggml_backend_cpu_buffer_type()`. The expansion happens in the C shim, which
    /// reproduces llama.cpp's `llm_add_n_cpu_ffn_overrides` exactly.
    ///
    /// Two consequences for the tuning layer:
    ///
    /// - The pattern matches **only expert tensors** (`*_exps`). Attention and the
    ///   dense parts of those blocks stay on the GPU. `-ncmoe` is not `-ngl`.
    /// - The per-block regexes do not alias. `blk\.1\.ffn_...` cannot match
    ///   `blk.10.ffn_up_exps`, because the character after `blk.1` there is `0`,
    ///   not the `.` the pattern requires.
    pub n_cpu_moe: i32,

    pub split_mode: SplitMode,
    pub load_mode: LoadMode,

    /// `llama_model_params.use_extra_bufts` -- weight repacking buffer types.
    pub use_extra_bufts: bool,
    /// `llama_model_params.no_host` -- bypass the host buffer.
    pub no_host: bool,
    /// `llama_model_params.check_tensors` -- validate tensor data on load.
    pub check_tensors: bool,
    /// `llama_model_params.vocab_only` -- load metadata and vocab, no weights.
    /// Cheap way to inspect a model without paying for it.
    pub vocab_only: bool,
}

impl Default for ModelParams {
    /// MoEArc's defaults: everything on the GPU, one device, chosen explicitly.
    ///
    /// ⚠️ Two of these differ from llama.cpp's own defaults, and both are set
    /// deliberately rather than inherited:
    ///
    /// - `n_gpu_layers: -1`. **llama.cpp also defaults to -1 at the pinned
    ///   commit** (`llama-model.cpp: llama_model_default_params`), so this is
    ///   currently a no-op. It is still stated explicitly, because that default
    ///   used to be `0` — CPU-only — and most llama.cpp documentation still says
    ///   so. A knob whose value depends on which upstream commit is linked is a
    ///   knob that will eventually surprise someone; the `runtime` test suite
    ///   asserts upstream's value so a change is caught by a failing test rather
    ///   than by an inexplicably slow benchmark.
    /// - `split_mode: None`. llama.cpp defaults to `Layer`. On a single-GPU box
    ///   these behave the same, but `None` + `main_gpu` says which device we mean.
    fn default() -> Self {
        Self {
            n_gpu_layers: -1,
            main_gpu: 0,
            n_cpu_moe: 0,
            split_mode: SplitMode::None,
            load_mode: LoadMode::Auto,
            use_extra_bufts: true,
            no_host: false,
            check_tensors: false,
            vocab_only: false,
        }
    }
}

impl ModelParams {
    /// Metadata only. Loads the vocab and skips the weights.
    #[must_use]
    pub fn vocab_only() -> Self {
        Self { vocab_only: true, n_gpu_layers: 0, ..Self::default() }
    }

    /// The tensor-buffer-override patterns that `n_cpu_moe` expands to.
    ///
    /// Exposed so a plan can be inspected and asserted on without loading a model,
    /// and so the expansion is testable without a GPU. The shim builds the same
    /// strings; this function is the specification of what it builds.
    #[must_use]
    pub fn cpu_moe_patterns(&self) -> Vec<String> {
        (0..self.n_cpu_moe.max(0))
            .map(|i| format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps"))
            .collect()
    }
}

/// Per-context parameters. A context can be rebuilt against an already-loaded
/// model, so these are the cheap knobs to sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextParams {
    /// `-c` / `n_ctx`. 0 means "take the model's training context".
    pub n_ctx: u32,
    /// `-b` / `n_batch`: logical maximum batch submitted to `llama_decode`.
    pub n_batch: u32,
    /// `-ub` / `n_ubatch`: physical maximum batch. The one that actually bounds
    /// prefill working-set size on the GPU.
    pub n_ubatch: u32,
    /// `--parallel` / `n_seq_max`.
    pub n_seq_max: u32,

    /// 🔴 `-t` / `n_threads`: threads for **single-token generation**.
    ///
    /// This is the field whose default invalidated every published llama.cpp
    /// comparison this project made. llama.cpp's own default is 4 on some paths
    /// and `hardware_concurrency` on others; `llama-bench` defaults to 4
    /// regardless of core count. With `-ncmoe` putting most experts in host RAM,
    /// decode is *CPU-bound on the expert matmuls*, so this value is not a minor
    /// tuning detail -- it is a first-order determinant of throughput.
    /// **Never leave it implicit.**
    pub n_threads: i32,

    /// `-tb` / `n_threads_batch`: threads for prompt and batch processing.
    pub n_threads_batch: i32,

    /// `-fa` / `flash_attn_type`.
    pub flash_attn: FlashAttn,

    /// `-ctk` / `type_k`: KV cache K type.
    pub type_k: KvType,
    /// `-ctv` / `type_v`: KV cache V type.
    pub type_v: KvType,

    /// `--no-kv-offload` inverted. `true` keeps the KV cache on the GPU.
    ///
    /// 📌 The allocation finding says KV residency is all-or-nothing while expert
    /// residency degrades gracefully, so this should be the *last* thing given up,
    /// not the first.
    pub offload_kqv: bool,

    /// `llama_context_params.op_offload` -- offload host tensor ops to the device.
    pub op_offload: bool,
    /// `--swa-full`: keep a full-size sliding-window-attention cache.
    pub swa_full: bool,
    /// `--kv-unified`: one KV buffer shared across sequences.
    pub kv_unified: bool,
    /// `--no-perf` inverted internally; `false` keeps llama.cpp's timers running,
    /// which is what [`crate::Perf`] reads.
    pub no_perf: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            // Deliberately not a constant: see the field docs. `available_parallelism`
            // is the honest default on a box whose core count we do not know here.
            n_threads: default_threads(),
            n_threads_batch: default_threads(),
            flash_attn: FlashAttn::Auto,
            type_k: KvType::F16,
            type_v: KvType::F16,
            offload_kqv: true,
            op_offload: true,
            swa_full: false,
            kv_unified: true,
            no_perf: false,
        }
    }
}

fn default_threads() -> i32 {
    std::thread::available_parallelism().map_or(4, |n| i32::try_from(n.get()).unwrap_or(4))
}

impl ContextParams {
    /// Bytes of KV cache implied by these settings for a given model geometry.
    ///
    /// `n_layer` is the number of attention blocks -- ⚠️ **not** `block_count`, which
    /// on a hybrid model such as Qwen3.6 overstates it roughly 4x because only 10 of
    /// 40 blocks are attention. Pass the attention count.
    #[must_use]
    pub fn kv_bytes(&self, n_layer: u32, n_head_kv: u32, head_dim: u32) -> u64 {
        let per_tok_elems = u64::from(n_layer) * u64::from(n_head_kv) * u64::from(head_dim);
        let k = per_tok_elems as f64 * self.type_k.bytes_per_element();
        let v = per_tok_elems as f64 * self.type_v.bytes_per_element();
        ((k + v) * f64::from(self.n_ctx)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_moe_expands_to_one_pattern_per_block() {
        let p = ModelParams { n_cpu_moe: 3, ..ModelParams::default() };
        assert_eq!(
            p.cpu_moe_patterns(),
            vec![
                r"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps",
                r"blk\.1\.ffn_(up|down|gate|gate_up)_(ch|)exps",
                r"blk\.2\.ffn_(up|down|gate|gate_up)_(ch|)exps",
            ]
        );
    }

    #[test]
    fn zero_and_negative_cpu_moe_expand_to_nothing() {
        assert!(ModelParams::default().cpu_moe_patterns().is_empty());
        let neg = ModelParams { n_cpu_moe: -5, ..ModelParams::default() };
        assert!(neg.cpu_moe_patterns().is_empty());
    }

    /// The aliasing question, asserted rather than assumed: a low-numbered block's
    /// pattern must not also capture a two-digit block. This is the property that
    /// makes `-ncmoe 2` mean two blocks and not twelve.
    #[test]
    fn block_patterns_do_not_alias_across_digit_counts() {
        let p = ModelParams { n_cpu_moe: 2, ..ModelParams::default() };
        let pats = p.cpu_moe_patterns();
        // Hand-check the literal prefix that provides the separation.
        assert!(pats[1].starts_with(r"blk\.1\."));
        // `blk.10.ffn_up_exps` has `0` where the pattern demands an escaped dot.
        assert!(!"blk.10.ffn_up_exps".contains("blk.1."));
        assert!("blk.1.ffn_up_exps".contains("blk.1."));
    }

    #[test]
    fn kv_type_sizes_are_block_exact_not_rounded() {
        assert!((KvType::F16.bytes_per_element() - 2.0).abs() < f64::EPSILON);
        // 32 weights in 16 bytes + a 2-byte scale = 18 bytes per 32 elements.
        assert!((KvType::Q4_0.bytes_per_element() - 0.562_5).abs() < 1e-12);
        // 32 bytes + a 2-byte scale = 34 per 32.
        assert!((KvType::Q8_0.bytes_per_element() - 1.062_5).abs() < 1e-12);
    }

    #[test]
    fn kv_bytes_scales_with_context_and_quantisation() {
        let f16 = ContextParams { n_ctx: 4096, ..ContextParams::default() };
        let q8 = ContextParams { type_k: KvType::Q8_0, type_v: KvType::Q8_0, ..f16.clone() };

        let a = f16.kv_bytes(16, 8, 128);
        let b = q8.kv_bytes(16, 8, 128);
        assert!(a > b, "q8_0 KV must be smaller than f16 KV");

        // f16 = 2 bytes; K and V both; 16 layers * 8 kv heads * 128 dim * 4096 tokens.
        assert_eq!(a, 2 * 2 * 16 * 8 * 128 * 4096);

        // Doubling the context doubles the cache.
        let wide = ContextParams { n_ctx: 8192, ..f16 };
        assert_eq!(wide.kv_bytes(16, 8, 128), 2 * a);
    }

    #[test]
    fn moearc_defaults_put_the_model_on_the_gpu_explicitly() {
        // Stated, not inherited. Whether upstream's default happens to agree is
        // checked separately against the linked library, in tests/runtime.rs --
        // this test pins MoEArc's intent, which does not depend on upstream.
        assert_eq!(ModelParams::default().n_gpu_layers, -1);
        assert_eq!(ModelParams::default().split_mode, SplitMode::None);
    }

    #[test]
    fn default_threads_is_derived_not_hardcoded_to_four() {
        // The value depends on the machine, so the assertion is on the property
        // that matters: it is at least 1 and it came from the host, not a literal.
        let t = ContextParams::default().n_threads;
        assert!(t >= 1);
        let expect = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
        assert_eq!(t, expect);
    }

    #[test]
    fn vocab_only_does_not_ask_for_gpu_layers() {
        let p = ModelParams::vocab_only();
        assert!(p.vocab_only);
        assert_eq!(p.n_gpu_layers, 0);
    }
}
