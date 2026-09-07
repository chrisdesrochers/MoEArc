//! Hand-written declarations for the MoEArc shim's C ABI.
//!
//! Every struct here mirrors one in `shim/moearc_llama_shim.cpp`, a file in this
//! repository. That is the whole point of the shim: these layouts are ours, they
//! are small, they are plain scalars, and they do not change when llama.cpp
//! reshuffles `llama_context_params`. Nothing in this file mirrors an upstream
//! struct.
//!
//! Bindings are hand-written rather than generated, matching the convention
//! established in `crates/moearc-kernels/src/ffi.rs`.

use std::ffi::c_char;

/// Opaque `llama_model *`.
#[repr(C)]
pub struct LlamaModel {
    _private: [u8; 0],
}

/// Opaque `llama_context *`.
#[repr(C)]
pub struct LlamaContext {
    _private: [u8; 0],
}

/// Opaque `llama_sampler *`.
#[repr(C)]
pub struct LlamaSampler {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MlaModelParams {
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub n_cpu_moe: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub use_extra_bufts: u8,
    pub no_host: u8,
    pub check_tensors: u8,
    pub vocab_only: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MlaCtxParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub flash_attn_type: i32,
    pub type_k: i32,
    pub type_v: i32,
    pub offload_kqv: u8,
    pub op_offload: u8,
    pub swa_full: u8,
    pub kv_unified: u8,
    pub no_perf: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MlaPerf {
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
    pub n_reused: i32,
}

unsafe extern "C" {
    pub fn mla_backend_init(log_to_stderr: i32, min_level: i32);
    pub fn mla_backend_free();
    pub fn mla_last_error(buf: *mut c_char, buf_len: usize) -> i32;
    pub fn mla_clear_error();
    pub fn mla_system_info() -> *const c_char;

    pub fn mla_model_default_params(out: *mut MlaModelParams);
    pub fn mla_model_load(path: *const c_char, p: *const MlaModelParams) -> *mut LlamaModel;
    pub fn mla_model_free(m: *mut LlamaModel);

    pub fn mla_model_n_layer(m: *const LlamaModel) -> i32;
    pub fn mla_model_n_embd(m: *const LlamaModel) -> i32;
    pub fn mla_model_n_head_kv(m: *const LlamaModel) -> i32;
    pub fn mla_model_n_ctx_train(m: *const LlamaModel) -> i32;
    pub fn mla_model_size(m: *const LlamaModel) -> u64;
    pub fn mla_model_n_params(m: *const LlamaModel) -> u64;
    pub fn mla_model_desc(m: *const LlamaModel, buf: *mut c_char, buf_len: usize) -> i32;
    pub fn mla_model_meta_val_str(
        m: *const LlamaModel,
        key: *const c_char,
        buf: *mut c_char,
        buf_len: usize,
    ) -> i32;

    pub fn mla_ctx_default_params(out: *mut MlaCtxParams);
    pub fn mla_ctx_create(m: *mut LlamaModel, p: *const MlaCtxParams) -> *mut LlamaContext;
    pub fn mla_ctx_free(c: *mut LlamaContext);
    pub fn mla_ctx_n_ctx(c: *const LlamaContext) -> u32;
    pub fn mla_ctx_set_n_threads(c: *mut LlamaContext, n_threads: i32, n_threads_batch: i32);
    pub fn mla_ctx_perf(c: *const LlamaContext, out: *mut MlaPerf);
    pub fn mla_ctx_perf_reset(c: *mut LlamaContext);
    pub fn mla_memory_clear(c: *mut LlamaContext, data: i32);

    pub fn mla_tokenize(
        m: *const LlamaModel,
        text: *const c_char,
        text_len: i32,
        out_tokens: *mut i32,
        n_max: i32,
        add_special: i32,
        parse_special: i32,
    ) -> i32;
    pub fn mla_token_to_piece(
        m: *const LlamaModel,
        token: i32,
        buf: *mut c_char,
        buf_len: i32,
        special: i32,
    ) -> i32;
    pub fn mla_vocab_n_tokens(m: *const LlamaModel) -> i32;
    pub fn mla_vocab_is_eog(m: *const LlamaModel, token: i32) -> i32;
    pub fn mla_vocab_bos(m: *const LlamaModel) -> i32;

    pub fn mla_decode(c: *mut LlamaContext, tokens: *const i32, n_tokens: i32) -> i32;

    pub fn mla_sampler_greedy() -> *mut LlamaSampler;
    pub fn mla_sampler_dist(seed: u32, top_k: i32, top_p: f32, temp: f32) -> *mut LlamaSampler;
    pub fn mla_sampler_sample(s: *mut LlamaSampler, c: *mut LlamaContext, idx: i32) -> i32;
    pub fn mla_sampler_accept(s: *mut LlamaSampler, token: i32);
    pub fn mla_sampler_free(s: *mut LlamaSampler);
}
