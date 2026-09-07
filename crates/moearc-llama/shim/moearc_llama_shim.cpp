// MoEArc <-> llama.cpp shim.
//
// This file exists so that Rust never has to reproduce the memory layout of
// `llama_model_params` or `llama_context_params`.
//
// Both are passed BY VALUE across the ABI, both are large, both mix enums,
// pointers, floats and a trailing bool block, and both change shape between
// llama.cpp releases -- `llama_context_params` alone carries 30+ members and
// upstream comments in it warn twice about "misalignment during copy-by-value".
// A hand-transcribed Rust `#[repr(C)]` mirror of those structs is correct only
// against the exact commit it was written for, and when it drifts it does not
// fail to compile: it silently passes `n_threads` in the slot the callee reads
// as `n_ubatch`. That is the single most dangerous failure mode available here,
// because the result is a working program that is quietly misconfigured -- and
// this project has already had three results invalidated by measuring something
// other than what it thought it was measuring.
//
// So: llama.h defines those structs, this translation unit is the only thing
// that touches them, and Rust sees a small flat POD surface that MoEArc owns.
// When the llama.cpp pin moves, this file recompiles against the new header and
// either keeps working or fails loudly at build time. Neither outcome is silent.
//
// This mirrors the existing house convention in `crates/moearc-kernels`
// (hand-written `ffi.rs` against our own C++ TU, deliberately not bindgen).
//
// NOTE: this file compiles with a plain g++. It includes no SYCL header. The
// SYCL dependency lives entirely inside libggml-sycl.so, which llama.cpp built
// with icpx and which records its own Intel runtime libraries in DT_NEEDED.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Diagnostics
//
// llama.cpp reports load failures by returning NULL and writing the reason to
// its log callback. Without capturing the log, a failed load in Rust is an
// unexplained `None`. We keep a bounded tail of the most recent log lines so a
// failure can carry its own cause.
// ---------------------------------------------------------------------------

namespace {

std::mutex  g_log_mu;
std::string g_log_tail;
int         g_log_min_level = GGML_LOG_LEVEL_ERROR;
bool        g_log_to_stderr = false;

constexpr size_t LOG_TAIL_MAX = 8192;

void mla_log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (text == nullptr) {
        return;
    }
    if (g_log_to_stderr && level >= g_log_min_level) {
        fputs(text, stderr);
    }
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (level >= GGML_LOG_LEVEL_WARN) {
        g_log_tail.append(text);
        if (g_log_tail.size() > LOG_TAIL_MAX) {
            g_log_tail.erase(0, g_log_tail.size() - LOG_TAIL_MAX);
        }
    }
}

// Copy a std::string into a caller-owned buffer, always NUL-terminated.
// Returns the number of bytes that WOULD have been written excluding the NUL,
// so the caller can detect truncation.
int32_t copy_out(const std::string & s, char * buf, size_t buf_len) {
    if (buf != nullptr && buf_len > 0) {
        const size_t n = s.size() < buf_len - 1 ? s.size() : buf_len - 1;
        memcpy(buf, s.data(), n);
        buf[n] = '\0';
    }
    return static_cast<int32_t>(s.size());
}

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Flat parameter structs -- MoEArc owns these layouts, not llama.cpp.
//
// Every field here is one the tuning layer is expected to set. Fields of the
// underlying llama.cpp structs that MoEArc has no reason to touch are left at
// llama.cpp's own defaults and are deliberately NOT exposed; adding one later
// is an additive change to a struct we control.
// ---------------------------------------------------------------------------

struct mla_model_params {
    int32_t n_gpu_layers;   // -ngl. negative = all layers on GPU.
    int32_t main_gpu;       // device index used when split_mode == NONE
    int32_t n_cpu_moe;      // -ncmoe. expert tensors of the first N blocks stay on CPU.
    int32_t split_mode;     // enum llama_split_mode
    int32_t load_mode;      // enum llama_load_mode (mmap / mlock / direct-io)
    uint8_t use_extra_bufts;
    uint8_t no_host;
    uint8_t check_tensors;
    uint8_t vocab_only;
};

struct mla_ctx_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;
    int32_t  n_threads_batch;
    int32_t  flash_attn_type; // enum llama_flash_attn_type: -1 auto, 0 off, 1 on
    int32_t  type_k;          // enum ggml_type for the K cache
    int32_t  type_v;          // enum ggml_type for the V cache
    uint8_t  offload_kqv;
    uint8_t  op_offload;
    uint8_t  swa_full;
    uint8_t  kv_unified;
    uint8_t  no_perf;
};

struct mla_perf {
    double  t_load_ms;
    double  t_p_eval_ms;
    double  t_eval_ms;
    int32_t n_p_eval;
    int32_t n_eval;
    int32_t n_reused;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void mla_backend_init(int32_t log_to_stderr, int32_t min_level) {
    g_log_to_stderr = log_to_stderr != 0;
    g_log_min_level = min_level;
    llama_log_set(mla_log_cb, nullptr);
    llama_backend_init();
}

void mla_backend_free(void) {
    llama_backend_free();
}

int32_t mla_last_error(char * buf, size_t buf_len) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    return copy_out(g_log_tail, buf, buf_len);
}

void mla_clear_error(void) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log_tail.clear();
}

const char * mla_system_info(void) {
    return llama_print_system_info();
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

void mla_model_default_params(struct mla_model_params * out) {
    const llama_model_params d = llama_model_default_params();
    out->n_gpu_layers    = d.n_gpu_layers;
    out->main_gpu        = d.main_gpu;
    out->n_cpu_moe       = 0;
    out->split_mode      = static_cast<int32_t>(d.split_mode);
    out->load_mode       = static_cast<int32_t>(d.load_mode);
    out->use_extra_bufts = d.use_extra_bufts ? 1 : 0;
    out->no_host         = d.no_host ? 1 : 0;
    out->check_tensors   = d.check_tensors ? 1 : 0;
    out->vocab_only      = d.vocab_only ? 1 : 0;
}

// The `-ncmoe N` translation.
//
// This is NOT a scalar knob on llama_model_params -- there is no `n_cpu_moe`
// field anywhere in llama.h. `-ncmoe N` expands to N *tensor buffer overrides*,
// one per block, each a (regex, buffer-type) pair pinning that block's expert
// tensors to the CPU buffer type. Reproduced here from llama.cpp's
// common/common.h (LLM_FFN_EXPS_REGEX + llm_ffn_block_regex) and
// common/arg.cpp (llm_add_n_cpu_ffn_overrides), so that MoEArc gets byte-identical
// behaviour to the CLI flag without depending on libllama-common.
//
// That library exists and is shared (5.6 MB), so linking it would in fact work.
// The reason not to is that common/ is llama.cpp's CLI and example support code,
// not its public API: common.h carries no LLAMA_API and no extern "C", its
// signatures are C++ (std::string, std::vector), llm_add_n_cpu_ffn_overrides is an
// `inline` function over a function-local `static std::list`, and upstream makes no
// stability promise about any of it. Depending on it would commit MoEArc to C++ ABI
// compatibility with llama.cpp's build in exchange for ten lines of string
// formatting.
//
// 🔴 Lifetime: llama_model_params holds BORROWED `const char *` pointers. The
// pattern strings must outlive llama_model_load_from_file(). They are locals of
// this function, which is fine because the load completes before we return --
// but a std::vector<std::string> that reallocates would invalidate the pointers
// already stored in the override vector, so the string storage is reserved up
// front and never grown afterwards.
struct llama_model * mla_model_load(const char * path, const struct mla_model_params * p) {
    llama_model_params mp = llama_model_default_params();

    mp.n_gpu_layers    = p->n_gpu_layers;
    mp.main_gpu        = p->main_gpu;
    mp.split_mode      = static_cast<enum llama_split_mode>(p->split_mode);
    mp.load_mode       = static_cast<enum llama_load_mode>(p->load_mode);
    mp.use_extra_bufts = p->use_extra_bufts != 0;
    mp.no_host         = p->no_host != 0;
    mp.check_tensors   = p->check_tensors != 0;
    mp.vocab_only      = p->vocab_only != 0;

    // Upstream: common/common.h
    //   LLM_FFN_EXPS_REGEX  = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
    //   llm_ffn_block_regex = "blk\\.%d" + LLM_FFN_EXPS_REGEX
    static const char * const EXPS_RE = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

    std::vector<std::string>                    patterns;
    std::vector<llama_model_tensor_buft_override> overrides;

    const int n_cpu_moe = p->n_cpu_moe > 0 ? p->n_cpu_moe : 0;
    if (n_cpu_moe > 0) {
        patterns.reserve(static_cast<size_t>(n_cpu_moe)); // no reallocation after this
        overrides.reserve(static_cast<size_t>(n_cpu_moe) + 1);
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        for (int i = 0; i < n_cpu_moe; ++i) {
            patterns.push_back("blk\\." + std::to_string(i) + EXPS_RE);
            overrides.push_back({ patterns.back().c_str(), cpu_buft });
        }
        overrides.push_back({ nullptr, nullptr }); // NULL-terminated, per llama.h
        mp.tensor_buft_overrides = overrides.data();
    }

    return llama_model_load_from_file(path, mp);
}

void mla_model_free(struct llama_model * m) {
    llama_model_free(m);
}

int32_t  mla_model_n_layer     (const struct llama_model * m) { return llama_model_n_layer(m); }
int32_t  mla_model_n_embd      (const struct llama_model * m) { return llama_model_n_embd(m); }
int32_t  mla_model_n_head_kv   (const struct llama_model * m) { return llama_model_n_head_kv(m); }
int32_t  mla_model_n_ctx_train (const struct llama_model * m) { return llama_model_n_ctx_train(m); }
uint64_t mla_model_size        (const struct llama_model * m) { return llama_model_size(m); }
uint64_t mla_model_n_params    (const struct llama_model * m) { return llama_model_n_params(m); }

int32_t mla_model_desc(const struct llama_model * m, char * buf, size_t buf_len) {
    return llama_model_desc(m, buf, buf_len);
}

// Reads a GGUF metadata key, e.g. "<arch>.expert_count" / "<arch>.expert_used_count",
// which the tuning layer needs to decide an -ncmoe split. Returns <0 if absent.
int32_t mla_model_meta_val_str(const struct llama_model * m, const char * key, char * buf, size_t buf_len) {
    return llama_model_meta_val_str(m, key, buf, buf_len);
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

void mla_ctx_default_params(struct mla_ctx_params * out) {
    const llama_context_params d = llama_context_default_params();
    out->n_ctx           = d.n_ctx;
    out->n_batch         = d.n_batch;
    out->n_ubatch        = d.n_ubatch;
    out->n_seq_max       = d.n_seq_max;
    out->n_threads       = d.n_threads;
    out->n_threads_batch = d.n_threads_batch;
    out->flash_attn_type = static_cast<int32_t>(d.flash_attn_type);
    out->type_k          = static_cast<int32_t>(d.type_k);
    out->type_v          = static_cast<int32_t>(d.type_v);
    out->offload_kqv     = d.offload_kqv ? 1 : 0;
    out->op_offload      = d.op_offload ? 1 : 0;
    out->swa_full        = d.swa_full ? 1 : 0;
    out->kv_unified      = d.kv_unified ? 1 : 0;
    out->no_perf         = d.no_perf ? 1 : 0;
}

struct llama_context * mla_ctx_create(struct llama_model * m, const struct mla_ctx_params * p) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = p->n_ctx;
    cp.n_batch         = p->n_batch;
    cp.n_ubatch        = p->n_ubatch;
    cp.n_seq_max       = p->n_seq_max;
    cp.n_threads       = p->n_threads;
    cp.n_threads_batch = p->n_threads_batch;
    cp.flash_attn_type = static_cast<enum llama_flash_attn_type>(p->flash_attn_type);
    cp.type_k          = static_cast<enum ggml_type>(p->type_k);
    cp.type_v          = static_cast<enum ggml_type>(p->type_v);
    cp.offload_kqv     = p->offload_kqv != 0;
    cp.op_offload      = p->op_offload != 0;
    cp.swa_full        = p->swa_full != 0;
    cp.kv_unified      = p->kv_unified != 0;
    cp.no_perf         = p->no_perf != 0;
    return llama_init_from_model(m, cp);
}

void     mla_ctx_free (struct llama_context * c) { llama_free(c); }
uint32_t mla_ctx_n_ctx(const struct llama_context * c) { return llama_n_ctx(c); }

void mla_ctx_set_n_threads(struct llama_context * c, int32_t n_threads, int32_t n_threads_batch) {
    llama_set_n_threads(c, n_threads, n_threads_batch);
}

void mla_ctx_perf(const struct llama_context * c, struct mla_perf * out) {
    const llama_perf_context_data d = llama_perf_context(c);
    out->t_load_ms   = d.t_load_ms;
    out->t_p_eval_ms = d.t_p_eval_ms;
    out->t_eval_ms   = d.t_eval_ms;
    out->n_p_eval    = d.n_p_eval;
    out->n_eval      = d.n_eval;
    out->n_reused    = d.n_reused;
}

void mla_ctx_perf_reset(struct llama_context * c) { llama_perf_context_reset(c); }

void mla_memory_clear(struct llama_context * c, int32_t data) {
    llama_memory_clear(llama_get_memory(c), data != 0);
}

// ---------------------------------------------------------------------------
// Vocab / tokenizer
// ---------------------------------------------------------------------------

int32_t mla_tokenize(const struct llama_model * m, const char * text, int32_t text_len,
                     int32_t * out_tokens, int32_t n_max, int32_t add_special, int32_t parse_special) {
    const llama_vocab * v = llama_model_get_vocab(m);
    static_assert(sizeof(llama_token) == sizeof(int32_t), "llama_token is not int32_t");
    return llama_tokenize(v, text, text_len, reinterpret_cast<llama_token *>(out_tokens), n_max,
                          add_special != 0, parse_special != 0);
}

int32_t mla_token_to_piece(const struct llama_model * m, int32_t token, char * buf, int32_t buf_len,
                           int32_t special) {
    const llama_vocab * v = llama_model_get_vocab(m);
    return llama_token_to_piece(v, token, buf, buf_len, 0, special != 0);
}

int32_t mla_vocab_n_tokens(const struct llama_model * m) {
    return llama_vocab_n_tokens(llama_model_get_vocab(m));
}

int32_t mla_vocab_is_eog(const struct llama_model * m, int32_t token) {
    return llama_vocab_is_eog(llama_model_get_vocab(m), token) ? 1 : 0;
}

int32_t mla_vocab_bos(const struct llama_model * m) {
    return llama_vocab_bos(llama_model_get_vocab(m));
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

// Positions are tracked automatically by llama_decode when batch.pos is NULL,
// which is what llama_batch_get_one produces. Sequence id is fixed to 0.
int32_t mla_decode(struct llama_context * c, const int32_t * tokens, int32_t n_tokens) {
    llama_batch b = llama_batch_get_one(const_cast<llama_token *>(
                                            reinterpret_cast<const llama_token *>(tokens)),
                                        n_tokens);
    return llama_decode(c, b);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

struct llama_sampler * mla_sampler_greedy(void) {
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
}

struct llama_sampler * mla_sampler_dist(uint32_t seed, int32_t top_k, float top_p, float temp) {
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(sp);
    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

int32_t mla_sampler_sample(struct llama_sampler * s, struct llama_context * c, int32_t idx) {
    return llama_sampler_sample(s, c, idx);
}

void mla_sampler_accept(struct llama_sampler * s, int32_t token) {
    llama_sampler_accept(s, token);
}

void mla_sampler_free(struct llama_sampler * s) {
    llama_sampler_free(s);
}

} // extern "C"
