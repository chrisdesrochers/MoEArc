#!/usr/bin/env bash
# VRAM headroom pass. NOT a timed measurement -- one rep, verbose, we only want
# the allocator's own report of what it placed on SYCL0.
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
hr() { # hr <tag> <model> <args...>
  local tag=$1; shift; local mdl=$1; shift
  local f="$OUT/headroom-$tag.txt"
  echo "===== headroom $tag  args: $*" >> "$f"
  "$LLAMA_BIN" -m "$MODELS/$mdl" -v -r 1 -p 0 -n 8 "$@" 2>&1 \
    | grep -aE "buffer size|KV self|kv_unified|n_ctx|SYCL0|graph splits|offloaded" >> "$f"
  echo >> "$f"
}
hr olmoe   olmoe-1b-7b-0924-instruct-q4_k_m.gguf      -ncmoe 0  -t 16
hr oss20   gpt-oss-20b-MXFP4.gguf                     -ncmoe 24 -t 16 -d 8192
hr qwen30  Qwen3-30B-A3B-Q4_K_M.gguf                  -ncmoe 21 -t 16 -d 8192
hr coder30 Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf   -ncmoe 21 -t 16 -d 8192
hr qwen35  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf             -ncmoe 22 -t 16 -d 8192
hr scout   Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf -ncmoe 46 -t 16 -d 8192
hr oss120  gpt-oss-120b-MXFP4.gguf                    -ncmoe 36 -t 16 -d 8192
hr oss120_31 gpt-oss-120b-MXFP4.gguf                  -ncmoe 31 -t 16 -d 8192
echo D7-DONE
