#!/usr/bin/env bash
#
# Capture an expert-routing trace from llama.cpp.
#
#   MODEL=/path/to/model.gguf LLAMACPP=/path/to/llama.cpp ./capture.sh <prompt-name> <n-decode>
#
# Requires the llama.cpp tree to have `llama.cpp-eval-callback-moearc.patch` applied and
# `llama-eval-callback` rebuilt. Reads prompts/<prompt-name>.txt; writes <prompt-name>.{prefill,
# decode}.ndjson and <prompt-name>.generated.txt beside this script.
#
# 🔴 -ngl 0 is deliberate. This is a functional capture, not a benchmark: routing is identical on
# any backend, and running on CPU leaves the GPU free (and keeps a concurrent GPU measurement
# uncontaminated).
set -eo pipefail

: "${MODEL:?set MODEL to the .gguf}"
: "${LLAMACPP:?set LLAMACPP to the llama.cpp checkout}"
NAME=${1:?prompt name}
NPRED=${2:-1024}
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT_FILE="$HERE/prompts/$NAME.txt"

COMMIT=$(git -C "$LLAMACPP" rev-parse HEAD)
DIRTY=$(git -C "$LLAMACPP" status --porcelain -- examples/eval-callback/eval-callback.cpp | wc -l)
PROMPT_JSON=$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()))' "$PROMPT_FILE")

META="\"model_file\":\"$(basename "$MODEL")\",\"quantisation\":\"Q4_K_M\",\"n_expert\":256,\"llama_cpp_commit\":\"$COMMIT\",\"llama_cpp_patched\":$([ "$DIRTY" -gt 0 ] && echo true || echo false),\"captured_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"prompt_name\":\"$NAME\",\"prompt\":$PROMPT_JSON,\"backend\":\"CPU (-ngl 0)\",\"sampling\":\"temp 0.7, top-k 20, top-p 0.8, seed 20260904\""

MOEARC_TRACE_OUT="$HERE/$NAME" \
MOEARC_TRACE_META="$META" \
MOEARC_TEXT_OUT="$HERE/$NAME.generated.txt" \
"$LLAMACPP/build/bin/llama-eval-callback" \
  -m "$MODEL" -ngl 0 -f "$PROMPT_FILE" -n "$NPRED" \
  -t 20 -c 4096 -b 2048 -ub 512 --temp 0.7 --top-k 20 --top-p 0.8 --seed 20260904
