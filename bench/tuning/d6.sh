#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OSS120=gpt-oss-120b-MXFP4.gguf
OSS20=gpt-oss-20b-MXFP4.gguf
Q30=Qwen3-30B-A3B-Q4_K_M.gguf
Q35=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
CODER=Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
SCOUT=Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf

# --- the ncmoe floor MOVES with context, because KV competes for the same VRAM.
run oss20-floor-d8192  "$OSS20"  -ncmoe 8,8,6,4,2,0 -t 16 -p 0 -n 64 -d 8192 -r 3
run qwen35-floor-d8192 "$Q35"    -ncmoe 28,28,26,24,23,22,21,20 -t 16 -p 0 -n 64 -d 8192 -r 3
# --- KV cache quantisation: q8_0 halves KV. Does it buy back offload blocks?
run qwen30-kvq8-d8192  "$Q30" -ncmoe 21,21,20,19,18,17,16 -t 16 -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d 8192 -r 3
# --- depth curves at the shipping config
run qwen30-depth "$Q30"  -ncmoe 21 -t 16 -p 0 -n 64 -d 0,0,512,2048,8192 -r 3
run qwen35-depth "$Q35"  -ncmoe 26 -t 16 -p 0 -n 64 -d 0,0,512,2048,8192 -r 3
run qwen35-threads "$Q35" -ncmoe 22 -t 16,8,12,16,20 -p 0 -n 64 -r 3
# --- flagship: floor at depth + a clean warm d0 (the cold d0 cell was discarded earlier)
run oss120-floor-d8192 "$OSS120" -ncmoe 34,34,33,32,31,30 -t 16 -p 0 -n 64 -d 8192 -r 3
run oss120-warm-d0     "$OSS120" -ncmoe 36,36,34,31 -t 16 -p 0 -n 64 -r 3
# --- Scout: threads (44 of 48 blocks host-side, so -t should matter a lot) + depth
run scout-threads "$SCOUT" -ncmoe 46 -t 16,8,12,16,20 -p 0 -n 64 -r 3
run scout-depth   "$SCOUT" -ncmoe 46 -t 16 -p 0 -n 64 -d 0,0,512,2048 -r 2
run coder30-depth "$CODER" -ncmoe 21 -t 16 -p 0 -n 64 -d 0,0,512,2048,8192 -r 3
