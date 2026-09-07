#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OSS120=gpt-oss-120b-MXFP4.gguf
OSS20=gpt-oss-20b-MXFP4.gguf
Q30=Qwen3-30B-A3B-Q4_K_M.gguf
Q35=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
CODER=Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
SCOUT=Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf

# 1. does the "offload everything" conclusion for gpt-oss survive at depth?
run oss20-ncmoe-d8192 "$OSS20"  -ncmoe 24,24,16,8,0 -t 16 -p 0 -n 64 -d 8192 -r 3
# 2. depth curves at the chosen config
run oss20-depth   "$OSS20"  -ncmoe 24 -t 16 -p 0 -n 64 -d 0,0,512,2048,8192 -r 3
run oss120-depth  "$OSS120" -ncmoe 36 -t 16 -p 0 -n 64 -d 0,0,512,2048,8192 -r 3
# 3. Qwen3-30B: threads at the floor, then the floor AT depth (KV grows -> floor moves)
run qwen30-threads     "$Q30" -ncmoe 18 -t 16,8,12,16,20,24 -p 0 -n 64 -r 3
run qwen30-floor-d8192 "$Q30" -ncmoe 26,26,24,22,21,20,19,18 -t 16 -p 0 -n 64 -d 8192 -r 3
# 4. Qwen3.6-35B: the near-floor rows were noisy; repeat them
run qwen35-repeat "$Q35" -ncmoe 28,28,26,24,22 -t 16 -p 0 -n 64 -r 5
# 5. does the Qwen3-30B profile transfer to the Coder variant? (spot check, not a transfer claim)
run coder30-spot  "$CODER" -ncmoe 22,22,20,19,18,17 -t 16 -p 0 -n 64 -r 3
# 6. Scout was noisy (streaming a 45.6 GiB model); repeat with more reps
run scout-repeat  "$SCOUT" -ncmoe 46,46,44 -t 16 -p 0 -n 64 -r 5
