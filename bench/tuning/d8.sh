#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OSS120=gpt-oss-120b-MXFP4.gguf
Q30=Qwen3-30B-A3B-Q4_K_M.gguf
SCOUT=Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf
# Qwen3-30B loses 1.31x from d0 to d8192 -- is flash-attn on, and is it worth anything?
run qwen30-fa-d8192   "$Q30" -ncmoe 21 -t 16 -fa on,on,off -p 0 -n 64 -d 8192 -r 3
# is -t 16 a peak or a plateau? (20-core box: 8 P-cores + 12 E-cores)
run oss120-t-fine  "$OSS120" -ncmoe 36 -t 16,14,15,16,17,18 -p 0 -n 64 -r 3
run scout-floor-d8192 "$SCOUT" -ncmoe 48,48,47,46,45,44 -t 16 -p 0 -n 64 -d 8192 -r 2
run scout-depth8192   "$SCOUT" -ncmoe 46 -t 16 -p 0 -n 64 -d 8192 -r 2
echo D8-DONE
