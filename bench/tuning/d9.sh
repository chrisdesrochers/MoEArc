#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OSS120=gpt-oss-120b-MXFP4.gguf
OSS20=gpt-oss-20b-MXFP4.gguf
Q30=Qwen3-30B-A3B-Q4_K_M.gguf
# oss120-t-fine (round 8) is DISCARDED: 86.6 GiB of disk reads, sd up to 36%, every
# value far below the established warm 29.7 -- it measured the storage (PROTOCOL 4).
# A 45.6 GiB Scout load immediately before it had evicted the flagship from page cache.
# Prime the cache first, then repeat. These three MUST stay adjacent -- no other model.
run oss120-primer  "$OSS120" -ncmoe 36 -t 16 -p 0 -n 32 -r 2
run oss120-t-fine2 "$OSS120" -ncmoe 36 -t 14,14,15,16,17,18,20 -p 0 -n 64 -r 3
run oss120-deep    "$OSS120" -ncmoe 36 -t 16 -p 0 -n 64 -d 8192,32768 -r 2
# long-context reachability at the recommended configs
run oss20-deep     "$OSS20"  -ncmoe 24 -t 16 -p 0 -n 64 -d 8192,32768,65536 -r 2
run qwen30-floor-d32768 "$Q30" -ncmoe 34,32,30,28,26 -t 16 -p 0 -n 8 -d 32768 -r 1
echo D9-DONE
