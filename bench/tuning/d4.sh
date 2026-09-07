#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OSS120=gpt-oss-120b-MXFP4.gguf
# 36 blocks. Prior published config was -ncmoe 31; gpt-oss-20b showed MORE offload wins,
# so the range is extended UP to 36 (all experts host-side) as well as down to the OOM floor.
run oss120-ncmoe   "$OSS120" -ncmoe 36,36,34,33,32,31,30,29,28 -t 16 -p 0 -n 64 -r 3
run oss120-threads "$OSS120" -ncmoe 36 -t 16,8,12,16,20,24 -p 0 -n 64 -r 3
