#!/usr/bin/env bash
source /zfs/swift/projects/MoEArc/bench/tuning/harness.sh
OLMOE=olmoe-1b-7b-0924-instruct-q4_k_m.gguf
OSS20=gpt-oss-20b-MXFP4.gguf
run oss20-ncmoe-hi  "$OSS20" -ncmoe 24,24,22,20,18,16,14 -t 16 -p 0 -n 64 -r 3
run oss20-threads-a "$OSS20" -ncmoe 24 -t 16,2,4,8,12,16,20,24 -p 0 -n 64 -r 3
run oss20-threads-b "$OSS20" -ncmoe 24 -t 16,2,4,8,12,16,20,24 -p 0 -n 64 -r 3
run oss20-endpoints "$OSS20" -ncmoe 24,0,24,0 -t 16 -p 0 -n 64 -r 3
run olmoe-thr-off   "$OLMOE" -ncmoe 16 -t 16,2,4,8,12,16,20,24 -p 0 -n 64 -r 3
run olmoe-depth     "$OLMOE" -ncmoe 0 -t 16 -p 0 -n 64 -d 0,0,512,1024,2048,3072 -r 3
