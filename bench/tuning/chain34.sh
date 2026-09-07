#!/usr/bin/env bash
cd /zfs/swift/projects/MoEArc
while pgrep -f "bash bench/tuning/d2.sh" >/dev/null; do sleep 10; done
bash bench/tuning/d3.sh
bash bench/tuning/d4.sh
echo CHAIN34-DONE
