#!/usr/bin/env bash
# MoEArc tuning harness -- enforces bench/PROTOCOL.md.
# NOTE: deliberately NOT `set -u` (oneAPI setvars.sh exits silently under it).
set -eo pipefail

LLAMA_BIN=/zfs/swift/projects/llama.cpp/build/bin/llama-bench
MODELS=/zfs/swift/models
OUT=${OUT:-/zfs/swift/projects/MoEArc/bench/results/tuning}
mkdir -p "$OUT"

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ONEAPI_DEVICE_SELECTOR=level_zero:0

LOAD_MAX=${LOAD_MAX:-2.5}
BUSY_MAX=${BUSY_MAX:-12}      # percent of the whole machine over a 2s window
SETTLE_MAX=${SETTLE_MAX:-300} # seconds to wait for the box to go quiet

cpu_busy_pct() {
  # honest instantaneous busy%, not ps's lifetime average (rule 3 / comm-truncation trap)
  read -r _ a b c d e f g h _ < /proc/stat
  local t1=$((a+b+c+d+e+f+g+h)) i1=$d
  sleep 2
  read -r _ a b c d e f g h _ < /proc/stat
  local t2=$((a+b+c+d+e+f+g+h)) i2=$d
  awk -v dt=$((t2-t1)) -v di=$((i2-i1)) 'BEGIN{ if(dt<=0){print 0}else{printf "%.1f", 100*(1-di/dt)} }'
}
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }

diskstats_read_sectors() {
  awk '$3 ~ /^(nvme[0-9]+n[0-9]+|sd[a-z]+)$/ {s+=$6} END{print s+0}' /proc/diskstats
}
arc_stat() { awk -v k="$1" '$1==k{print $3}' /proc/spl/kstat/zfs/arcstats; }

# Wait for the box to be quiet. loadavg decays slowly after our OWN run, so we
# WAIT rather than skip -- but we never measure above the threshold.
settle() {
  local waited=0 l1 busy
  while :; do
    l1=$(awk '{print $1}' /proc/loadavg)
    busy=$(cpu_busy_pct); waited=$((waited+2))
    if ! gt "$l1" "$LOAD_MAX" && ! gt "$busy" "$BUSY_MAX"; then
      echo "GUARD ok after ${waited}s: load1=$l1 busy=${busy}%"; return 0
    fi
    if [ $waited -ge $SETTLE_MAX ]; then
      echo "REFUSE after ${waited}s: load1=$l1 busy=${busy}% (limits $LOAD_MAX / ${BUSY_MAX}%)"; return 1
    fi
    sleep 8; waited=$((waited+8))
  done
}

# run <tag> <model-file> <args...>
run() {
  local tag=$1; shift
  local mdl=$1; shift
  local f="$OUT/$tag.txt"
  { echo "===== $tag  $(date -Is)"; echo "model: $mdl"; echo "args:  $*"; } >> "$f"
  if ! settle >> "$f" 2>&1; then echo "SKIPPED (guard)" >> "$f"; echo >> "$f"; return 0; fi
  free -m | head -2 >> "$f"
  local d0 h0 m0 t0
  d0=$(diskstats_read_sectors); h0=$(arc_stat hits); m0=$(arc_stat misses); t0=$SECONDS
  set +e
  "$LLAMA_BIN" -m "$MODELS/$mdl" -o csv "$@" >> "$f" 2>>"$f.err"
  local rc=$?
  set -e
  local d1 h1 m1
  d1=$(diskstats_read_sectors); h1=$(arc_stat hits); m1=$(arc_stat misses)
  {
    echo "exit: $rc  elapsed: $((SECONDS-t0))s"
    echo "post-run: load1=$(cut -d' ' -f1 /proc/loadavg) busy=$(cpu_busy_pct)%"
    echo "disk_read_MiB: $(( (d1-d0)*512/1048576 ))   arc_hits:+$((h1-h0)) arc_misses:+$((m1-m0))"
    if [ $rc -ne 0 ]; then echo "--- stderr tail ---"; tail -8 "$f.err"; fi
    echo
  } >> "$f" 2>&1
  return 0
}
