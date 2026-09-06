#!/usr/bin/env bash
# Reproduce MoEArc's headline decode number on your own card, and say what it could not measure.
#
#   bench/reproduce.sh /path/to/Qwen3-30B-A3B-Q4_K_M.gguf
#
# `bench/README.md`: "A result without the box, the commit hash and the quant is not a result."
# So this prints the box, the commit, the model and the runtime before it prints a number, and
# it prints an explicit list of the things the README claims that this run does *not* touch.
# A number without provenance is not a result, and a benchmark that quietly measures less than
# the claim it is checking is worse than no benchmark.
#
# What it runs is the single configuration behind the headline: 2952 resident expert slots,
# host policy `frac:0.75`, against the `off` control at the same capacity. Both rows come from
# one invocation so they share a process, a cache and a thermal state.

set -euo pipefail

REPEATS=${REPEATS:-3}
NPREDICT=${NPREDICT:-128}
NCTX=${NCTX:-512}
SLOTS=${SLOTS:-2952}
POLICIES=${POLICIES:-off,frac:0.75}
# `def fibonacci(n):` + newline + four spaces, tokenised. bench/baselines/qwen3-30b-a3b.md.
PROMPT_IDS=${PROMPT_IDS:-"750 75698 1445 982 257"}

usage() {
    cat <<USAGE
usage: reproduce.sh MODEL.gguf [options]

  --repeats N     independent runs of the configuration (default 3; see the spread note)
  --quick         one run of 32 tokens -- checks the harness, not the number
  --out FILE      also write the transcript here

environment: REPEATS NPREDICT NCTX SLOTS POLICIES PROMPT_IDS MOEARC_BENCH MOEARC_REF_IDS
USAGE
}

model=""
out=""
while [ $# -gt 0 ]; do
    case $1 in
        --repeats) REPEATS=$2; shift 2 ;;
        --quick) REPEATS=1; NPREDICT=32; shift ;;
        --out) out=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "reproduce.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
        *) model=$1; shift ;;
    esac
done
[ -n "$model" ] || { usage >&2; exit 2; }
[ -f "$model" ] || { echo "reproduce.sh: $model does not exist" >&2; exit 1; }

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$here/.." && pwd -P)

# The bench binary: an installed bundle, this repo's build, or told to us.
bench=${MOEARC_BENCH:-}
if [ -z "$bench" ]; then
    for c in "$root/moearc-bench" \
             "${CARGO_TARGET_DIR:-$root/target}/release/examples/hybrid_sweep" \
             "$root/target/release/examples/hybrid_sweep"; do
        [ -x "$c" ] && { bench=$c; break; }
    done
fi
[ -n "$bench" ] || {
    echo "reproduce.sh: cannot find the bench binary." >&2
    echo "  installed bundle: it is ./moearc-bench beside this script's parent" >&2
    echo "  from source:      cargo build --release -p moearc-engine --features gpu \\" >&2
    echo "                      --example hybrid_sweep" >&2
    echo "  or set MOEARC_BENCH=/path/to/it" >&2
    exit 1
}

refs=${MOEARC_REF_IDS:-$here/references/qwen3-30b-a3b.fibonacci.ids}
gate="$refs"
gate_note="token ids checked against $(basename "$refs")"
if [ ! -f "$refs" ]; then
    gate="-"
    gate_note="🔴 NO REFERENCE IDS -- rows are checked against each other only, not against llama.cpp"
fi

emit() { if [ -n "$out" ]; then tee -a "$out"; else cat; fi; }
[ -n "$out" ] && : > "$out"

{
echo "==================== MoEArc reproduction run ===================="
echo
echo "-- when"
date -u +'%Y-%m-%dT%H:%M:%SZ (UTC)'
echo
echo "-- the box"
printf '%-14s %s\n' cpu "$(sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo | head -1)"
printf '%-14s %s\n' cores "$(nproc) online"
printf '%-14s %s\n' ram "$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%.1f GiB total, %.1f GiB available", t/1048576, a/1048576}' /proc/meminfo)"
printf '%-14s %s\n' kernel "$(uname -sr)"
printf '%-14s %s\n' distro "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo unknown)"
printf '%-14s %s\n' glibc "$(ldd --version 2>/dev/null | head -1)"
# 🔴 A contended box produces a low reading that looks like a result. This repo has already
# had to retract one number to GPU contention, and the contending process is not always
# yours. Load average cannot see a second process on the GPU, so it reports what it can and
# says plainly what it cannot.
load1=$(cut -d' ' -f1 /proc/loadavg)
printf '%-14s %s\n' load "$load1 (1-minute) against $(nproc) cores"
if awk -v l="$load1" -v n="$(nproc)" 'BEGIN { exit !(l > n / 4) }'; then
    echo "               *** THE BOX IS BUSY. This run is CONTENDED and is not citable. ***"
fi
echo "               GPU contention is invisible from here -- check intel_gpu_top separately."
echo
echo "-- the gpu (as MoEArc sees it, not as the box advertises it)"
moearc_bin=""
for c in "$root/moearc" "${CARGO_TARGET_DIR:-$root/target}/release/moearc" "$root/target/release/moearc"; do
    [ -x "$c" ] && { moearc_bin=$c; break; }
done
if [ -n "$moearc_bin" ] && command -v python3 >/dev/null 2>&1; then
    # %-formatting, not an f-string: an f-string needs quotes around the dict keys, and
    # those quotes have to survive both the shell and Python. This does not.
    "$moearc_bin" --json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit("  (device report unavailable -- moearc --json produced nothing parseable)")
devs = d.get("devices", [])
if not devs:
    sys.exit("  🔴 NO GPU DEVICE. Whatever follows is not a measurement of this card.")
for i, g in enumerate(devs):
    print("  [%d] %s  %s  %.2f GiB" % (i, g["name"], g["driver"], g["total_bytes"] / 2 ** 30))
' || echo "  (device report unavailable)"
else
    echo "  (no moearc binary beside this script; cannot report the device)"
fi
echo
echo "-- the build"
if [ -f "$root/share/moearc/BUILD-INFO.txt" ]; then
    sed 's/^/  /' "$root/share/moearc/BUILD-INFO.txt"
elif command -v git >/dev/null 2>&1 && git -C "$root" rev-parse HEAD >/dev/null 2>&1; then
    printf '  %-12s %s\n' commit "$(git -C "$root" rev-parse HEAD)"
    printf '  %-12s %s\n' dirty "$(git -C "$root" diff --quiet && echo no || echo YES)"
else
    echo "  (no build metadata -- this is not a citable run)"
fi
echo "  bench binary $bench"
echo
echo "-- the sycl runtime"
if [ -f "${MOEARC_RUNTIME_DIR:-$root/runtime}/PROVENANCE.txt" ]; then
    grep -E '^[a-z]' "${MOEARC_RUNTIME_DIR:-$root/runtime}/PROVENANCE.txt" | sed 's/^/  /'
elif [ -n "${ONEAPI_ROOT:-}" ]; then
    echo "  a locally installed oneAPI at $ONEAPI_ROOT (developer build, not the shipped path)"
else
    echo "  unknown -- neither a bundled runtime nor ONEAPI_ROOT is set"
fi
echo
echo "-- the model"
printf '  %-12s %s\n' path "$model"
printf '  %-12s %s\n' bytes "$(stat -c %s "$model") ($(du -h --apparent-size "$model" | cut -f1))"
if command -v sha256sum >/dev/null 2>&1; then
    printf '  %-12s %s\n' head-sha256 \
        "$(head -c 67108864 "$model" | sha256sum | cut -d' ' -f1)"
    echo "               ^ SHA-256 of the FIRST 64 MiB only, not the whole file. It identifies"
    echo "                 the quantisation and tensor layout cheaply; it is not an integrity check."
fi
echo
echo "-- the configuration"
printf '  %-12s %s\n' slots "$SLOTS"
printf '  %-12s %s\n' policies "$POLICIES"
printf '  %-12s %s\n' tokens "$NPREDICT greedy, n_ctx $NCTX"
printf '  %-12s %s\n' prompt "$PROMPT_IDS"
printf '  %-12s %s\n' repeats "$REPEATS"
printf '  %-12s %s\n' gate "$gate_note"
echo
echo "================================================================"

tmp=$(mktemp "${TMPDIR:-/tmp}/moearc-repro.XXXXXX")
trap 'rm -f "$tmp"' EXIT

for i in $(seq 1 "$REPEATS"); do
    echo
    echo "##### run $i of $REPEATS #####"
    # shellcheck disable=SC2086
    "$bench" "$model" "$NPREDICT" "$NCTX" "$SLOTS" "$POLICIES" "$gate" $PROMPT_IDS \
        | tee -a "$tmp"
done

echo
echo "==================== summary ===================="
python3 - "$tmp" <<'PY'
import re, sys, statistics
rows = {}
verdicts = set()
for line in open(sys.argv[1]):
    if not line.startswith("|"):
        continue
    f = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(f) < 12 or not f[0].isdigit():
        continue
    pol = f[1].strip("`")
    try:
        toks = float(f[3])
    except ValueError:
        continue
    rows.setdefault(pol, []).append((toks, f[5], f[6]))
    verdicts.add(f[11])
if not rows:
    sys.exit("no result rows were produced -- the run failed, see above")
print(f"{'policy':<12} {'n':>2} {'min':>7} {'median':>7} {'max':>7}   {'hit':>6} {'staged MiB':>10}")
for pol, vals in rows.items():
    t = sorted(v[0] for v in vals)
    print(f"{pol:<12} {len(t):>2} {t[0]:>7.2f} {statistics.median(t):>7.2f} {t[-1]:>7.2f}"
          f"   {vals[-1][1]:>6} {vals[-1][2]:>10}")
best = max((statistics.median(sorted(v[0] for v in vals)), p) for p, vals in rows.items())
ctrl = rows.get("off")
if ctrl:
    c = statistics.median(sorted(v[0] for v in ctrl))
    print(f"\nbest policy {best[1]}: {best[0]:.2f} tok/s median, "
          f"{100 * (best[0] / c - 1):+.0f}% against the `off` control at the same capacity")
print("\ntoken-id verdicts across all rows: " + ", ".join(sorted(verdicts)))
PY

cat <<'NOTES'

==================== what this run did NOT measure ====================

Stated so the number above is not mistaken for more than it is. Every item here is a claim
the project makes elsewhere, on evidence this script does not produce.

  - Prefill. There is none in the engine. llama.cpp's 3218 tok/s has no counterpart, and the
    figure above is decode only.
  - A llama.cpp baseline. The "90% of llama.cpp" comparison needs llama.cpp built and run on
    this same box at the swept `-ncmoe`; nothing here does that.
  - Perplexity. Correctness here is a token-id gate against a stored reference, not a
    wikitext-2 run.
  - Power, and therefore tok/s/W.
  - Kernel efficiency against the card's peak bandwidth.
  - Any residency other than the one configured, and any model other than the one you passed.
  - Whether the policy was *chosen*. It was not: the engine has no adaptive policy, and
    `frac:0.75` came from a sweep. See docs/roadmap.md.

⚠️  Run-to-run spread on the reference box is real and it is about ±10%. The headline row
reads 43.05, 43.16, 43.78 and 43.35 across four focused runs and 39.12 inside a long sweep.
One run is not a number; that is why the default is three and why the summary prints the
spread rather than an average.
NOTES
} 2>&1 | emit
