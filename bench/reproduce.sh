#!/usr/bin/env bash
# Reproduce MoEArc's published result on your own machine, and say plainly what it cannot.
#
#   bench/reproduce.sh                       # the result, from the committed traces
#   bench/reproduce.sh --model M.gguf        # ...with that model's byte columns attached
#
# 🔴 WHAT THIS REPRODUCES, AND WHY IT CHANGED.
#
# This script used to run `moearc-bench` — the retired SYCL engine's `hybrid_sweep` example —
# and reproduce a decode-throughput sweep over expert-residency slots and host-offload
# policies. That binary is no longer built and is no longer in the release tarball; it existed
# only to exercise an engine the project has retired, and putting it back would mean shipping
# that engine. So the honest statement is the one this script now makes:
#
#   **The published absolute throughput numbers cannot be reproduced from the release.**
#
# That is not a workaround, it is the accurate position. `bench/PROTOCOL.md` §0 says so in its
# own words, and said so before the pivot:
#
#   "Absolute throughput does not reproduce and we should stop implying it does. [...] What
#    must reproduce on any Arc box is the **shape**. [...] Report shape as the result and
#    absolutes as an artefact of one machine."
#
# The shape is what this script reproduces, and it reproduces it *exactly* — it is a
# deterministic replay of the routing traces committed under `bench/traces`, it reads no clock,
# it touches no GPU, and it needs no model. If your run disagrees with ours in any digit, that
# is a real disagreement worth an issue.
#
# ⬜ The timed half (`moearc bench --absolutes`) is not gone, it is unbuilt: it still reaches
# the card through the retired engine's `gpu` feature, and a release binary is deliberately
# built without it. On such a binary `--absolutes` refuses with `this binary has no GPU backend
# compiled in` and exit code 3 — a refusal, not a crash, and correct. When the timed path is
# rewired onto the llama.cpp runtime, this script should grow an `--absolutes` mode; until
# then it does not pretend to have one.

set -euo pipefail

model=""
out=""
extra=()

usage() {
    cat <<USAGE
usage: reproduce.sh [--model MODEL.gguf] [--out FILE] [-- ARGS...]

  --model M       attach M's slot size and this card's capacity to the tables, and replay
                  only the traces captured from M (PROTOCOL §9: a slot size and a coverage
                  curve belong to one model and do not transfer to another)
  --out FILE      write the artefact here instead of a generated name
  -- ARGS...      passed through to \`moearc bench\` verbatim, e.g. -- --policy lfu --optimal

environment: MOEARC_BIN  path to the moearc binary, if it is not beside this script's parent
USAGE
}

while [ $# -gt 0 ]; do
    case $1 in
        --model) model=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "reproduce.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$here/.." && pwd -P)

# 🔴 The binary is looked for in three known places and never by glob. PROTOCOL §2: a
# glob-ordered pick once selected a Vulkan build 4.8x slower than SYCL, and it produced real
# output with exit 0.
bin=${MOEARC_BIN:-}
if [ -z "$bin" ]; then
    for c in "$root/bin/moearc" "$root/moearc" \
             "${CARGO_TARGET_DIR:-$root/target}/release/moearc" \
             "$root/target/release/moearc"; do
        [ -x "$c" ] && { bin=$c; break; }
    done
fi
[ -n "$bin" ] || {
    echo "reproduce.sh: cannot find the moearc binary." >&2
    echo "  installed bundle: it is ./bin/moearc beside this script's parent" >&2
    echo "  from source:      cargo build --release -p moearc-cli" >&2
    echo "  or set MOEARC_BIN=/path/to/moearc" >&2
    exit 1
}

# ⚠️ Release, not debug. The replay is a tight loop over hundreds of thousands of cache
# operations; a debug build takes the better part of an hour where release takes seconds.
# `moearc bench` prints the profile it was built with in the artefact, so this is a courtesy
# check rather than the record.
case "$bin" in
    *"/debug/"*) echo "reproduce.sh: ⚠️ that is a debug build; the replay will take ~1000x longer." >&2 ;;
esac

# The captures, wherever they ended up: a checkout puts them beside this script, and a
# release could reasonably put them under share/. Checked in order, never globbed.
traces=""
for c in "$here/traces" "$root/share/moearc/traces" "$root/bench/traces"; do
    [ -d "$c" ] && { traces=$c; break; }
done
[ -n "$traces" ] || {
    echo "reproduce.sh: no routing traces found." >&2
    echo "  Looked in: $here/traces, $root/share/moearc/traces, $root/bench/traces" >&2
    echo >&2
    echo "  🔴 The result IS the replay of those captures, so there is nothing to reproduce" >&2
    echo "  without them -- and this is not a broken install, it is a gap in the release:" >&2
    echo "  the tarball ships this script but not the ~7 MB of traces it replays. Run it" >&2
    echo "  from a git checkout, or point it at a checkout's bench/traces:" >&2
    echo >&2
    echo "      moearc bench --traces /path/to/MoEArc/bench/traces" >&2
    exit 1
}

[ -n "$out" ] || out=moearc-bench-$(date -u +%Y%m%dT%H%M%SZ).md

args=(bench --traces "$traces" --out "$out")
[ -n "$model" ] && args+=(--model "$model")
[ ${#extra[@]} -gt 0 ] && args+=("${extra[@]}")

echo "==================== MoEArc reproduction run ===================="
echo
echo "  binary   $bin"
echo "  traces   $traces"
echo "  artefact $out"
echo
echo "  Everything else — the box, the build commit, the device, the driver, the thresholds"
echo "  every check was judged against, and what the run did not measure — is recorded by"
echo "  \`moearc bench\` itself, in the artefact. It is written to be pasted whole into an"
echo "  issue; a summary retyped here would be a second place for it to go stale."
echo
echo "\$ $bin ${args[*]}"
echo "================================================================"
echo

set +e
"$bin" "${args[@]}"
status=$?
set -e

cat <<NOTES

==================== what this run did NOT measure ====================

Stated so the result above is not mistaken for more than it is.

  - Absolute throughput on your card. Not measured, and not reproducible from the release --
    see the header of this script. PROTOCOL §0 is the reason and predates the pivot:
    absolutes are an artefact of one machine and the shape is the result.
  - A llama.cpp baseline. Every published comparison against llama.cpp was WITHDRAWN
    (bench/PROTOCOL.md §1: llama-bench defaulted to 4 threads on a 20-core box and no
    invocation in the repository passed -t). \`moearc bench --llama-bench <PATH>\` runs one
    correctly -- pinned, and read back from the tool's own output -- on a build that has the
    timed half compiled in.
  - Prefill, and perplexity. Neither is a residency question and neither is replayed here.
  - A predicted tok/s. Hit rate predicts staged bytes exactly and predicts throughput not at
    all; PROTOCOL §9 forbids publishing the conversion, and nothing above contains one.

Exit code: 0 ran and produced a headline - 1 failed - 2 a subsystem is not built -
3 REFUSED, meaning the guards stopped it and the artefact says which.
NOTES

exit $status
