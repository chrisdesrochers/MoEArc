#!/usr/bin/env bash
# Build the distributable MoEArc tarball.
#
# Produces a directory tree that runs on a machine with an Arc card, an Intel GPU driver, and
# nothing else -- specifically, no oneAPI. What makes that possible is documented in
# docs/packaging.md.
#
# 🔴 WHAT THIS DOES NOT BUILD, AS OF 2026-09-08
#
# Until today this ran `cargo build --features moearc-server/engine,moearc-engine/gpu` and then
# asserted that `moearc-server` linked `libmoearc_kernels.so`. Both are gone. MoEArc's engine is
# llama.cpp; the hand-written SYCL engine is retired, the project has said so publicly, and a
# release that carries it in the payload contradicts the product shipped beside it. The
# assertion has been kept and turned round: nothing staged here may link that object.
#
# Two consequences, stated here rather than discovered in a release:
#
#   1. `elf-relocatable.py` is no longer run. Its job was to shorten one absolute DT_SONAME --
#      the kernel object's -- and no staged file has one. The guard that mattered is kept: no
#      DT_NEEDED in any staged binary may contain a slash.
#   2. ⬜ **llama.cpp's shared objects are not bundled yet.** `moearc serve` supervises
#      `llama-server` and resolves it beside its own binary first, so a tarball that also
#      carried `llama-server`, `libllama.so.0` and the `libggml*.so.0` family would be
#      self-contained. Staging them is real work with a real licence obligation --
#      packaging/THIRD-PARTY.md does not yet cover shipping MIT-licensed llama.cpp binaries --
#      and it is tracked in docs/pivot-inventory.md under `packaging/`. Until it lands, the
#      tarball needs a llama.cpp on the target machine.
#
# The launcher puts the runtime directory on LD_LIBRARY_PATH before exec'ing the real binary,
# which is the only search path a dlopened UR adapter's own dependencies inherit. `runtime/` is
# populated at install time by fetch-runtime.py from Intel's published redistributable packages;
# `--with-runtime` vendors them into the tarball instead, which is for air-gapped installs and
# carries a licence position you should read in docs/packaging.md before publishing one.

set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$here/.." && pwd -P)

target_dir=${CARGO_TARGET_DIR:-$repo/target}
out_dir=$repo/dist
with_runtime=0
do_build=0
version=""

usage() {
    cat <<USAGE
usage: packaging/bundle.sh [options]

  --target-dir DIR   cargo target directory (default: \$CARGO_TARGET_DIR or ./target)
  --out DIR          where to write the tarball (default: ./dist)
  --version VER      version string for the artefact name (default: from Cargo.toml + git)
  --with-runtime     vendor the Intel SYCL runtime into the tarball instead of fetching it
                     at install time. Read docs/packaging.md first.
  --build            run cargo build --release with the features this needs, first
USAGE
}

while [ $# -gt 0 ]; do
    case $1 in
        --target-dir) target_dir=$2; shift 2 ;;
        --out) out_dir=$2; shift 2 ;;
        --version) version=$2; shift 2 ;;
        --with-runtime) with_runtime=1; shift ;;
        --build) do_build=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "bundle.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$do_build" = 1 ]; then
    # No feature flags. Every one that used to be here (`moearc-server/engine`,
    # `moearc-engine/gpu`) pulls the retired SYCL engine, and `--examples` built its
    # benchmark harnesses. `moearc` is the product; it reaches Level Zero through
    # moearc-device's dlopen and links no SYCL at all, which is the property that lets the
    # tarball work on first unpack.
    echo "==> cargo build --release -p moearc-cli"
    ( cd "$repo" && CARGO_TARGET_DIR=$target_dir cargo build --release \
        -p moearc-cli --bins )
fi

rel=$target_dir/release

# name in the bundle : path in the build tree
#
# 🔴 THREE BINARIES LEFT THIS LIST AND EACH FOR A DIFFERENT REASON. Written out because a
# payload that silently shrinks is how a release ships something nobody meant to ship.
#
#   * `moearc-bench` (the retired engine's `hybrid_sweep`) and `moearc-selftest` (the kernel
#     object's dlopen smoke test) existed ONLY to exercise the SYCL engine. Keeping either
#     means putting that engine back in the payload. `bench/reproduce.sh` is still installed
#     below and still looks for `./moearc-bench`; it will now report that it cannot find it,
#     which is correct and loud. It needs re-pointing at `moearc bench`.
#   * `moearc-server` is dropped because WITHOUT llama.cpp LINKED IN IT IS A STUB -- an echo
#     server that answers OpenAI-format requests with the prompt read back, and does so over
#     the real routing, templating and SSE path. That stub is the right thing to have in the
#     tree (it is what let the whole serving layer be written and tested before an engine
#     existed) and the wrong thing to have in a tarball, where its output is indistinguishable
#     from a model's to anyone not reading /health. Building it non-stub means linking
#     llama.cpp, which means shipping llama.cpp's shared objects, which is the licence work
#     noted in the header. It comes back the same day that lands.
#
# `moearc serve` -- the documented way to run a server -- does not use `moearc-server` at all.
# It supervises llama.cpp's own `llama-server` as a child process. Nothing is lost here that a
# user of this tarball had.
declare -a payload=(
    "moearc:$rel/moearc"
)

for entry in "${payload[@]}"; do
    src=${entry#*:}
    [ -x "$src" ] || { echo "bundle.sh: missing $src -- build first, or pass --build" >&2; exit 1; }
done

# 🔴 This check used to assert the opposite: that `moearc-server` DID link
# libmoearc_kernels.so, on the reasoning that a server which cannot infer must not ship. That
# reasoning is now inverted, not abandoned. The engine is llama.cpp; the kernel object is the
# retired one; and a stale `target/release` from a build that predates the pivot is exactly how
# it would get into a release unnoticed. Checking rather than trusting, because that is the
# exact class of mistake docs/packaging.md records twice.
for entry in "${payload[@]}"; do
    src=${entry#*:}
    if readelf -d "$src" 2>/dev/null | grep -q 'libmoearc_kernels\.so'; then
        echo "bundle.sh: $src links libmoearc_kernels.so -- the RETIRED SYCL engine." >&2
        echo "           That build predates the pivot to llama.cpp. Delete $rel and rebuild" >&2
        echo "           with --build; a release must not carry that engine." >&2
        exit 1
    fi
done

if [ -z "$version" ]; then
    v=$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$repo/Cargo.toml" | head -1)
    g=$(cd "$repo" && git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
    version="${v:-0.0.0}+g$g"
fi

name=moearc-$version-linux-x86_64
stage=$(mktemp -d "${TMPDIR:-/tmp}/moearc-bundle.XXXXXX")
trap 'rm -rf "$stage"' EXIT
root=$stage/$name

mkdir -p "$root/libexec" "$root/share/moearc" "$root/share/doc/moearc" "$root/bench"

echo "==> staging $name"
for entry in "${payload[@]}"; do
    dst=${entry%%:*}; src=${entry#*:}
    install -m 0755 "$src" "$root/libexec/$dst"
    install -m 0755 "$here/launcher.sh" "$root/$dst"
done
install -m 0755 "$here/fetch-runtime.py" "$root/libexec/fetch-runtime.py"
install -m 0644 "$here/runtime.lock.json" "$root/share/moearc/runtime.lock.json"

# `elf-relocatable.py` is not run: it rewrites one absolute DT_SONAME into a bare file name,
# and the only object that ever had one was the retired kernel build's. The *guard* it existed
# to satisfy is what matters and is kept below.
#
# A path left in DT_NEEDED is the failure that step existed to prevent, and it is silent until
# someone unpacks the tarball on another machine. Assert it.
for f in "$root"/libexec/moearc*; do
    case $f in *.py) continue ;; esac
    if readelf -d "$f" 2>/dev/null | awk '/NEEDED/ {print $NF}' | grep -q '/'; then
        echo "bundle.sh: $f still names a dependency by absolute path:" >&2
        readelf -d "$f" | grep NEEDED >&2
        exit 1
    fi
done

install -m 0644 "$repo/LICENSE" "$root/share/doc/moearc/LICENSE"
install -m 0644 "$repo/NOTICE" "$root/share/doc/moearc/NOTICE"
[ -f "$here/THIRD-PARTY.md" ] && install -m 0644 "$here/THIRD-PARTY.md" "$root/share/doc/moearc/THIRD-PARTY.md"
[ -f "$repo/bench/reproduce.sh" ] && install -m 0755 "$repo/bench/reproduce.sh" "$root/bench/reproduce.sh"
for f in "$repo"/bench/references/*.ids; do
    [ -e "$f" ] && install -D -m 0644 "$f" "$root/bench/references/$(basename "$f")"
done

if [ "$with_runtime" = 1 ]; then
    echo "==> vendoring the Intel SYCL runtime into the tarball"
    python3 "$here/fetch-runtime.py" --dest "$root/runtime" --lock "$here/runtime.lock.json"
fi

# 🔴 The artefact has to be able to be built twice and come out the same, or the sha256 in a
# release is only checkable by the person who produced it. Two things made this tarball differ
# from itself: the wall-clock `built:` stamp, and the mtime/uid/gid/order tar records for every
# member. Both are pinned to SOURCE_DATE_EPOCH, defaulting to the commit's own timestamp, so
# they are a property of the commit rather than of the minute someone typed the command.
#
# This is repeatability on one machine, and packaging/RELEASE.md says so rather than claiming
# more: a different rustc or icpx may still produce different bytes, and nobody has checked.
source_date_epoch=${SOURCE_DATE_EPOCH:-$(cd "$repo" && git log -1 --format=%ct 2>/dev/null || date -u +%s)}

# Provenance. A tarball that cannot say what built it is not evidence of anything.
{
    echo "name:        $name"
    echo "built:       $(date -u -d "@$source_date_epoch" +%Y-%m-%dT%H:%M:%SZ)"
    echo "source date: $source_date_epoch (SOURCE_DATE_EPOCH)"
    echo "commit:      $(cd "$repo" && git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty:       $(cd "$repo" && { git diff --quiet 2>/dev/null && echo no || echo YES; })"
    echo "rustc:       $(rustc --version 2>/dev/null || echo unknown)"
    echo "build glibc: $(ldd --version 2>/dev/null | head -1 || echo unknown)"
    echo "runtime:     $([ "$with_runtime" = 1 ] && echo vendored || echo 'fetched at install time')"
    # No `icpx:` line. Nothing in this payload is compiled by it any more, and a version string
    # for a compiler that touched none of these bytes is provenance that describes the wrong
    # machine. The engine's provenance is llama.cpp's, and llama.cpp is not in this tarball.
    echo "engine:      llama.cpp, supervised as a child process -- NOT BUNDLED (see the header)"
    echo
    echo "minimum target glibc (max GLIBC_ symbol version required by the shipped binaries):"
    for f in "$root"/libexec/moearc*; do
        case $f in *.py) continue ;; esac
        printf '  %-24s %s\n' "$(basename "$f")" \
            "$(objdump -p "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1)"
    done
} > "$root/share/moearc/BUILD-INFO.txt"

mkdir -p "$out_dir"
# --sort/--owner/--group/--mtime remove everything about *when and by whom* this ran from the
# archive; `gzip -n` keeps the original filename and timestamp out of the gzip header, which
# `tar -z` would otherwise leave there.
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$source_date_epoch" \
    --format=gnu -C "$stage" -cf - "$name" | gzip -n -9 > "$out_dir/$name.tar.gz"
( cd "$out_dir" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256" )

echo
echo "==> $out_dir/$name.tar.gz  ($(du -h --apparent-size "$out_dir/$name.tar.gz" | cut -f1))"
cat "$root/share/moearc/BUILD-INFO.txt"

echo
echo "==> what is NOT in this tarball"
echo "    * no inference engine. \`moearc serve\` supervises llama.cpp's \`llama-server\`, and"
echo "      finds it beside itself, via \$MOEARC_LLAMA_SERVER, or on \$PATH. A target machine"
echo "      needs one. ⬜ Bundling llama.cpp's binaries is tracked in docs/pivot-inventory.md"
echo "      and needs packaging/THIRD-PARTY.md written first."
echo "    * no libmoearc_kernels.so, no moearc-bench, no moearc-selftest -- all three belong to"
echo "      the retired SYCL engine, and the check above fails the build if one comes back."
echo "    * no moearc-server. Built without a linked llama.cpp it is an ECHO STUB, and a"
echo "      stub that speaks fluent OpenAI does not belong in a release. \`moearc serve\`"
echo "      is the server, and it does not use that binary."
echo "    * bench/reproduce.sh is installed but has no bench binary to run until it is"
echo "      re-pointed at \`moearc bench\`."

# A tarball whose provenance reads `unknown` is not evidence of anything, and the way to get
# one is undramatic: run bundle.sh in a shell where rustc is not on PATH and every field
# quietly falls back. Said out loud here rather than discovered in a release.
if grep -qE ': +unknown|^dirty: +YES' "$root/share/moearc/BUILD-INFO.txt"; then
    echo
    echo "bundle.sh: ⚠️  this build is not release-grade -- BUILD-INFO.txt above has an" >&2
    echo "           'unknown' field or a dirty tree. See packaging/RELEASE.md." >&2
fi
