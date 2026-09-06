#!/usr/bin/env bash
# Build the distributable MoEArc tarball.
#
# Produces a directory tree that runs on a machine with an Arc card, an Intel GPU driver, and
# nothing else -- specifically, no oneAPI. What makes that possible is documented in
# docs/packaging.md; the two mechanical steps are here:
#
#   1. `elf-relocatable.py` shortens the kernel object's DT_SONAME (and the matching DT_NEEDED
#      in every binary that links it) from this build tree's absolute OUT_DIR path down to the
#      bare file name, so the loader searches for it instead of opening a path that will not
#      exist on the target machine.
#   2. The launcher puts the runtime directory on LD_LIBRARY_PATH before exec'ing the real
#      binary, which is the only search path a dlopened UR adapter's own dependencies inherit.
#
# By default the tarball contains no Intel binaries at all: `runtime/` is populated at install
# time by fetch-runtime.py from Intel's published redistributable packages. `--with-runtime`
# vendors them into the tarball instead, which is for air-gapped installs and carries a
# licence position you should read in docs/packaging.md before publishing one.

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
    echo "==> cargo build --release (moearc-server/engine, moearc-engine/gpu)"
    ( cd "$repo" && CARGO_TARGET_DIR=$target_dir cargo build --release \
        -p moearc-cli -p moearc-kernels -p moearc-server -p moearc-engine \
        --features moearc-server/engine,moearc-engine/gpu --bins --examples )
fi

rel=$target_dir/release
kernel_so=$(ls "$rel"/build/moearc-kernels-*/out/libmoearc_kernels.so 2>/dev/null | head -1 || true)

[ -n "$kernel_so" ] || { echo "bundle.sh: no libmoearc_kernels.so under $rel/build -- build first, or pass --build" >&2; exit 1; }

# name in the bundle : path in the build tree
declare -a payload=(
    "moearc:$rel/moearc"
    "moearc-server:$rel/moearc-server"
    "moearc-bench:$rel/examples/hybrid_sweep"
    "moearc-selftest:$rel/moearc-kernels-smoke"
)

for entry in "${payload[@]}"; do
    src=${entry#*:}
    [ -x "$src" ] || { echo "bundle.sh: missing $src -- build first, or pass --build" >&2; exit 1; }
done

# `moearc-server` must actually link the kernels; built without --features engine it does not,
# and the tarball would ship a server that cannot infer. Checking rather than trusting,
# because that is the exact class of mistake docs/packaging.md records twice.
if ! readelf -d "$rel/moearc-server" | grep -q 'libmoearc_kernels\.so'; then
    echo "bundle.sh: $rel/moearc-server does not link libmoearc_kernels.so." >&2
    echo "           It was built without --features moearc-server/engine. Rebuild with --build." >&2
    exit 1
fi

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
install -m 0755 "$kernel_so" "$root/libexec/libmoearc_kernels.so"
install -m 0755 "$here/fetch-runtime.py" "$root/libexec/fetch-runtime.py"
install -m 0644 "$here/runtime.lock.json" "$root/share/moearc/runtime.lock.json"

echo "==> making the kernel object relocatable"
python3 "$here/elf-relocatable.py" "$root/libexec/libmoearc_kernels.so" \
    "$root/libexec/moearc" "$root/libexec/moearc-server" \
    "$root/libexec/moearc-bench" "$root/libexec/moearc-selftest"

# A path left in DT_NEEDED is the failure this whole step exists to prevent, and it is silent
# until someone unpacks the tarball on another machine. Assert it.
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

# Provenance. A tarball that cannot say what built it is not evidence of anything.
{
    echo "name:        $name"
    echo "built:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:      $(cd "$repo" && git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty:       $(cd "$repo" && { git diff --quiet 2>/dev/null && echo no || echo YES; })"
    echo "rustc:       $(rustc --version 2>/dev/null || echo unknown)"
    echo "icpx:        $("${ONEAPI_ROOT:-/opt/intel/oneapi}/compiler/latest/bin/icpx" --version 2>/dev/null | head -1 || echo unknown)"
    echo "build glibc: $(ldd --version 2>/dev/null | head -1 || echo unknown)"
    echo "runtime:     $([ "$with_runtime" = 1 ] && echo vendored || echo 'fetched at install time')"
    echo
    echo "minimum target glibc (max GLIBC_ symbol version required by the shipped binaries):"
    for f in "$root"/libexec/moearc* "$root/libexec/libmoearc_kernels.so"; do
        case $f in *.py) continue ;; esac
        printf '  %-24s %s\n' "$(basename "$f")" \
            "$(objdump -p "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1)"
    done
} > "$root/share/moearc/BUILD-INFO.txt"

mkdir -p "$out_dir"
tar -C "$stage" -czf "$out_dir/$name.tar.gz" "$name"
( cd "$out_dir" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256" )

echo
echo "==> $out_dir/$name.tar.gz  ($(du -h --apparent-size "$out_dir/$name.tar.gz" | cut -f1))"
cat "$root/share/moearc/BUILD-INFO.txt"
