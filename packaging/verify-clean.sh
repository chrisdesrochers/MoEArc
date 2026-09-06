#!/usr/bin/env bash
# Run a built tarball on a machine that has never had oneAPI installed, and fail loudly if it
# does not find the GPU.
#
# This is the test the packaging exists to pass, and it is deliberately not a unit test.
# docs/packaging.md records that the previous packaging bug survived 309 green tests because
# every one of them ran in a shell with setvars.sh sourced. A container is the only cheap way
# to be certain the toolkit is unreachable: no /opt/intel, a different distro release, a
# different glibc, and an environment we control completely.
#
#   packaging/verify-clean.sh dist/moearc-*.tar.gz
#
# 🔴 It passes the *render node*, not all of /dev/dri. Handing a container the card* nodes as
# well makes Intel's compute runtime abort at teardown --
# "Abort was called at 433 line in file: ./shared/source/os_interface/linux/drm_neo.cpp" --
# after the workload has already succeeded. That is a container-configuration artefact and not
# a MoEArc failure, and it cost time to tell apart, so it is pinned here.

set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$here/.." && pwd -P)

tarball=${1:-}
render=${MOEARC_RENDER_NODE:-}
image=${MOEARC_CLEAN_IMAGE:-moearc-clean:noble}
driver=${MOEARC_CLEAN_DRIVER:-intel-repo}
base=${MOEARC_CLEAN_BASE:-docker.io/library/ubuntu:24.04}
runtime_cache=${MOEARC_RUNTIME_CACHE:-}
# 🔴 Optional but strongly recommended, and the reason is measured: a driver stack can pass
# every check below and still be unable to load a model. See docs/packaging.md, "The GPU driver
# floor is higher for inference than for detection". Point this at a small MoE gguf.
verify_model=${MOEARC_VERIFY_MODEL:-}
verify_model_args=${MOEARC_VERIFY_MODEL_ARGS:-16 512 256 off}
verify_model_ids=${MOEARC_VERIFY_MODEL_IDS:-}
# A reference token-id file *inside the bundle*, e.g. bench/references/olmoe-1b-7b.capital.ids.
# Without one the forward pass proves it ran; with one it proves it ran correctly.
verify_model_ref=${MOEARC_VERIFY_MODEL_REF:--}

if [ -z "$tarball" ]; then
    tarball=$(ls -t "$repo"/dist/moearc-*.tar.gz 2>/dev/null | head -1 || true)
fi
[ -n "$tarball" ] && [ -f "$tarball" ] || {
    echo "verify-clean.sh: no tarball. Run packaging/bundle.sh first, or pass one." >&2
    exit 2
}
tarball=$(readlink -f "$tarball")

engine=$(command -v podman || command -v docker) || {
    echo "verify-clean.sh: needs podman or docker." >&2; exit 2; }

# Pick the discrete card's render node if we were not told one. renderD128 is the first DRM
# render device, which on a box with an iGPU is the iGPU -- and bench/README.md is emphatic
# that a run on the iGPU "does not fail, it succeeds and lies". Prefer the last node, which is
# the discrete card on every machine this has been run on, and say which was chosen.
if [ -z "$render" ]; then
    render=$(ls -1 /dev/dri/renderD* 2>/dev/null | tail -1 || true)
fi
[ -n "$render" ] || { echo "verify-clean.sh: no /dev/dri/render* on this host." >&2; exit 2; }

echo "==> tarball  $tarball"
echo "==> engine   $engine"
echo "==> device   $render"
echo "==> image    $image (BASE=$base DRIVER=$driver)"
echo

if ! "$engine" image exists "$image" 2>/dev/null; then
    echo "==> building the clean image"
    "$engine" build --build-arg "DRIVER=$driver" --build-arg "BASE=$base" \
        -f "$here/Containerfile.clean" -t "$image" "$repo"
fi

mounts=(-v "$tarball:/dist/moearc.tar.gz:ro")
if [ -n "$verify_model" ]; then
    mounts+=(-v "$(readlink -f "$verify_model"):/model.gguf:ro")
fi
if [ -n "$runtime_cache" ]; then
    mkdir -p "$runtime_cache"
    mounts+=(-v "$runtime_cache:/rtcache")
fi

log=$(mktemp "${TMPDIR:-/tmp}/moearc-verify.XXXXXX")
trap 'rm -f "$log"' EXIT

set +e
"$engine" run --rm --device "$render" --group-add keep-groups "${mounts[@]}" "$image" \
    bash -lc '
set -e
echo "---- the machine ----"
grep PRETTY_NAME /etc/os-release
ldd --version | head -1
echo "oneAPI:            $(ls -d /opt/intel 2>/dev/null || echo ABSENT)"
echo "LD_LIBRARY_PATH:   [${LD_LIBRARY_PATH:-unset}]"
echo "level zero driver:"; dpkg -l libze-intel-gpu1 2>/dev/null | tail -1 || echo "  none"
echo
mkdir -p /opt/m && tar -C /opt/m --strip-components=1 -xzf /dist/moearc.tar.gz
[ -d /rtcache ] && export MOEARC_RUNTIME_DIR=/rtcache
echo "---- 1. device report, with no SYCL runtime installed at all ----"
/opt/m/moearc --no-tui
echo
echo "---- 2. the SYCL path, from a completely empty environment ----"
env -i ${MOEARC_RUNTIME_DIR:+MOEARC_RUNTIME_DIR=$MOEARC_RUNTIME_DIR} /opt/m/moearc-selftest
echo
echo "---- 3. the server binary starts ----"
env -i ${MOEARC_RUNTIME_DIR:+MOEARC_RUNTIME_DIR=$MOEARC_RUNTIME_DIR} /opt/m/moearc-server --help >/dev/null \
  && echo "moearc-server --help: ok"
if [ -f /model.gguf ]; then
  echo
  echo "---- 4. a real forward pass, which is a strictly harder test than 2 ----"
  env -i ${MOEARC_RUNTIME_DIR:+MOEARC_RUNTIME_DIR=$MOEARC_RUNTIME_DIR} \
    /opt/m/moearc-bench /model.gguf '"$verify_model_args"' \
    '"$verify_model_ref"' '"$verify_model_ids"' \
    && echo "forward pass: completed"
fi
' 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e

echo
echo "==================== verdict ===================="
fail=0
check() {
    if grep -qE "$1" "$log"; then
        echo "  PASS  $2"
    else
        echo "  FAIL  $2"
        fail=1
    fi
}
refute() {
    if grep -qE "$1" "$log"; then
        echo "  FAIL  $2"
        fail=1
    else
        echo "  PASS  $2"
    fi
}

check 'oneAPI: +ABSENT' 'the test machine genuinely has no oneAPI'
check 'Intel\(R\) Arc' 'moearc names an Intel Arc device'
check 'moearc-kernels-smoke: ok device=Intel\(R\) Arc' 'SYCL finds the Arc card from an empty environment'
refute 'device=<none' 'SYCL did not fall back to "no usable GPU"'
refute 'cannot open shared object file' 'nothing failed in the dynamic loader'
check 'moearc-server --help: ok' 'the server binary starts'
if [ -n "$verify_model" ]; then
    refute 'LOAD FAILED|Abort was called|Segmentation fault' \
        'a real model loaded and ran (this is what a stale GPU driver fails)'
    check 'forward pass: completed' 'the forward pass finished'
    if [ "$verify_model_ref" != "-" ]; then
        check 'ref [0-9]+/[0-9]+' 'the token ids matched the stored llama.cpp reference'
    fi
fi

if [ "$status" != 0 ]; then
    echo "  FAIL  container exited $status"
    fail=1
fi

echo
if [ "$fail" = 0 ]; then
    echo "clean-environment verification PASSED"
else
    echo "clean-environment verification FAILED"
fi
exit "$fail"
