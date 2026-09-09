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
# 🔴 WHAT THIS NO LONGER PROVES, AND IT USED TO
#
# Three of the four gates below ran binaries that left the payload with the retired SYCL
# engine: `moearc-selftest` (SYCL reaches the card from an empty environment), `moearc-server`
# (the server binary starts), and `moearc-bench` (a real forward pass, checked against stored
# token ids). All three are gone from the tarball -- see packaging/bundle.sh -- so this script
# can no longer prove that a model loads and produces correct tokens on a clean machine.
#
# That is a REDUCTION IN COVERAGE and it is stated here rather than quietly absorbed. What is
# left still tests the thing this file was written for: that MoEArc finds the Arc card on a
# machine that has never had oneAPI. What is missing moved to llama.cpp, which the tarball
# does not carry yet; when it does, the forward-pass gate comes back and the verdict below
# stops printing NOT PROVEN.
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
# ⬜ MOEARC_VERIFY_MODEL and its three companions are gone with the binary that consumed them.
# The finding they existed for is not: a driver stack can pass every check below and STILL be
# unable to load a model -- docs/packaging.md, "The GPU driver floor is higher for inference
# than for detection". That gap is now unguarded here, which is why the verdict says so out
# loud instead of printing a clean PASS that covers less than it used to.

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
echo "---- 2. the same, from a completely empty environment ----"
# The point of env -i is that LD_LIBRARY_PATH, ONEAPI_ROOT and PATH are all unset, so anything
# that resolves does so through the launcher and the bundle, not through the shell that ran it.
env -i ${MOEARC_RUNTIME_DIR:+MOEARC_RUNTIME_DIR=$MOEARC_RUNTIME_DIR} /opt/m/moearc --no-tui \
  && echo "empty-environment device report: ok"
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
check 'empty-environment device report: ok' 'the card is still found with the environment emptied'
refute 'no usable GPU|<none>' 'detection did not fall back to "no usable GPU"'
refute 'cannot open shared object file' 'nothing failed in the dynamic loader'

if [ "$status" != 0 ]; then
    echo "  FAIL  container exited $status"
    fail=1
fi

echo "  ----  NOT PROVEN: a model loading, a forward pass, or correct token ids."
echo "        Those gates ran moearc-selftest, moearc-server and moearc-bench, and all three"
echo "        left the payload with the retired SYCL engine. They come back when llama.cpp is"
echo "        bundled. Until then a PASS here means the CARD IS FOUND, not that it computes."

echo
if [ "$fail" = 0 ]; then
    echo "clean-environment verification PASSED (detection only -- see NOT PROVEN above)"
else
    echo "clean-environment verification FAILED"
fi
exit "$fail"
