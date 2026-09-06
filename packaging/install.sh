#!/bin/sh
# MoEArc installer.
#
#   curl -fsSL https://raw.githubusercontent.com/chrisdesrochers/MoEArc/main/packaging/install.sh | sh
#
# Unpacks a release into a prefix, fetches the Intel SYCL runtime once, and links the
# commands onto PATH. `docs/ux.md` allows exactly one thing to be asked of the user, and it
# is the kernel GPU driver; if that is missing this says so by name and stops.
#
# 🔴 One artefact name, for every tag. `releases/latest/download/<name>` only works when the
# asset name is fixed — a versioned filename cannot be resolved through it — so the tarball is
# published as `moearc-linux-x86_64.tar.gz` under every tag and the version is carried inside,
# in `share/moearc/BUILD-INFO.txt` and in the directory name. `packaging/RELEASE.md` is the
# checklist that keeps that true; if the two disagree, this file is wrong.

set -eu

REPO=chrisdesrochers/MoEArc
ASSET=moearc-linux-x86_64.tar.gz
PREFIX=${MOEARC_PREFIX:-$HOME/.local/share/moearc}
BINDIR=${MOEARC_BINDIR:-$HOME/.local/bin}
TARBALL=${MOEARC_TARBALL:-}
VERSION=${MOEARC_VERSION:-latest}
# The smallest tarball this project has ever produced is 4.8 MB. Anything under a megabyte is
# an error page, a redirect stub or a truncated transfer, none of which should reach `tar`.
MIN_BYTES=1048576

say() { printf 'moearc: %s\n' "$*" >&2; }
die() { printf 'moearc: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "MoEArc is Linux-only today (found $(uname -s))."
[ "$(uname -m)" = x86_64 ] || die "MoEArc ships x86_64 only (found $(uname -m))."

# The one dependency we are allowed to have. Named exactly, with the reason.
#
# Any render node, not renderD128. On a box with an iGPU the discrete card is renderD129, and
# a container is often given only the node it needs -- the first version of this check named
# renderD128 and refused to install on exactly the machine the packaging was proved on.
if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
    say "no /dev/dri/render* device."
    say "MoEArc needs the kernel GPU driver -- 'xe' for Arc B-series (Battlemage),"
    say "'i915' for A-series. It ships with the kernel; check 'lsmod | grep -E \"^xe|^i915\"'."
    die "stopping: nothing else here can work without it."
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/moearc-install.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# Downloads $1 to $2 and prints the HTTP status on stdout. Returns the downloader's exit
# status, so a caller can tell "the server said no" from "there was no server".
#
# The status is what makes the difference between the two messages below, and getting it out of
# both curl and wget is the only reason this is more than one line.
http_get() {
    if command -v curl >/dev/null 2>&1; then
        # curl's own "(22) The requested URL returned error: 404" is kept out of the way and
        # replayed only when we have nothing better to say. A raw 404 above our explanation
        # buries the explanation.
        curl -fsSL --connect-timeout 20 --retry 2 -o "$2" -w '%{http_code}' "$1" \
            2>"$tmp/download-error"
    elif command -v wget >/dev/null 2>&1; then
        # `-S` puts the response headers on stderr; the last status line is the one that
        # counts, after any redirects.
        rc=0
        wget -q -S -O "$2" "$1" 2>"$tmp/headers" || rc=$?
        awk '/^ *HTTP\/[0-9.]+ [0-9]+/ { code = $2 } END { printf "%d", code }' "$tmp/headers"
        return $rc
    else
        die "need curl or wget."
    fi
}

# A tarball, or an explanation. Never a truncated file handed to `tar`, and never a raw 404.
verify_archive() {
    file=$1
    [ -s "$file" ] || die "the download produced an empty file."
    size=$(wc -c < "$file" | tr -d ' ')
    if [ "$size" -lt "$MIN_BYTES" ]; then
        say "downloaded only $size bytes, which is too small to be a MoEArc release."
        say "That is usually an error page or a truncated transfer, not an archive."
        die "stopping before unpacking something that is not a tarball."
    fi
    # gzip magic, 1f 8b. `od` is in POSIX and `file` is not installed everywhere.
    magic=$(od -An -tx1 -N2 "$file" | tr -d ' \n')
    [ "$magic" = "1f8b" ] || die "the download is not a gzip archive (starts with 0x$magic)."
}

no_release() {
    say "the server has no such release: $1"
    say ""
    if [ "$VERSION" = latest ]; then
        say "MoEArc has no published release yet, or the latest one has no $ASSET attached."
        say "Check https://github.com/$REPO/releases for what is available."
    else
        say "There is no tag '$VERSION' with a $ASSET attached."
        say "Check https://github.com/$REPO/releases, or leave MOEARC_VERSION unset for latest."
    fi
    say ""
    say "You can build and install one yourself from a checkout -- it needs the oneAPI"
    say "toolkit on the build machine, and nothing on the target machine:"
    say ""
    say "    packaging/bundle.sh --build"
    say "    MOEARC_TARBALL=dist/moearc-*-linux-x86_64.tar.gz sh packaging/install.sh"
    say ""
    die "stopping: there is nothing to download."
}

if [ -n "$TARBALL" ]; then
    say "installing from $TARBALL"
    [ -f "$TARBALL" ] || die "MOEARC_TARBALL=$TARBALL does not exist."
    cp "$TARBALL" "$tmp/moearc.tar.gz"
else
    if [ "$VERSION" = latest ]; then
        url="https://github.com/$REPO/releases/latest/download/$ASSET"
    else
        url="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
    fi
    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
        || die "need curl or wget to download a release. Or build one: see packaging/RELEASE.md."
    say "downloading $url"

    rc=0
    status=$(http_get "$url" "$tmp/moearc.tar.gz") || rc=$?
    if [ "$rc" != 0 ]; then
        case $status in
            404) no_release "$url" ;;
            403)
                say "the server refused the download with HTTP 403: $url"
                say "That is a rate limit or a network policy, not a missing release."
                die "stopping. Retry later, or set MOEARC_TARBALL=/path/to/tarball."
                ;;
            000|"")
                say "could not reach $url."
                say "Nothing answered -- check the network, a proxy, or DNS."
                [ -s "$tmp/download-error" ] && cat "$tmp/download-error" >&2
                die "stopping: the download did not start."
                ;;
            *)
                say "the download failed with HTTP $status: $url"
                [ -s "$tmp/download-error" ] && cat "$tmp/download-error" >&2
                die "stopping. Set MOEARC_TARBALL=/path/to/tarball to install a local build."
                ;;
        esac
    fi
fi

verify_archive "$tmp/moearc.tar.gz"

# Checksum when the release publishes one. Advisory rather than required: a local
# MOEARC_TARBALL has none, and an older release may not either -- but a *mismatch* always
# stops, because that is the case where continuing is worse than failing.
if [ -z "$TARBALL" ] && command -v sha256sum >/dev/null 2>&1; then
    if http_get "$url.sha256" "$tmp/moearc.sha256" >/dev/null 2>&1; then
        # Compare the digest only. The published file names the artefact as it was built,
        # which is not necessarily the name it was uploaded under; the bytes are what matter.
        want=$(awk 'NR == 1 { print $1 }' "$tmp/moearc.sha256")
        have=$(sha256sum "$tmp/moearc.tar.gz" | awk '{ print $1 }')
        if [ "$want" != "$have" ]; then
            say "checksum mismatch for $ASSET:"
            say "  published $want"
            say "  received  $have"
            die "stopping: this is not the file the release says it is."
        fi
        say "sha256 verified against the published checksum"
    else
        say "note: this release publishes no .sha256; the download was not checksum-verified."
    fi
fi

tar -C "$tmp" -xzf "$tmp/moearc.tar.gz"
# -mindepth 1: `find` lists its own starting directory first, and the staging directory is
# itself called moearc-install.XXXXXX -- which matches `moearc-*`. Without this the installer
# moves the whole staging directory into the prefix and the tree ends up one level too deep.
src=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'moearc-*' | head -1)
[ -n "$src" ] || die "tarball did not contain a moearc-* directory."
[ -x "$src/moearc" ] || die "tarball has no executable 'moearc' at its root -- it is not a MoEArc release."

# `rm -rf` on a path that came out of the environment. MOEARC_PREFIX is meant to be a
# directory we own; these are the values where getting it wrong is unrecoverable.
case $PREFIX in
    ""|"/"|"$HOME"|"$HOME/") die "refusing to install over PREFIX='$PREFIX'." ;;
esac

rm -rf "$PREFIX"
mkdir -p "$(dirname "$PREFIX")"
mv "$src" "$PREFIX"

# Runtime up front rather than on first serve: an install that finishes is a better promise
# than one that downloads 230 MB the first time someone is trying to run a model.
if command -v python3 >/dev/null 2>&1; then
    python3 "$PREFIX/libexec/fetch-runtime.py" --dest "$PREFIX/runtime" \
        --lock "$PREFIX/share/moearc/runtime.lock.json"
else
    say "python3 not found -- skipping the SYCL runtime for now."
    say "'moearc' (device report) works without it; running a model does not."
fi

mkdir -p "$BINDIR"
for cmd in moearc moearc-server moearc-bench moearc-selftest; do
    ln -sf "$PREFIX/$cmd" "$BINDIR/$cmd"
done

say "installed to $PREFIX, linked into $BINDIR"
if [ -f "$PREFIX/share/moearc/BUILD-INFO.txt" ]; then
    say "$(awk '/^name:/ { $1 = ""; sub(/^ +/, ""); print }' "$PREFIX/share/moearc/BUILD-INFO.txt")"
fi
case ":${PATH}:" in
    *":$BINDIR:"*) ;;
    *) say "note: $BINDIR is not on PATH. Add it, or run $BINDIR/moearc directly." ;;
esac
echo
"$PREFIX/moearc" --no-tui || true
