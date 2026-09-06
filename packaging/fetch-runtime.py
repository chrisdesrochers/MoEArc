#!/usr/bin/env python3
"""Fetch the Intel SYCL runtime MoEArc needs, from Intel's own channel.

`docs/ux.md`: "MoEArc must never hand the user a list of things to go install." This is how
that is kept without redistributing anyone else's binaries. It downloads the runtime packages
Intel publishes *for exactly this purpose* -- their own description is "shared common
libraries required to deploy executables on systems without the Intel oneAPI development
toolkits installed" -- verifies each against a pinned SHA-256, and extracts a named set of
files into one directory.

The user never types this. `packaging/install.sh` runs it once, and the launcher runs it if a
tarball is unpacked without it.

# What it is not

It is not a package manager and does not want to be. No dependency resolution, no environment,
no `pip`: a wheel is a zip file, `urllib` and `zipfile` are in the standard library, and the
digests are pinned in `runtime.lock.json`. The only requirement it adds to the target machine
is a Python 3 interpreter, which is why it does not use `pip` (a wheel installed with `pip`
would land in a site-packages tree we would then have to find) or `unzip` (frequently absent).

# Integrity

Every archive is hashed before a byte of it is extracted, and the expected digest is in the
lock file rather than fetched alongside the archive. The index is queried for the pinned
version and the file whose digest matches is chosen -- so the lock, not the index, decides
what gets installed, and a compromised or merely reorganised index cannot substitute a
different artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

INDEX = "https://pypi.org/pypi/{name}/{version}/json"
# A wheel lays its payload out under `<dist>-<ver>.data/data/...`; we only ever want files
# under `lib/` or `licensing/`, and we match on the tail so the prefix can change.
WANTED_PARENTS = ("lib", "licensing", "compiler")


def _log(msg: str) -> None:
    print(f"moearc: {msg}", file=sys.stderr, flush=True)


def _download(url: str, dest: Path, expect_sha: str, size: int | None) -> None:
    # Progress only on a terminal. A carriage-returned counter written to a log or a CI
    # transcript produces one enormous line, which is how the first version of this looked
    # in the clean-room run that proved the packaging worked.
    tty = sys.stderr.isatty()
    h = hashlib.sha256()
    got = 0
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
            got += len(chunk)
            if tty and size:
                print(f"\r  {dest.name}  {100 * got // size:3d}%", end="", file=sys.stderr,
                      flush=True)
    if tty:
        print("", file=sys.stderr)
    else:
        _log(f"downloaded {dest.name}, {got // (1 << 20)} MiB")
    actual = h.hexdigest()
    if actual != expect_sha:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"moearc: digest mismatch for {dest.name}\n"
            f"  expected {expect_sha}\n  got      {actual}\n"
            "This is either a corrupt download or a substituted artefact. Nothing was "
            "installed. Retry; if it happens twice, report it."
        )


def _resolve(name: str, version: str, sha256: str) -> tuple[str, int]:
    """Ask the index for this exact version and return the URL of the file we pinned."""
    try:
        with urllib.request.urlopen(INDEX.format(name=name, version=version), timeout=30) as r:
            meta = json.load(r)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"moearc: could not reach the package index for {name} {version} ({e.reason}).\n"
            "The Intel SYCL runtime is downloaded once, at install time. If this machine has "
            "no network, fetch it on one that does and copy the directory across -- see "
            "docs/packaging.md, 'Installing without a network'."
        ) from e
    for u in meta["urls"]:
        if u["digests"]["sha256"] == sha256:
            return u["url"], u.get("size") or 0
    raise SystemExit(
        f"moearc: {name} {version} is on the index but no file matches the pinned digest "
        f"{sha256}. packaging/runtime.lock.json and the index disagree; not guessing."
    )


def _extract(archive: Path, wanted: list[str], dest: Path) -> list[str]:
    """Pull exactly the named basenames out of the wheel. Returns what it wrote."""
    written = []
    with zipfile.ZipFile(archive) as z:
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            if len(parts) < 2 or parts[-2] not in WANTED_PARENTS:
                continue
            # First match wins; a wheel carries libsycl.so, .so.9 and .so.9.0.0 as three
            # identical copies, and we asked for one of them by name.
            by_name.setdefault(parts[-1], info)
        for name in wanted:
            info = by_name.get(name)
            if info is None:
                raise SystemExit(
                    f"moearc: {archive.name} does not contain {name!r}. The pinned version "
                    "and the file list in runtime.lock.json have diverged."
                )
            target = dest / name
            with z.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            if name.startswith("lib"):
                target.chmod(0o755)
            written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, required=True, help="directory to install into")
    ap.add_argument("--lock", type=Path, default=here / "runtime.lock.json")
    ap.add_argument("--keep-archives", action="store_true", help="leave the wheels on disk")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 0 if --dest already looks complete, 1 otherwise; download nothing",
    )
    args = ap.parse_args(argv)

    lock = json.loads(args.lock.read_text())
    wanted_all = [f for p in lock["packages"] for f in p["files"]]

    if args.check:
        missing = [f for f in wanted_all if not (args.dest / f).exists()]
        if missing:
            print(" ".join(missing))
            return 1
        return 0

    if all((args.dest / f).exists() for f in wanted_all):
        _log(f"runtime already present in {args.dest}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    _log(f"fetching the Intel SYCL runtime into {args.dest}")
    _log("this happens once; it is Intel's redistributable runtime, not the oneAPI toolkit")

    with tempfile.TemporaryDirectory(prefix="moearc-runtime-") as tmp:
        tmpd = Path(tmp)
        for pkg in lock["packages"]:
            if all((args.dest / f).exists() for f in pkg["files"]):
                continue
            url, size = _resolve(pkg["name"], pkg["version"], pkg["sha256"])
            archive = tmpd / url.rsplit("/", 1)[-1]
            _download(url, archive, pkg["sha256"], size)
            got = _extract(archive, pkg["files"], args.dest)
            _log(f"{pkg['name']} {pkg['version']}: {', '.join(got)}")
            if args.keep_archives:
                shutil.copy2(archive, args.dest.parent / archive.name)

    (args.dest / "PROVENANCE.txt").write_text(
        "These files were downloaded from the Python Package Index by\n"
        "packaging/fetch-runtime.py and verified against the SHA-256 digests in\n"
        "packaging/runtime.lock.json. They are Intel's, not MoEArc's, and are\n"
        "governed by the licences named there -- the Intel End User License Agreement\n"
        "for Developer Tools (text alongside this file), the Intel Simplified Software\n"
        "License, and Apache-2.0 WITH LLVM-exception.\n\n"
        + "".join(
            f"{p['name']:<22} {p['version']:<10} {p['license']}\n" for p in lock["packages"]
        )
    )
    total = sum(f.stat().st_size for f in args.dest.iterdir() if f.is_file())
    _log(f"runtime installed, {total // (1 << 20)} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
