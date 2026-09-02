#!/usr/bin/env bash
# Scan staged content (or given files) for credential-shaped strings.
# Exit 1 on a hit. Used by the pre-commit hook and by CI.
#
#   tools/scan-secrets.sh              # scan staged diff
#   tools/scan-secrets.sh file...      # scan specific files
set -uo pipefail

PATTERNS=(
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'AGE-SECRET-KEY-1[0-9A-Z]+'
  'gh[pousr]_[A-Za-z0-9]{16,}'
  'github_pat_[A-Za-z0-9_]{20,}'
  'sk-[A-Za-z0-9_-]{20,}'
  'AKIA[0-9A-Z]{16}'
  'xox[baprs]-[A-Za-z0-9-]{10,}'
  'glpat-[A-Za-z0-9_-]{20,}'
  '(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{20,}'
  '(?i)\b(password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*["'"'"'][^"'"'"']{12,}["'"'"']'
)

if [ $# -gt 0 ]; then
  get() { cat "$@"; }
  SRC="$*"
else
  get() { git diff --cached -U0 --diff-filter=ACM | grep '^+' | sed 's/^+//'; }
  SRC="staged changes"
fi

content="$(get 2>/dev/null || true)"
[ -z "$content" ] && exit 0

rc=0
for p in "${PATTERNS[@]}"; do
  if hits=$(printf '%s' "$content" | grep -nPI -- "$p" 2>/dev/null); then
    [ -z "$hits" ] && continue
    printf '\033[31mSECRET-SHAPED MATCH\033[0m in %s\n  pattern: %s\n' "$SRC" "$p"
    # show the match, never the full line — the value itself must not be echoed
    printf '%s\n' "$hits" | head -3 | sed -E 's/(.{0,40}).*/  \1…/'
    rc=1
  fi
done

[ $rc -eq 0 ] && echo "scan-secrets: clean"
exit $rc
