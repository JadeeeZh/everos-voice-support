#!/usr/bin/env bash
# Vendor tau2-bench (pinned) and sync the demo venv.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TAU2_REPO="https://github.com/sierra-research/tau2-bench"
TAU2_SHA="a2c024725189473d2d7cea3a5cfdbcc67478e41f" # main, 2026-08-18 (post v1.0.1)
VENDOR="$HERE/vendor/tau2-bench"

if [ ! -d "$VENDOR/.git" ]; then
    echo "Cloning tau2-bench into vendor/ ..."
    git clone "$TAU2_REPO" "$VENDOR"
fi
git -C "$VENDOR" fetch --quiet origin "$TAU2_SHA" 2>/dev/null || git -C "$VENDOR" fetch --quiet origin
git -C "$VENDOR" checkout --quiet "$TAU2_SHA"
echo "tau2-bench pinned at $(git -C "$VENDOR" rev-parse --short HEAD)"

# EverOS itself: a source checkout as the parent directory wins; otherwise the
# released CLI (see voicedemo.config.everos_command for the same rule).
if [ -f "$HERE/../pyproject.toml" ] && grep -q 'name = "everos"' "$HERE/../pyproject.toml"; then
    echo "EverOS source checkout found at $HERE/.. — the server will run from it"
elif command -v everos >/dev/null 2>&1; then
    echo "everos CLI found: $(command -v everos)"
else
    echo "Installing the released everos CLI (uv tool install everos) ..."
    uv tool install everos
fi

cd "$HERE"
uv sync
echo "Done. Try: make smoke"
