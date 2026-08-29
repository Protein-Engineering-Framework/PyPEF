#!/usr/bin/env bash
# test_pip_install.sh — Verify that packaging works and that the installed
# pypef package imports and loads its bundled data files correctly.
# Runs against every Python version declared in pyproject.toml (3.10–3.14).
#
# Packaging integrity (that the .pt/.npy/AAindex data files actually ship) is
# validated by scripts/CLI/check_wheel_packaging.py, which is the single source of
# truth for the required-files list and is also run in CI (ci.yml). See that
# script for why it builds from a git-less / MANIFEST-less copy.
#
# Usage:
#   bash scripts/CLI/test_pip_install.sh [--editable] [--keep-venv] [--python 3.11]
#
# Options:
#   --editable        Install with `pip install -e .` instead of a normal
#                     install. NOTE: editable installs reference the source
#                     tree in place, so they cannot validate data-file
#                     packaging — the packaging-integrity check is skipped in
#                     this mode and only import/runtime behaviour is checked.
#   --keep-venv       Do not delete temporary dirs after the run (handy for
#                     manual post-mortem inspection).
#   --python VERSION  Only test a single Python version (e.g. 3.11).
#
# Exit codes:
#   0   All checks passed for every tested Python version
#   1   One or more checks failed (skipped versions are not failures)

set -euo pipefail

# ── Versions declared in pyproject.toml ───────────────────────────────────
ALL_VERSIONS=(3.10 3.11 3.12 3.13 3.14)

# ── Parse flags ────────────────────────────────────────────────────────────
# Script lives in scripts/CLI/, so the repo root is two levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG_CHECK="$REPO_ROOT/scripts/CLI/check_wheel_packaging.py"
EDITABLE=0
KEEP_VENV=0
FILTER_VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --editable)   EDITABLE=1; shift ;;
    --keep-venv)  KEEP_VENV=1; shift ;;
    --python)     FILTER_VERSION="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$FILTER_VERSION" ]]; then
  VERSIONS=("$FILTER_VERSION")
else
  VERSIONS=("${ALL_VERSIONS[@]}")
fi

# ── State ──────────────────────────────────────────────────────────────────
PASSED=()
FAILED=()
SKIPPED=()
CLEAN_DIRS=()  # all temp dirs tracked for cleanup
GITLESS_SRC="" # populated by make_gitless_source_copy

cleanup() {
  if [[ "$KEEP_VENV" -eq 0 ]]; then
    for d in "${CLEAN_DIRS[@]+"${CLEAN_DIRS[@]}"}"; do
      [[ -d "$d" ]] && rm -rf "$d"
    done
  else
    for d in "${CLEAN_DIRS[@]+"${CLEAN_DIRS[@]}"}"; do
      [[ -d "$d" ]] && echo "  Keeping temp dir: $d"
    done
  fi
}
trap cleanup EXIT

# ── Build a git-less, MANIFEST-less copy of the minimal source ─────────────
# Used to install the package for the runtime tests without letting the git
# file-finder or MANIFEST.in mask a broken package-data path. (The dedicated
# packaging-integrity check in scripts/CLI/check_wheel_packaging.py makes its own copy.)
make_gitless_source_copy() {
  GITLESS_SRC="$(mktemp -d -t pypef_gitless_src_XXXXXX)"
  CLEAN_DIRS+=("$GITLESS_SRC")

  cp "$REPO_ROOT/pyproject.toml" "$GITLESS_SRC/"
  [[ -f "$REPO_ROOT/README.md" ]]        && cp "$REPO_ROOT/README.md" "$GITLESS_SRC/"
  [[ -f "$REPO_ROOT/requirements.txt" ]] && cp "$REPO_ROOT/requirements.txt" "$GITLESS_SRC/"
  # NOTE: MANIFEST.in is intentionally NOT copied.

  cp -r "$REPO_ROOT/pypef" "$GITLESS_SRC/pypef"

  find "$GITLESS_SRC" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  find "$GITLESS_SRC" -type d -name '*.egg-info'  -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$GITLESS_SRC/.git" "$GITLESS_SRC/build" "$GITLESS_SRC/MANIFEST.in"

  echo "  Git-less / MANIFEST-less source copy: $GITLESS_SRC"
}

run_for_version() {
  local pyver="$1"
  local pybin=""

  # Look for python3.X, or pyenv shims
  for candidate in "python${pyver}" "python3.${pyver##3.}" "python3" "python"; do
    if command -v "$candidate" &>/dev/null; then
      local actual
      actual="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "$actual" == "$pyver" ]]; then
        pybin="$candidate"
        break
      fi
    fi
  done

  if [[ -z "$pybin" ]]; then
    echo "  [SKIP] python${pyver} not found on PATH"
    SKIPPED+=("$pyver")
    return 0
  fi

  local venv_dir
  venv_dir="$(mktemp -d -t "pypef_pip_test_${pyver}_XXXXXX")"
  CLEAN_DIRS+=("$venv_dir")

  echo ""
  echo "──────────────────────────────────────────────────"
  echo "  Python ${pyver}  (${pybin})  →  ${venv_dir}"
  echo "──────────────────────────────────────────────────"

  # Use full venv binary paths — no `source activate` needed.
  # This avoids the failure mode where venv creation partially succeeds
  # (directory created but activate missing) and `source` silently falls
  # through to whatever python/pip is on PATH (e.g. a conda base env).
  local PY="$venv_dir/bin/python"
  local PIP="$venv_dir/bin/pip"

  echo "  [0/5] Creating venv..."
  if ! "$pybin" -m venv "$venv_dir"; then
    echo "  [FAIL] python${pyver} -m venv failed."
    echo "         On Debian/Ubuntu try: sudo apt install python${pyver}-venv"
    FAILED+=("$pyver")
    return 1
  fi

  # Guard: venv must have produced a usable interpreter
  if [[ ! -x "$PY" ]]; then
    echo "  [FAIL] venv created but $PY is missing — venv is broken."
    FAILED+=("$pyver")
    return 1
  fi

  # Verify the interpreter inside the venv really is the requested version
  local venv_ver
  venv_ver="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$venv_ver" != "$pyver" ]]; then
    echo "  [FAIL] venv interpreter is Python ${venv_ver}, expected ${pyver}."
    FAILED+=("$pyver")
    return 1
  fi

  local rc=0

  echo "  [1/5] Upgrading pip / setuptools / wheel..."
  "$PIP" install --quiet --upgrade pip setuptools wheel

  if [[ "$EDITABLE" -eq 1 ]]; then
    echo "  [2/5] Packaging integrity — SKIPPED (editable can't test packaging)"
    echo "  [3/5] Installing pypef (editable, from source tree)..."
    "$PIP" install -e "$REPO_ROOT"
  else
    echo "  [2/5] Checking package-data integrity (shared checker)..."
    if ! "$PY" "$PKG_CHECK"; then
      echo "  [FAIL] Packaging integrity check failed for Python ${pyver}."
      FAILED+=("$pyver")
      return 1
    fi

    echo "  [3/5] Installing pypef from git-less copy (+ dependencies)..."
    "$PIP" install "$GITLESS_SRC"
  fi

  echo "  [4/5] Installing pytest..."
  "$PIP" install --quiet pytest

  echo "  [5/5] Running pip_specific tests..."
  # PYTHONPATH is cleared and --import-mode=importlib is set so that
  # `import pypef` resolves to the installed package in the venv's
  # site-packages, not to the source tree (which happens when PYTHONPATH
  # contains the repo root or when pytest adds rootdir to sys.path).
  PYTHONPATH="" "$PY" -m pytest \
    "$REPO_ROOT/tests/test_pip_install.py" \
    "$REPO_ROOT/tests/cli/test_version.py" \
    -v \
    -m "pip_specific" \
    --log-cli-level=INFO \
    --tb=short \
    --import-mode=importlib || rc=$?

  if [[ $rc -eq 0 ]]; then
    PASSED+=("$pyver")
    echo "  → PASSED (Python ${pyver})"
  else
    FAILED+=("$pyver")
    echo "  → FAILED (Python ${pyver})"
  fi
  return $rc
}

# ── Main ───────────────────────────────────────────────────────────────────
echo "=================================================="
echo "  PyPEF pip-install smoke test"
echo "  Repo:    $REPO_ROOT"
echo "  Mode:    $([ "$EDITABLE" -eq 1 ] && echo editable || echo normal)"
echo "  Testing: ${VERSIONS[*]}"
echo "=================================================="

if [[ ! -f "$PKG_CHECK" ]]; then
  echo "ERROR: packaging checker not found at $PKG_CHECK" >&2
  exit 1
fi

echo ""
echo "Preparing git-less source copy for install..."
make_gitless_source_copy

ANY_FAILED=0
for ver in "${VERSIONS[@]}"; do
  run_for_version "$ver" || ANY_FAILED=1
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Results"
echo "=================================================="
[[ ${#PASSED[@]}  -gt 0 ]] && echo "  PASSED:  ${PASSED[*]}"
[[ ${#SKIPPED[@]} -gt 0 ]] && echo "  SKIPPED: ${SKIPPED[*]}  (interpreter not found)"
[[ ${#FAILED[@]}  -gt 0 ]] && echo "  FAILED:  ${FAILED[*]}"

if [[ $ANY_FAILED -ne 0 ]]; then
  echo ""
  echo "  One or more versions FAILED."
  exit 1
fi

if [[ ${#PASSED[@]} -eq 0 ]]; then
  echo ""
  echo "  WARNING: no Python interpreters were found — nothing was tested."
  exit 1
fi

echo ""
echo "  All tested versions passed."
exit 0
