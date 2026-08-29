#!/bin/bash
# Build the PyPEF Qt GUI into a standalone folder with PyInstaller (Linux/macOS).
#
# The full dependency collection lives in the portable qt_window.spec, which is
# the single source of truth shared with the Windows build. This script only
# installs the dependencies and invokes that spec.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

python -m pip install --upgrade pip pyinstaller
python -m pip install -e .[gui]

python -m PyInstaller --noconfirm qt_window.spec

echo
echo "Build complete. Run the GUI with:"
echo "  ./dist/qt_window/qt_window"
