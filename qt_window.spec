# -*- mode: python ; coding: utf-8 -*-
#
# Portable PyInstaller spec for the PyPEF Qt GUI (pypef/gui/qt_window.py).
# Tested with Python 3.12.
#
# The GUI imports pypef.main.run_main and runs it IN-PROCESS, so the frozen
# app needs pypef's *entire* runtime dependency tree (torch, transformers,
# peft, huggingface_hub, gpytorch, scikit-learn, biotite, biopython, ...),
# not just a handful of packages. We therefore collect_all() the heavy/lazy
# packages and copy_metadata() the ones that do importlib.metadata version
# lookups at runtime (transformers/peft/hf_hub in particular).
#
# Build (from the repo root) with:
#   pyinstaller --noconfirm qt_window.spec
#
# All paths are derived from SPECPATH so this spec is machine-independent.

import os
import sys
import glob
import ctypes.util

from PyInstaller.utils.hooks import collect_all, copy_metadata

REPO_ROOT = SPECPATH  # noqa: F821 (injected by PyInstaller)
ENTRY = os.path.join(REPO_ROOT, 'pypef', 'gui', 'qt_window.py')

datas = []
binaries = []
hiddenimports = ['docopt', 'pynvml']

# Packages needing full collection (data files, dylibs, and submodules that
# are imported lazily / dynamically and thus invisible to static analysis).
_collect_all_pkgs = [
    'pypef',            # bundled data: AAindex .txt, ProSST static .pt/.npy
    'torch',
    'torch_geometric',
    'transformers',
    'peft',
    'huggingface_hub',
    'gpytorch',
    'safetensors',
    'biotite',
    'sklearn',
    'adjustText',
    'schema',
    'Bio',              # biopython
]
for _pkg in _collect_all_pkgs:
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# Packages that call importlib.metadata.version(...) at runtime. Without their
# dist-info metadata the frozen app raises PackageNotFoundError on startup.
# For the big frameworks we copy metadata *recursively* to avoid PackageNotFoundError 
# that only surfaces on another machine / fresh install.
_metadata_recursive_pkgs = [
    'torch', 'transformers', 'peft', 'huggingface_hub',
    'gpytorch', 'scikit-learn', 'tokenizers', 'numpy',
]
for _pkg in _metadata_recursive_pkgs:
    try:
        datas += copy_metadata(_pkg, recursive=True)
    except TypeError:
        datas += copy_metadata(_pkg)  # older PyInstaller without recursive=
    except Exception:
        pass  # optional / not installed under that dist name

# A few leaf dists that may be imported indirectly and are not always reachable
# from the recursive roots above.
_metadata_pkgs = [
    'tqdm', 'safetensors', 'regex', 'requests', 'packaging', 'filelock',
    'pyyaml', 'fsspec', 'docopt-ng', 'nvidia-ml-py', 'sympy', 'networkx',
]
for _pkg in _metadata_pkgs:
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass  # optional / not installed under that dist name

# Some Linux/PySide6 setups fail at runtime with a missing libexpat. Bundle it
# if the system provides one — resolved portably, no hardcoded path.
_expat = ctypes.util.find_library('expat')
if _expat and os.path.exists(_expat):
    binaries += [(_expat, '.')]

# Bundle the *interpreter's own* OpenSSL (libssl/libcrypto). The frozen _ssl
# module is built against the env's OpenSSL, but PyInstaller may otherwise pick
# up an older system libcrypto lacking the required version symbols (e.g.
# "OPENSSL_3.3.0 not found"). Grabbing them from sys.prefix keeps _ssl / hf_hub
# / requests working. Portable: on Windows these globs simply match nothing.
for _libpat in ('libssl.so*', 'libcrypto.so*'):
    for _lib in glob.glob(os.path.join(sys.prefix, 'lib', _libpat)):
        binaries += [(_lib, '.')]


a = Analysis(
    [ENTRY],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='qt_window',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can corrupt torch/Qt shared libs; keep disabled
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='qt_window',
)
