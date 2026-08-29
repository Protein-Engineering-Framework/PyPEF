
# Run from repo root after `pip install .`:
#   Linux:   PYTHONPATH="" python -m pytest tests/test_pip_install.py -v -m pip_specific --log-cli-level=INFO --import-mode=importlib
#   Windows: $env:PYTHONPATH=""; python -m pytest tests\test_pip_install.py -v -m pip_specific --log-cli-level=INFO --import-mode=importlib
#
# Both PYTHONPATH="" and --import-mode=importlib are required:
#   - PYTHONPATH="" stops Python from adding an exported repo root to sys.path
#   - --import-mode=importlib stops pytest itself from adding rootdir to sys.path
# Without both, `import pypef` silently resolves to the source tree instead of
# the installed package, making all package-data tests meaningless.
#
# All tests are marked pip_specific and are excluded from the standard dev test run.

import subprocess
import sys
import os
from pathlib import Path
import shutil
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(*args) -> tuple[str, str, int]:
    """Execute a CLI command within the active Python environment's bin directory."""
    venv_bin_dir = Path(sys.executable).parent
    
    # Resolve command from current environment's bin/ directory if possible
    cmd = shutil.which(args[0], path=str(venv_bin_dir)) or shutil.which(args[0]) or args[0]
    
    # Ensure current venv's bin dir is prepended to PATH for spawned child processes
    env = os.environ.copy()
    p = f"{venv_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["PATH"] = p

    proc = subprocess.Popen([cmd, *args[1:]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out, err = proc.communicate()
    return out.decode(), err.decode(), proc.returncode


# ---------------------------------------------------------------------------
# Guard: must be testing the installed package, not the source tree
# ---------------------------------------------------------------------------

@pytest.mark.pip_specific
def test_pypef_loaded_from_installed_package():
    """pypef must resolve to site-packages, not to the source repo.

    If this test fails, pytest is silently importing from the source tree
    (typically because it added the repo root to sys.path). Fix: run pytest
    with --import-mode=importlib so sys.path is not modified.
    """
    import pypef
    pypef_file = Path(pypef.__file__).resolve()
    # This test file lives at tests/test_pip_install.py — repo root is one level up
    repo_root = Path(__file__).resolve().parent.parent
    assert not pypef_file.is_relative_to(repo_root), (
        f"pypef was imported from the source tree:\n  {pypef_file}\n"
        f"The tests are NOT exercising the installed package data.\n"
        f"Run pytest with --import-mode=importlib to fix this."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@pytest.mark.pip_specific
def test_cli_version_matches_package():
    """pypef --version returns the same string as pypef.__version__."""
    from pypef import __version__
    out, err, rc = _run("pypef", "--version")
    assert rc == 0, f"pypef --version failed:\n{err}"
    assert __version__ in out, f"Version mismatch: package={__version__!r}, CLI output={out!r}"


@pytest.mark.pip_specific
def test_cli_help():
    """pypef -h exits cleanly."""
    _out, err, rc = _run("pypef", "-h")
    assert rc == 0, f"pypef -h failed:\n{err}"


# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------

@pytest.mark.pip_specific
def test_import_core_modules():
    """All public sub-packages import without error."""
    import pypef
    import pypef.main
    import pypef.settings
    import pypef.ml.regression
    import pypef.dca.gremlin_inference
    import pypef.hybrid.hybrid_model
    import pypef.plm.inference
    import pypef.utils.variant_data
    import pypef.utils.helpers
    import pypef.gaussian_process.gauss_opt


@pytest.mark.pip_specific
def test_import_hybrid_model_class():
    from pypef.hybrid.hybrid_model import (
        DCALLMHybridModel, parse_llm_flag, setup_llm_input,
        performance_ls_ts, predict_ps, predict_directed_evolution,
        get_model_and_type, save_model_to_dict_pickle,
    )


@pytest.mark.pip_specific
def test_import_plm_modules():
    from pypef.plm.inference import esm_setup, prosst_setup, tokenize_sequences, get_plm_embeddings
    from pypef.plm.esm_lora_tune import get_esm_models
    from pypef.plm.prosst_lora_tune import get_prosst_models


# ---------------------------------------------------------------------------
# Package-data: AAindex files
# ---------------------------------------------------------------------------

@pytest.mark.pip_specific
def test_aaindex_package_data_accessible():
    """AAindex txt files are present in the installed package-data directory."""
    from pypef.ml.regression import path_aaindex_dir
    aaidx_dir = Path(path_aaindex_dir())
    assert aaidx_dir.is_dir(), f"AAindex directory not found: {aaidx_dir}"
    txt_files = list(aaidx_dir.glob("*.txt"))
    assert len(txt_files) > 500, (
        f"Expected >500 AAindex .txt files, found {len(txt_files)} in {aaidx_dir}"
    )


@pytest.mark.pip_specific
def test_aaindex_refined_cluster_indices_accessible():
    """Refined cluster index files shipped with the package are readable."""
    from pypef.ml.regression import path_aaindex_dir
    cluster_dir = Path(path_aaindex_dir()) / "Refined_cluster_indices_r0.93_r0.97"
    assert cluster_dir.is_dir(), f"Cluster index directory not found: {cluster_dir}"
    cluster_files = list(cluster_dir.glob("*.txt"))
    assert len(cluster_files) > 0, f"No cluster index .txt files in {cluster_dir}"


@pytest.mark.pip_specific
def test_aaindex_encoding_loads():
    """AAIndexEncoding can be instantiated (reads an AAindex file from package-data)."""
    from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path
    # Use a known bundled AAindex file as a smoke test
    enc = AAIndexEncoding(aaindex_file=full_aaidx_txt_path("ANDN920101.txt"))
    assert enc.dictionary is not None and len(enc.dictionary) > 0


# ---------------------------------------------------------------------------
# Package-data: ProSST static model files (.pt / .npy)
# ---------------------------------------------------------------------------

@pytest.mark.pip_specific
def test_prosst_static_files_exist():
    """AE_CPU.pt, AE.pt, and 2048_kmeans_cluster_centers.npy are present in the installed package."""
    import pypef.plm.prosst_structure.quantizer as _q_mod
    static_dir = Path(_q_mod.__file__).parent / "static"
    assert static_dir.is_dir(), f"ProSST static dir not found: {static_dir}"
    for fname in ("AE_CPU.pt", "AE.pt", "2048_kmeans_cluster_centers.npy"):
        fpath = static_dir / fname
        assert fpath.exists(), f"Missing ProSST static file: {fpath}"
        assert fpath.stat().st_size > 0, f"ProSST static file is empty: {fpath}"


@pytest.mark.pip_specific
@pytest.mark.parametrize("fname", ["AE_CPU.pt", "AE.pt"])
def test_prosst_static_ae_loads(fname):
    """Both AE_CPU.pt (CPU weights) and AE.pt (GPU weights) can be loaded via
    torch.load with map_location='cpu', catching file truncation or corruption.
    AE.pt is normally only used on CUDA devices, but we always load it to cpu
    here so the test runs without a GPU."""
    import torch
    import pypef.plm.prosst_structure.quantizer as _q_mod
    static_dir = Path(_q_mod.__file__).parent / "static"
    ae_path = static_dir / fname
    # weights_only=True is intentionally used here — the .pt files contain only
    # state dicts (plain tensors), so this is safe and avoids arbitrary code exec.
    state_dict = torch.load(ae_path, map_location="cpu", weights_only=True)
    assert isinstance(state_dict, dict), f"{fname}: expected a state-dict, got {type(state_dict)}"
    assert len(state_dict) > 0, f"{fname}: state-dict is empty"


@pytest.mark.pip_specific
def test_prosst_static_cluster_centers_loads():
    """2048_kmeans_cluster_centers.npy can be loaded with numpy from package-data."""
    import numpy as np
    import pypef.plm.prosst_structure.quantizer as _q_mod
    static_dir = Path(_q_mod.__file__).parent / "static"
    centers_path = static_dir / "2048_kmeans_cluster_centers.npy"
    centers = np.load(str(centers_path))
    assert centers.ndim == 2, f"Expected 2-D cluster centers array, got shape {centers.shape}"
    assert centers.shape[0] == 2048, f"Expected 2048 cluster centers, got {centers.shape[0]}"


# ---------------------------------------------------------------------------
# Pickle round-trip sanity check (no external data needed)
# ---------------------------------------------------------------------------

@pytest.mark.pip_specific
def test_pickle_roundtrip_hybrid_model_class(tmp_path):
    """DCALLMHybridModel and GREMLIN instances survive a pickle round-trip.

    Uses object.__new__ to bypass __init__ (avoids training overhead) so the
    test only checks that the installed package's module paths are correct for
    pickle deserialization — the most common breakage point when classes are
    renamed or moved after a pip install.
    """
    import pickle
    from pypef.hybrid.hybrid_model import DCALLMHybridModel, check_model_type
    from pypef.dca.gremlin_inference import GREMLIN

    for cls, expected_type in [(DCALLMHybridModel, 'Hybrid'), (GREMLIN, 'GREMLIN')]:
        bare = object.__new__(cls)
        # DCALLMHybridModel.check_model_type needs llm_keys to exist
        if cls is DCALLMHybridModel:
            bare.llm_keys = None
            bare.llm_data = None

        pkl_path = tmp_path / f"{cls.__name__}.pkl"
        model_dict = {'model': bare, 'model_type': expected_type}
        with open(pkl_path, "wb") as fh:
            pickle.dump(model_dict, fh)
        with open(pkl_path, "rb") as fh:
            loaded = pickle.load(fh)

        assert isinstance(loaded['model'], cls), (
            f"Unpickled object is {type(loaded['model'])!r}, expected {cls!r}. "
            "This means the class module path changed — installed package may be stale."
        )
        assert check_model_type(loaded) == expected_type


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "pip_specific"])
