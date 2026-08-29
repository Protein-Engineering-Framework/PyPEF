# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import platform
import random
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import set_seed, AutoConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as ts_logging
ts_logging.set_verbosity_error()

import logging
logger = logging.getLogger('pypef.plm.utils')


warn_counter = 0


def _set_seeds(seed: int, use_deterministic_algorithms: bool = True):
    global warn_counter
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if use_deterministic_algorithms:
        # For cross-machine consistency, before run:
        # export CUBLAS_WORKSPACE_CONFIG=:4096:8
        # or
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
        rw = False
        try:
            if not os.environ["CUBLAS_WORKSPACE_CONFIG"]:
                rw = True
        except KeyError:
            rw = True
        if rw:
            if warn_counter == 0:
                logger.warning(
                    "'CUBLAS_WORKSPACE_CONFIG' not set, "
                    "will likely face a torch RuntimeError. "
                    "Make sure to e.g. run 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' (Linux/Mac) "
                    "or '$env:CUBLAS_WORKSPACE_CONFIG=\":4096:8\"' (Windows PowerShell) "
                    "before running with set seeds and determinism."
                )
            warn_counter += 1
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hybrid_corr_mse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    method: str = "spearman",
    tau: float = 0.1,
    alpha: float | None = None,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Hybrid differentiable loss combining a ranking/correlation component and MSE.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        method: Loss method. Base methods: "spearman" (soft-ranking), "pearson",
            "listMLE" (ListMLE ranking), "pairwise-margin" (pairwise hinge).
            Append "-hybrid" for 50/50 mix with MSE.
        tau: Temperature for soft-rank approximation (spearman only).
        alpha: Weight for ranking/correlation loss component.
            (1 - alpha) weights MSE. None auto-sets: 1.0 for base, 0.5 for hybrid.
        margin: Margin for pairwise-margin loss (default 1.0).
    """
    base_methods = ["spearman", "pearson", "listMLE", "pairwise-margin"]
    hybrid_methods = [f"{m}-hybrid" for m in base_methods]
    if alpha is None:
        if method in base_methods:
            alpha = 1.0
        elif method in hybrid_methods:
            alpha = 0.5
        else:
            raise RuntimeError(
                "Alpha parameter for loss function is not defined. Define alpha or a method "
                f"from within {base_methods + hybrid_methods}."
            )

    if method.startswith("listMLE"):
        _, indices = torch.sort(y_true, descending=True, dim=-1)
        y_pred_sorted = y_pred[indices]
        log_cumsum = torch.logcumsumexp(y_pred_sorted.flip(-1), dim=-1).flip(-1)
        loss_corr = -(y_pred_sorted - log_cumsum).mean()
    elif method.startswith("pairwise-margin"):
        y_true_col = y_true.unsqueeze(-1)
        y_pred_col = y_pred.unsqueeze(-1)
        diff_true = y_true_col - y_true_col.transpose(-1, -2)
        diff_pred = y_pred_col - y_pred_col.transpose(-1, -2)
        mask = (diff_true > 0).float()
        losses = torch.relu(margin - diff_pred) * mask
        loss_corr = losses.sum() / mask.sum().clamp(min=1)
    elif method.startswith("spearman"):
        def get_soft_ranks(z, t):
            z_expanded = z.unsqueeze(-1)
            diff = z_expanded - z_expanded.transpose(-1, -2)
            P = torch.sigmoid(diff / t)
            return P.sum(dim=-1) + 0.5
        rx = get_soft_ranks(y_true, tau)
        ry = get_soft_ranks(y_pred, tau)
        rx_c = rx - rx.mean(dim=-1, keepdim=True)
        ry_c = ry - ry.mean(dim=-1, keepdim=True)
        loss_corr = -F.cosine_similarity(rx_c, ry_c, dim=-1).mean()
    elif method.startswith("pearson"):
        rx_c = y_true - y_true.mean(dim=-1, keepdim=True)
        ry_c = y_pred - y_pred.mean(dim=-1, keepdim=True)
        loss_corr = -F.cosine_similarity(rx_c, ry_c, dim=-1).mean()
    else:
        raise ValueError(f"Method {method} not supported.")

    loss_mse = F.mse_loss(y_pred, y_true)
    return (alpha * loss_corr) + ((1 - alpha) * loss_mse)


def get_batches(
        a, 
        dtype, 
        batch_size=5,
        keep_remaining=False, 
        verbose: bool = False
) -> list | list[np.ndarray]:
    a = np.asarray(a, dtype=dtype)
    a_remaining = None
    orig_shape = np.shape(a)
    remaining = len(a) % batch_size
    if remaining != 0:
        if len(a) > batch_size:
            # Capture remaining elements before truncating 'a'
            a_remaining = a[-remaining:]
            a = a[:-remaining]
        else:
            logger.info(f"Batch size greater than or equal to total array length: "
                        f"returning full array (of shape: {np.shape(a)})...")
            if keep_remaining:
                return [a]
            else:
                raise RuntimeError(
                    "To return the full list of arrays that is smaller than the batch "
                    "size, set `keep_remaining=True`."
                )
    if len(orig_shape) == 2:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size, np.shape(a)[1])
    else: # elif len(orig_shape) == 1:
        a = a.reshape(np.shape(a)[0] // batch_size, batch_size)
    new_shape = np.shape(a)
    if verbose:
        logger.info(f'Batch reshape:{orig_shape} -> {new_shape} (dropped {remaining})...')
    if keep_remaining: 
        if a_remaining is not None:
            logger.info(f'Appending remaining to collected batches as last batch '
                        f'(the resulting inhomogenous list shape is '
                        f'{np.shape(a)} + {np.shape(a_remaining)} = ('
                        f'{np.shape(a)[0] + 1}, *, {np.shape(a)[-1]}))...')
            a = a.tolist()
            a.append(a_remaining)
            a = [np.asarray(it) for it in a]
    return a


def parse_mut_position(mut_string: str | list | tuple, mut_separator: str = '/') -> list[int]:
    """
    Parses single or multi-substitution mutations to extract 1-indexed 
    positions and convert them to 0-indexed positions.
    
    Supports:
      - Slash-separated strings: 'A123C/F429G'
      - Iterables/Lists of strings: ['A123C', 'F429G']
      - Single point mutations: 'M1A'
      - Wild-type flags: 'WT' or NaN
    """
    # Standardize fallbacks for Wild-Type reference sequences
    if mut_string is None or str(mut_string).lower() in ["wt", "nan", ""]:
        return [0] 
        
    # Extract substitution tokens depending on input type
    if isinstance(mut_string, (list, tuple)):
        substitutions = [str(s).strip() for s in mut_string]
    else:
        substitutions = [s.strip() for s in str(mut_string).split(mut_separator)]
        
    positions = []
    for sub in substitutions:
        # Regex captures the digits representing the sequence position
        match = re.search(r'\d+', sub)
        if match:
            positions.append(int(match.group()) - 1)
        else:
            raise ValueError(f"Could not parse position from mutation component: '{sub}' inside '{mut_string}'")
            
    return positions


def extract_mean_or_pos_embeddings(
    full_sequence_embeddings: torch.Tensor, 
    mode: str = "mean",
    mutation_strings: list | None = None, 
    verbose: bool = False
) -> torch.Tensor:
    assert mode in ["positional", "mean"], "mode must be either 'positional' or 'mean'"
    
    num_seqs, seq_len, dim = full_sequence_embeddings.shape
    
    if mode == "mean":
        if verbose:
            logger.info(f"Global pooling: compressing [{num_seqs}, {seq_len}, {dim}] via mean(dim=1)")
        return full_sequence_embeddings.mean(dim=1)
        
    elif mode == "positional":
        if verbose:
            logger.info(f"Site-specific pooling: extracting indices for {num_seqs} variants")
        
        # 1. Parse all mutation positions out of the strings
        mut_indices = [parse_mut_position(mut) for mut in mutation_strings]
        mut_indices_tensor = torch.tensor(mut_indices, dtype=torch.long, device=full_sequence_embeddings.device)
        
        # Safety check: ensure no parsed index falls outside your sequence length
        if (mut_indices_tensor >= seq_len).any() or (mut_indices_tensor < 0).any():
            max_idx = mut_indices_tensor.max().item()
            raise IndexError(
                f"Parsed a mutation index ({max_idx}) that exceeds the sequence "
                f"length of your embedding tensor ({seq_len-1}). Check if your sequence length "
                f"matches the reference used."
            )
            
        # 2. Advanced matrix indexing
        # Creates an array [0, 1, 2, ..., Num_Sequences-1] to coordinate the rows
        batch_indices = torch.arange(num_seqs, device=full_sequence_embeddings.device)
        
        # Pulls out exactly full_sequence_embeddings[i, mut_indices[i], :] for every row i
        site_embeddings = full_sequence_embeddings[batch_indices, mut_indices_tensor]
        return site_embeddings


def get_default_cache_dir():
    """
    Detect OS and set Hugging Face transformers cache directory accordingly
    """
    system = platform.system()
    if system == "Windows":
        return os.path.join(
            os.environ.get("USERPROFILE", ""), ".cache",
            "huggingface", "hub"
        )
    elif system == "Darwin":
        return os.path.expanduser("~/.cache/huggingface/hub")
    else: # Assume Linux or other Unix-like systems
        return os.path.expanduser("~/.cache/huggingface/hub")


def is_model_cached(repo_id: str, cache_dir: str):
    """
    Check if the required model and tokenizer files are cached locally.
    """
    snapshot_dir = None
    ref_file = None
    if os.path.isdir(cache_dir):
        ref_file = os.path.join(
            cache_dir, f'models--{repo_id.replace("/", "--")}', 'refs', 'main'
        )
        if os.path.isfile(ref_file):
            with open(ref_file, 'r') as fh:
                t = fh.readlines()  # Getting hash contents
            ref = t[0].strip()
        else:
            return False, snapshot_dir, ref_file
        snapshot_dir = os.path.join(
            cache_dir, f'models--{repo_id.replace("/", "--")}', 'snapshots', ref
        )
        if os.path.isdir(snapshot_dir):
            return True, snapshot_dir, ref_file
        else:
            return False, None, ref_file
    else:
        return False, snapshot_dir, ref_file


def load_model_and_tokenizer(
    model_name: str, 
    cache_dir: str | os.PathLike | None = None, 
    model_loader=None, 
    tokenizer_loader=None,
    revision: str | None = None
):
    """
    Enhanced loader that bypasses broken Windows symlinks by manually 
    injecting weights from the HF blob store if ProSST is detected.
    """
    if cache_dir is None:
        # Assuming you have a helper for this, or use default
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
    if model_loader is None:
        model_loader = AutoModelForMaskedLM
    if tokenizer_loader is None:
        tokenizer_loader = AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", None)

    # Check if model is cached locally
    # Note: Even if exists=True, Windows symlinks might be broken pointers
    exists, snapshot_dir, _ = is_model_cached(model_name, cache_dir)
    if exists:
        logger.info(f"Model snapshot extists at {snapshot_dir}...")
    is_windows = platform.system().lower() == "windows"

        # Loading the model
    logger.info(f"Loading model architecture for {model_name}...")
    
    # We first try a standard load. 
    # On Windows, we use the model_name (repo_id) rather than snapshot_dir 
    # to let HF attempt its internal resolution.
    # Update same for Linux, just use model name, so commented: 
    # load_path = model_name if is_windows else (snapshot_dir if exists else model_name)
    load_path = model_name
    use_local_files = exists

    try:
        config = AutoConfig.from_pretrained(
            load_path, 
            trust_remote_code=True, 
            revision=revision, 
            cache_dir=cache_dir,
            local_files_only=use_local_files,
            token=hf_token
        )
    except Exception as e:
        logger.warning(f"Failed to load config for {model_name} locally ({e}). Forcing online repair...")
        use_local_files = False
        config = AutoConfig.from_pretrained(
            load_path, 
            trust_remote_code=True, 
            revision=revision, 
            cache_dir=cache_dir,
            local_files_only=False,
            force_download=True,  # Nuke corrupted config.json
            token=hf_token
        )

    # ProSST (with older transformers versions) needs an explicitly separate,
    # untied decoder head, which is then weight-copied from the embeddings below.
    # This must NOT be applied to ESM: ESM ships tied weights only, so untying
    # would leave its lm_head.decoder randomly initialized, destroying zero-shot
    # scoring (near-random Spearman).
    if "prosst" in model_name.lower():
        # Force the architecture to create a separate decoder layer
        config.tie_word_embeddings = False

    # Common arguments for Model and Tokenizer
    load_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "revision": revision,
        "local_files_only": use_local_files, # Will be False if repair was needed
        "token": hf_token,
        "config": config
    }

    # Add the config to your load_kwargs
    load_kwargs["config"] = config
    
    try:
        # Attempt 1: Prioritize SafeTensors
        model = model_loader.from_pretrained(
            load_path,
            use_safetensors=True, 
            **load_kwargs
        )
    except Exception as e:
        # If it fails, check if we need to drop offline mode and go online
        if use_local_files:
            logger.warning(f"Local weight load failed ({type(e).__name__}). Falling back to online mode...")
            load_kwargs["local_files_only"] = False
            
        try:
            # Attempt 2: Try online with SafeTensors
            model = model_loader.from_pretrained(
                load_path,
                use_safetensors=True,
                **load_kwargs
            )
        except Exception as e2:
            # Attempt 3: If it strictly complains about missing SafeTensors, drop to .bin
            if "safetensors" in str(e2).lower():
                logger.info(f"{model_name} lacks SafeTensors format. Falling back to legacy .bin weights...")
                model = model_loader.from_pretrained(
                    load_path,
                    use_safetensors=False, # Explicitly disable to stop background ghost threads
                    **load_kwargs
                )
            else:
                # If it's a completely different error (like out of memory), crash normally
                raise e2

    # THE WINDOWS SYMLINK BYPASS (Specific for ProSST)
    # If the weights didn't load, they are forced here
    if is_windows and "prosst" in model_name.lower():
        try:
            logger.info("Windows detected: Forcing manual weight injection from blobs...")
            
            # This identifies the actual large binary file in the blobs folder
            # NOTE: If ProSST ever becomes a 3B+ parameter model, this will crash because 
            # of sharding. Currently safe only for single-file safetensor models (e.g., 
            # model-00001-of-00002.safetensors).
            real_weight_path = hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",
                cache_dir=cache_dir,
                local_files_only=True
            )
            
            # Load weights manually
            state_dict = load_file(real_weight_path)

            # Prepare the state_dict (Inject the missing key if it's not there)
            if 'cls.predictions.decoder.weight' not in state_dict:
                logger.info("Injecting cloned embedding weights into model state dictionary for decoder...")
                state_dict['cls.predictions.decoder.weight'] = state_dict['prosst.embeddings.word_embeddings.weight'].clone()
        
            # Apply the weights to the model
            # Use strict=False so it doesn't crash on minor metadata mismatches
            msg = model.load_state_dict(state_dict, strict=False) 
            #assert (model.cls.predictions.decoder.weight.sum().item() == 
            #        model.prosst.embeddings.word_embeddings.weight.sum().item())
            
            if len(msg.missing_keys) > 0:
                logger.warning(f"Weights injected, but some keys still missing: {msg.missing_keys}")
                
        except Exception as e:
            logger.error(f"Manual weight injection failed: {e}")

    # GLOBAL PROSST PARITY (Runs on Linux/GitHub and Windows)
    # This ensures that even on Linux, the decoder is a bit-perfect clone of the embeddings.
    if "prosst" in model_name.lower():
        logger.info("Enforcing cross-platform weight parity for ProSST...")
        
        # Ensure the decoder layer exists (if not loaded via state_dict)
        if not hasattr(model.cls.predictions, 'decoder'):
            logger.info("Manually attaching linear decoder head...")
            model.cls.predictions.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        with torch.no_grad():
            embedding_weight = model.prosst.embeddings.word_embeddings.weight
            model.cls.predictions.decoder.weight.copy_(embedding_weight)
        
        # Do NOT call model.tie_weights() here, or PyTorch will turn them back into pointers

    # Loading the tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = tokenizer_loader.from_pretrained(
            load_path,
            use_fast=False, # Avoids the sentencepiece/tiktoken conversion error
            **load_kwargs
        )
    except Exception as e:
        logger.warning(f"Tokenizer load failed with use_fast=False, trying default: {e}")
        tokenizer = tokenizer_loader.from_pretrained(
            load_path,
            **load_kwargs
        )

    logger.info(f"Successfully finished loading {model_name}")
    return model, tokenizer


def load_model_and_tokenizer__(
        model_name: str, 
        cache_dir: str | os.PathLike | None = None, 
        model_loader=None, 
        tokenizer_loader=None,
        revision: str | None = None
):
    """
    Load the model and tokenizer from cache directory. Downloads to cache if not present.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if model_loader is None:
        model_loader = AutoModelForMaskedLM
    if tokenizer_loader is None:
        tokenizer_loader = AutoTokenizer
    exists, snapshot_dir, ref_file = is_model_cached(model_name, cache_dir)

    # Check for Windows to avoid the 'Snapshot Path' trap
    is_windows = platform.system().lower() == "windows"
    if exists and not is_windows:
        try:
            logger.info(f"Loading model and tokenizer from cache {snapshot_dir}...")
            model = model_loader.from_pretrained(
                snapshot_dir, trust_remote_code=True, revision=revision, #use_safetensors=True
            )
            tokenizer = tokenizer_loader.from_pretrained(
                snapshot_dir, trust_remote_code=True, revision=revision, #use_safetensors=True
            )
        except OSError as e:
            logger.warning(f"Snapshot load failed: {e}. Falling back to repo_id load.")
            exists = False # Trigger the fallback below
    if not exists or is_windows:
        logger.info(f"Loading via repo_id '{model_name}' (safe mode for Windows/missing cache)...")
        # Using the model_name instead of snapshot_dir allows HF to resolve 
        # the actual weight blobs even if the symlink structure is broken.
        model = model_loader.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True, 
            revision=revision,
            local_files_only=exists, # Use cache if it's there, but let HF handle the mapping
            #use_safetensors=True
        )
        tokenizer = tokenizer_loader.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True, 
            revision=revision,
            local_files_only=exists,
            #use_safetensors=True
        )
    logger.info("Model and tokenizer loaded successfully...")
    return model, tokenizer


class KermutFeaturizer:
    def __init__(self, aa_cond_probs, struct_coords, wt_seq):
        """
        Direct API connector for Kermut without disk writes.
        
        Args:
            aa_cond_probs (Tensor): Shape [L, 20] (conditional probabilities)
            struct_coords (NDArray/Tensor): Shape [L, 3] (3D coordinates)
            wt_seq (str): The raw wild-type sequence string (length L)
        """
        self.wt_seq = wt_seq
        self.device = aa_cond_probs.device
        
        # Ensure everything is a PyTorch tensor on the same device for fast kernel math
        self.aa_cond_probs = aa_cond_probs.float()
        self.struct_coords = torch.tensor(struct_coords, device=self.device).float() if isinstance(struct_coords, np.ndarray) else struct_coords.float()
        
        # Standard Amino Acid alphabet map used by Kermut/ProteinMPNN
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.aa_alphabet)}

    def featurize_variant(self, mutation_str):
        """
        Converts a mutation string like 'M1A' into its respective matrix features.
        Supports single or multiple mutations.
        """
        if mutation_str == "WT" or mutation_str == "wildtype":
            # For WT, return dummy/neutral values or standard base tokens
            return {"prob_vectors": torch.zeros((1, 20), device=self.device), 
                    "coords": torch.zeros((1, 3), device=self.device)}
        
        mutations = mutation_str.split(":") # Handles multi-mutants separated by colons
        prob_vectors = []
        coords = []
        
        for mut in mutations:
            wt_aa = mut[0]
            pos = int(mut[1:-1]) - 1 # Convert 1-based PDB index to 0-based matrix index
            mut_aa = mut[-1]
            
            # Fetch the precise row belonging to this residue position
            prob_vectors.append(self.aa_cond_probs[pos])
            coords.append(self.struct_coords[pos])
            
        return {
            "prob_vectors": torch.stack(prob_vectors), # Shape [num_mutations, 20]
            "coords": torch.stack(coords)             # Shape [num_mutations, 3]
        }