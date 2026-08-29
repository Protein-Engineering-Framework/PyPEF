# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut

from typing import Dict, Tuple, Any, Optional
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior

from pypef.utils.helpers import get_device
from pypef.gaussian_process.kermut.gp.kermut_gp import KermutGP
from pypef.gaussian_process.kermut.kernels.structure_kernel import StructureKernel
from pypef.gaussian_process.kermut.kernels.sequence_kernel import SequenceKernel
from pypef.gaussian_process.kermut.tokenizer import Tokenizer


def instantiate_gp(
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    gp_inputs: Dict[str, Any],
    use_structure_kernel: bool = True,
    use_sequence_kernel: bool = True,
    use_zero_shot: bool = True,
    use_prior: bool = True,
    noise_prior_scale: float = 0.1,
    sequence_kernel_type: str = "RBF",
    device: str | None = None,
    sequence_kernel_kwargs: Optional[Dict[str, Any]] = None,
    structure_kernel_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[KermutGP, GaussianLikelihood]:
    """Instantiates a KermutGP model by explicitly constructing sequence and structural kernels.
    Sequence and structure kernels can be used providing two different model inputs (e.g. sequence 
    kernel input information from ESM and structure kernel information from ProSST or PMPNN). 
    """

    if device is None:
        device = get_device()
    
    train_inputs = tuple(
        x.float() if (isinstance(x, torch.Tensor) and torch.is_floating_point(x)) else x 
        for x in train_inputs
    )
    train_targets = train_targets.float()

    seq_kwargs = sequence_kernel_kwargs or {}
    struct_kwargs = structure_kernel_kwargs or {}

    if use_prior:
        noise_prior = HalfCauchyPrior(scale=noise_prior_scale)
    else:
        noise_prior = None

    likelihood = GaussianLikelihood(noise_prior=noise_prior)
    composite = use_structure_kernel and use_sequence_kernel
    
    # Filter using the already downcasted float32 inputs
    filtered_train_inputs = tuple([x for x in train_inputs if x is not None])

    # Construct Sub-Kernels with Explicit Parameter Routing
    seq_kernel = None
    if use_sequence_kernel:
        normalized_type = "RBF" if sequence_kernel_type.upper() == "RBF" else "Matern"
        if normalized_type == "Matern" and "nu" not in seq_kwargs:
            seq_kwargs["nu"] = 2.5
        seq_kernel = SequenceKernel(kernel_type=normalized_type, **seq_kwargs)

    struct_kernel = None
    if use_structure_kernel:
        # Extract and map keys from gp_inputs straight into the structural kernel
        # Handles both 'wt_seq' and 'wt_sequence' naming variations gracefully
        wt_sequence = gp_inputs.get("wt_sequence") or gp_inputs.get("wt_seq")
        
        if not wt_sequence:
            raise ValueError("StructureKernel requires 'wt_sequence' or 'wt_seq' inside gp_inputs.")
        
        if isinstance(wt_sequence, str):
            tokenizer = Tokenizer()
            wt_sequence_tensor = tokenizer(wt_sequence).float()
        else:
            wt_sequence_tensor = wt_sequence.float()
        struct_kwargs.setdefault("wt_sequence", wt_sequence_tensor.to(device))

        # Safe extraction of Conditional Probabilities (prevents multi-element array crashes)
        cond_probs = gp_inputs.get("conditional_probs")
        if cond_probs is None:
            cond_probs = gp_inputs.get("aa_cond_probs")
            
        if cond_probs is None:
            raise ValueError("StructureKernel requires 'conditional_probs' or 'aa_cond_probs' inside gp_inputs.")
        struct_kwargs.setdefault("conditional_probs", cond_probs.to(device))
        
        # Extract coordinates and assign to the exact key 'coords' that the kernel requires
        coords = gp_inputs.get("coords") 
        if coords is None: 
            coords = gp_inputs.get("struct_coords") 
        if coords is None:
            coords = gp_inputs.get("structure_coords")
        if coords is None:
            raise ValueError("StructureKernel requires 'coords', 'struct_coords', or 'structure_coords' inside gp_inputs.")
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords)
        struct_kwargs.setdefault("coords", coords.to(device))

        struct_kernel = StructureKernel(**struct_kwargs)

    # Instantiate Main GP
    gp = KermutGP(
        train_inputs=filtered_train_inputs,
        train_targets=train_targets,
        likelihood=likelihood,
        use_zero_shot_mean=use_zero_shot,
        composite=composite,
        sequence_kernel=seq_kernel,
        structure_kernel=struct_kernel,
        **gp_inputs,
    )
    
    gp = gp.to(device)
    likelihood = likelihood.to(device)

    return gp, likelihood
