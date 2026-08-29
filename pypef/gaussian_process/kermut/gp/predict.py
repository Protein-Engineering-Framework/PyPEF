# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut

from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from pypef.gaussian_process.kermut.gp.kermut_gp import KermutGP


def _slice_inputs(inputs: Tuple, start_idx: int, end_idx: int) -> Tuple:
    """Slices elements of test_inputs (Tensors, lists, or numpy arrays) for mini-batching."""
    sliced = []
    for item in inputs:
        if isinstance(item, (torch.Tensor, np.ndarray, list)):
            sliced.append(item[start_idx:end_idx])
        else:
            sliced.append(item)
    return tuple(sliced)


def predict(
    gp: KermutGP,
    likelihood: GaussianLikelihood,
    test_inputs: Tuple[torch.Tensor, ...],
    batch_size: int = 1000,
    *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Makes predictions using a trained Gaussian Process in mini-batches to avoid CUDA OOM.
    
    Returns raw prediction means and variances.
    """
    gp.eval()
    likelihood.eval()

    # Cast all floating-point test inputs to float32 to match the model's parameters
    x_test = tuple(
        x.float() if (isinstance(x, torch.Tensor) and torch.is_floating_point(x)) else x 
        for x in test_inputs
    )

    # Determine total number of test samples from the first input item
    n_samples = len(x_test[0]) if len(x_test) > 0 else 0

    # Allow overriding batch_size via kwargs (e.g. eval_batch_size) if passed
    effective_batch_size = kwargs.get("eval_batch_size", batch_size)
    step_size = effective_batch_size if (
        effective_batch_size is not None and effective_batch_size > 0
        ) else n_samples

    all_means = []
    all_variances = []

    # Wrap in standard GPyTorch inference managers
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start_idx in range(0, n_samples, step_size):
            end_idx = min(start_idx + step_size, n_samples)
            x_batch = _slice_inputs(x_test, start_idx, end_idx)

            # Pass the sanitized float32 test elements into the GP
            y_preds_dist = likelihood(gp(*x_batch))
            
            # Extract means and variances
            all_means.append(y_preds_dist.mean.detach())
            all_variances.append(y_preds_dist.variance.detach())

    test_means = torch.cat(all_means, dim=0) if all_means else torch.tensor([])
    test_variances = torch.cat(all_variances, dim=0) if all_variances else torch.tensor([])

    return test_means, test_variances
