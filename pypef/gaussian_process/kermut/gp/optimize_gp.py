# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut


from typing import Optional, Tuple
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from pypef.utils.helpers import tqdm


def _slice_inputs(inputs: Tuple, indices_or_slice):
    """Helper to slice tuple of tensors, lists, or numpy arrays safely."""
    sliced = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            sliced.append(item[indices_or_slice])
        elif isinstance(item, np.ndarray):
            sliced.append(item[indices_or_slice])
        elif isinstance(item, list):
            if isinstance(indices_or_slice, slice):
                sliced.append(item[indices_or_slice])
            else:
                idx_list = (
                    indices_or_slice.tolist() if isinstance(indices_or_slice, torch.Tensor) 
                    else indices_or_slice
                )
                sliced.append([item[i] for i in idx_list])
        else:
            sliced.append(item)
    return tuple(sliced)


def optimize_gp(
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    lr: float = 0.05,
    n_steps: int = 150,
    batch_size: Optional[int] = 256,
    batch_threshold: int = 2000,
    progress_bar: bool = True,
) -> Tuple[ExactGP, GaussianLikelihood]:
    """
    Optimizes a Gaussian Process using marginal likelihood maximization.
    Uses exact full-batch training when dataset size <= batch_threshold (default 2000).
    Switches to mini-batched training when dataset size > batch_threshold to prevent GPU OOM.
    """
    gp.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)
    optimizer = torch.optim.AdamW(gp.parameters(), lr=lr)

    x_train = tuple(
        x.float() if (isinstance(x, torch.Tensor) and torch.is_floating_point(x)) else x 
        for x in train_inputs
    )
    y_train = train_targets.float()
    n_samples = len(y_train)
    device = y_train.device if isinstance(y_train, torch.Tensor) else torch.device("cpu")

    # Only enable mini-batching if dataset exceeds threshold
    use_batching = n_samples > batch_threshold

    if not use_batching:
        # EXACT FULL-BATCH TRAIN (Default for N <= batch_threshold)
        gp.set_train_data(inputs=x_train, targets=y_train, strict=True)

        pbar = tqdm(range(n_steps), desc="GP training", disable=not progress_bar)
        for i in pbar:
            optimizer.zero_grad()
            output = gp(*x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"GP training: step {i+1}/{n_steps}, loss: {loss.item():.4f} "
                f"({str(loss.device).split(':')[0].upper()})"
            )
    else:
        # MINI-BATCHED TRAIN (Fallback for N > batch_threshold)
        # Prevents CUDA OOM on large training sets
        step_size = batch_size if batch_size is not None else 256
        
        pbar = tqdm(range(n_steps), desc="GP training (batched)", disable=not progress_bar)
        for epoch in pbar:
            perm = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, step_size):
                batch_indices = perm[start_idx : start_idx + step_size]
                x_batch = _slice_inputs(x_train, batch_indices)
                y_batch = y_train[batch_indices]

                optimizer.zero_grad()
                gp.set_train_data(inputs=x_batch, targets=y_batch, strict=False)
                output = gp(*x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            pbar.set_description(
                f"GP epoch {epoch+1}/{n_steps} | Avg Loss: {avg_loss:.4f} "
                f"({str(device).split(':')[0].upper()})"
            )

        # Restore full dataset on the GP model for downstream evaluation
        gp.set_train_data(inputs=x_train, targets=y_train, strict=False)

    gp.eval()
    likelihood.eval()

    return gp, likelihood
