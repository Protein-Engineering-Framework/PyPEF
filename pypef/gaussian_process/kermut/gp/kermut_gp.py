# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut

import torch
from typing import Tuple, Optional
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from pypef.gaussian_process.kermut.kernels.composite_kernel import CompositeKernel


class KermutGP(ExactGP):
    """Gaussian Process regression model for supervised variant effects predictions.

    Args:
        train_inputs: Training input data for the GP model. Expects tuple of
            (kermut_tokens, sequence_embeddings, zero_shot_scores).
        train_targets: Target values corresponding to the training inputs.
        likelihood: Gaussian likelihood function for the GP model.
        use_zero_shot_mean: Whether to use a linear mean function for zero-shot predictions.
        composite: Whether to use a composite kernel combining sequence and structure.
        kernel_kwargs: Additional keyword arguments passed directly to the kernel initialization.
    """

    def __init__(
        self,
        train_inputs: Tuple[torch.Tensor, ...],
        train_targets: torch.Tensor,
        likelihood: GaussianLikelihood,
        use_zero_shot_mean: bool = True,
        composite: bool = True,
        **kernel_kwargs,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        
        if composite:
            # Safely pop out kernel configurations from kwargs if nested, otherwise pass along
            seq_k = kernel_kwargs.pop("sequence_kernel", None)
            struct_k = kernel_kwargs.pop("structure_kernel", None)
            self.covar_module = CompositeKernel(
                sequence_kernel=seq_k,
                structure_kernel=struct_k,
                **kernel_kwargs,
            )
        else:
            raise NotImplementedError(
                "Single kernel dynamic instantiation via Hydra is deprecated. Use composite=True."
            )

        self.use_zero_shot_mean = use_zero_shot_mean
        if self.use_zero_shot_mean:
            # Dynamically read feature width from the 3rd element of train_inputs
            zero_shot_dim = 1
            if train_inputs is not None and len(train_inputs) >= 3:
                x_zero_train = train_inputs[2]
                if isinstance(x_zero_train, torch.Tensor):
                    # If it's a 2D matrix [samples, features], grab features - otherwise default to 1.
                    zero_shot_dim = x_zero_train.size(-1) if x_zero_train.dim() > 1 else 1

            self.mean_module = LinearMean(input_size=zero_shot_dim, bias=True)
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_toks: torch.Tensor, x_embed: torch.Tensor, x_zero: Optional[torch.Tensor] = None) -> MultivariateNormal:
        if x_zero is None:
            x_zero = x_toks
        mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module((x_toks, x_embed))
        return MultivariateNormal(mean_x, covar_x)
