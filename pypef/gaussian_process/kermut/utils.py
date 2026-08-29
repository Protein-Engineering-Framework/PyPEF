# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License; 
# available at https://github.com/petergroth/kermut


from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from pypef.utils.helpers import get_device
from pypef.gaussian_process.kermut.tokenizer import Tokenizer


def prepare_kermut_inputs(
    seqs: List[str],
    x_embed: torch.Tensor,
    x_zero_shot: torch.Tensor,
    device: str | None = None
) -> Tuple[torch.Tensor, ...]:
    """Generates the properly routed and structured tensor tuples for training and testing.
    Handles internal structural alignment constraints using Kermut's specific Tokenizer.
    """
    if device is None:
        device == get_device()
    tokenizer = Tokenizer()
    x_kermut_toks = torch.stack([tokenizer(seq) for seq in seqs]).float()
    inputs = (x_kermut_toks.to(device), x_embed.to(device), x_zero_shot.to(device))
    return inputs
