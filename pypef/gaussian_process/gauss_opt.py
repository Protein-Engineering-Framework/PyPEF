# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Using adapted Kermut code published under MIT License

"""
Gaussian process optimization similar (but less sophisticated compared) to 
Kermut: Composite kernel regression for protein variant effects
Peter Mørch Groth, Mads Herbert Kerrn, Lars Olsen, Jesper Salomon, Wouter Boomsma
2024, 38th Conference on Neural Information Processing Systems (NeurIPS 2024).
TL;DR: Gaussian process regression model with a novel composite kernel, Kermut, achieves 
state-of-the-art variant effect prediction while providing meaningful uncertainties.
Literature: https://openreview.net/forum?id=jM9atrvUii.
Used under MIT license; code available at https://github.com/petergroth/kermut.
"""


import torch
import gpytorch

from pypef.utils.helpers import get_device, tqdm

import logging
logger = logging.getLogger('pypef.gaussian_process.gauss_opt')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HellingerRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True  # GPyTorch handles log-lengthscale automatically

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Amplitude/variance parameter
        self.register_parameter(
            name="raw_variance",
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )
        self.register_constraint("raw_variance", gpytorch.constraints.Positive())

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self.raw_variance.data = self.raw_variance_constraint.inverse_transform(value)

    def forward(self, x1, x2, diag=False, **params):
            # Normalize inputs to probabilities
            x1 = torch.clamp(x1, min=1e-9)
            x2 = torch.clamp(x2, min=1e-9)
            x1 = x1 / x1.sum(dim=-1, keepdim=True)
            x2 = x2 / x2.sum(dim=-1, keepdim=True)
    
            # Hellinger square root
            x1_sqrt = torch.sqrt(x1)
            x2_sqrt = torch.sqrt(x2)
    
            if diag:
                # Diagonal case is much cheaper: 0.5 * sum((sqrt(p) - sqrt(q))^2)
                h2 = 0.5 * (x1_sqrt - x2_sqrt).pow(2).sum(dim=-1)
            else:
                # 3. Optimized Distance: (a-b)^2 = a^2 + b^2 - 2ab
                # Since sum(x1) = 1, sum(x1_sqrt^2) is always 1.0
                # This eliminates the need for broadcasting subtraction!
                
                # x1_norm and x2_norm are 1.0 because these are probabilities
                # but we calculate them for numerical stability/different input types
                x1_norm = x1_sqrt.pow(2).sum(dim=-1, keepdim=True) # [n1, 1]
                x2_norm = x2_sqrt.pow(2).sum(dim=-1, keepdim=True).transpose(-1, -2) # [1, n2]
                
                # Matrix multiplication: [n1, d] @ [d, n2] -> [n1, n2]
                # This is the "2ab" part
                dot_prod = torch.matmul(x1_sqrt, x2_sqrt.transpose(-1, -2))
                
                # Hellinger Distance Squared (H2)
                # The 0.5 factor comes from the Hellinger definition
                h2 = 0.5 * (x1_norm + x2_norm - 2 * dot_prod)
                h2 = h2.clamp_min(0.0) # Guard against precision-induced negatives
    
            # RBF-like kernel
            return self.variance * torch.exp(-h2 / (2 * self.lengthscale ** 2))


class CombinedKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel_seq, kernel_struct, d_seq=None):
        super().__init__()
        self.kernel_seq = kernel_seq
        self.kernel_struct = kernel_struct
        self.d_seq = d_seq

    def forward(self, X1, X2, **params):
        if self.d_seq is None:
            raise ValueError("d_seq must be specified for CombinedKernel")

        X1_seq, X1_struct = X1[..., :self.d_seq], X1[..., self.d_seq:]
        X2_seq, X2_struct = X2[..., :self.d_seq], X2[..., self.d_seq:]

        return self.kernel_seq(X1_seq, X2_seq) + self.kernel_struct(X1_struct, X2_struct)


class MultiInputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_gp_kernel_model(y_train, x_tokseqs_seq_kernel_train=None, x_tokseqs_struct_kernel_train=None, 
                        device=None, opt_steps: int = 100, train: bool = False):
    # Define kernels and model: x_train is by default seq kernel and 
    # x_train_2 is struct kernel emb for now
    if device is None:
        device = get_device()

    seq_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    struct_kernel = HellingerRBFKernel()
    if x_tokseqs_struct_kernel_train is None:
        logger.info(f"Using only sequence kernel ({x_tokseqs_seq_kernel_train.shape})")
        kernel = seq_kernel
        x_train = x_tokseqs_seq_kernel_train
    elif x_tokseqs_seq_kernel_train is None:
        logger.info(f"Using only structure kernel ({x_tokseqs_struct_kernel_train.shape})")
        kernel = struct_kernel
        x_train = x_tokseqs_struct_kernel_train
    else:
        # [ sequence_features | structure_features ]
        #   <---- d_seq -----> 
        logger.info(
            f"Taking first sequence embeddings for sequence kernel and concatenting "
            f"second sequence embeddings for structure kernel processing ({x_tokseqs_seq_kernel_train.shape}"
            f" + {x_tokseqs_struct_kernel_train.shape} -> "
            f"{torch.cat([x_tokseqs_seq_kernel_train, x_tokseqs_struct_kernel_train], dim=-1).shape}; "
            f"d_seq for split: {x_tokseqs_seq_kernel_train.shape[1]})"
        )
        d_seq = x_tokseqs_seq_kernel_train.shape[1]
        x_train = torch.cat([x_tokseqs_seq_kernel_train, x_tokseqs_struct_kernel_train], dim=-1)
        kernel = CombinedKernel(seq_kernel, struct_kernel, d_seq=d_seq)

    kernel = kernel.to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    x_train = torch.as_tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32).to(device)
    model = MultiInputGP(x_train, y_train, likelihood, kernel).to(device)

    # Train
    # -----------------------------
    if train:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        pbar = tqdm(range(opt_steps), desc='GP training')
        # To save memory but loosing "exactness": Conjugate gradients instead of massive Cholesky decomposition
        # with gpytorch.settings.max_cholesky_size(0), gpytorch.settings.max_preconditioner_size(10):
        for i in pbar:
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"GP training: step {i+1}/{100}, loss: {loss:.4f} "
                                 f"({device.upper()})")
    return model
