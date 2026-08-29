# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Contains Python code used for the approach presented in our 'hybrid modeling' paper
# Preprint available at: https://doi.org/10.1101/2022.06.07.495081
# Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

from __future__ import annotations

import os
import pickle
import re
from os import listdir
from os.path import isfile, join
from typing import Union
import warnings
import gc
from pypef.gaussian_process.kermut.gp.instantiate_gp import instantiate_gp
from pypef.gaussian_process.kermut.gp.optimize_gp import optimize_gp
from pypef.gaussian_process.kermut.gp.predict import predict
from pypef.gaussian_process.kermut.utils import prepare_kermut_inputs
import torch
import numpy as np
import sklearn.base
from scipy.stats import spearmanr
#from sklearnex import patch_sklearn
#patch_sklearn(verbose=False)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.optimize import differential_evolution

from pypef.settings import USE_RAY
from pypef.utils.variant_data import (
    block_random_train_test_split, contiguous_train_test_split, extract_pdb_coords, 
    get_sequences_from_file, modulo_train_test_split, positional_train_test_split, 
    remove_nan_encoded_positions
)
import pypef.dca.plmc_encoding
from pypef.dca.plmc_encoding import PLMC, get_dca_data_parallel, get_encoded_sequence
from pypef.utils.to_file import predictions_out
from pypef.utils.helpers import get_device
from pypef.utils.plot import plot_y_true_vs_y_pred
import pypef.dca.gremlin_inference
from pypef.plm.utils import hybrid_corr_mse_loss
from pypef.dca.gremlin_inference import GREMLIN, get_delta_e_statistical_model
from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.prosst_lora_tune import get_prosst_models
from pypef.plm.inference import (
    esm_setup, prosst_setup, 
    tokenize_sequences, plm_inference, get_plm_embeddings
)

# sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and 
# will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function 
# becomes public and is part of the scikit-learn developer API.
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')

import logging
logger = logging.getLogger('pypef.hybrid.hybrid_model')


# TODO: Add meta-learning model (e.g., learn2learn MAML learning option on PGym dataset)?
class DCALLMHybridModel:
    def __init__(
            self,
            x_train_dca: np.ndarray,
            y_train: np.ndarray,
            llm_model_input: dict | None = None,
            x_dca_wt: np.ndarray | None = None,
            sequences: list[str] | None = None,
            wt_sequence: str | None = None,
            alphas: np.ndarray | None = None,
            parameter_range: list[tuple] | None = None,
            ensemble_func: str = 'torch',
            splitting_scheme: str = 'random',
            shared_fraction: float | None = None,
            lora_train: bool = False,
            gauss_opt: bool = False,
            gauss_comb_plm_emb: bool = False,
            pdb_struct: str | os.PathLike | None = None,
            batch_size: int | None = None,
            n_epochs: int | None = None,
            device: str | None = None,
            seed: int | None = None,
            n_ensemble_splits: int = 1,
            verbose: bool = True,
            progress_cb=None, 
            abort_cb=None
    ):
        if llm_model_input is not None:
            if not isinstance(llm_model_input, dict):
                raise RuntimeError("Model input must be in form of a dictionary.")
            
            # Get the list of provided models (normalized to uppercase)
            self.llm_keys = [k.upper() for k in llm_model_input.keys()]
            supported_models = {'ESM', 'PROSST'}
            
            # Check for unsupported models
            unsupported = set(self.llm_keys) - supported_models
            if unsupported:
                raise RuntimeError(f"PLM input models {unsupported} not supported. "
                                   f"Currently supported models are {supported_models}")

            logger.info(f"Using PLM(s) ({', '.join(self.llm_keys)}) next to DCA for hybrid modeling...")
            
            # Store the entire dictionary (re-keyed to uppercase) so we can loop through it later
            self.llm_data = {k.upper(): v for k, v in llm_model_input.items()}
            if parameter_range is None:
                parameter_range = [(0, 1), (0, 1), (0, 1), (0, 1)] 
        else:
            logger.info("No PLM inputs were defined for hybrid modelling. "
                  "Using only DCA for hybrid modeling...")
            self.llm_keys = None
            self.llm_model_input = None
            self.llm_attention_mask = None
            if parameter_range is None:
                parameter_range = [(0, 1), (0, 1)]
        self.sequences = sequences
        self.wt_sequence = wt_sequence
        if alphas is None:
            alphas = np.logspace(-6, 6, 100)
        self.parameter_range = parameter_range
        self.ensemble_func = ensemble_func
        self.splitting_scheme = splitting_scheme
        if shared_fraction is None:
            shared_fraction = 0.0
        self.shared_fraction = shared_fraction
        self.gauss_opt = gauss_opt
        self.gauss_comb_plm_emb = gauss_comb_plm_emb
        self.pdb_struct = pdb_struct
        self.lora_train = lora_train
        self.alphas = alphas
        self.x_train_dca = x_train_dca
        self.y_train = y_train
        self.x_wild_type = x_dca_wt
        if device is None:
            device = get_device()
        self.device = device
        logger.info(f'Using device {device.upper()} for hybrid modeling...')
        self.seed = seed
        # Seed all RNGs when a seed is given so that the PLM-based paths are
        # reproducible too: LoRA fine-tuning (torch) and Gaussian-process
        # optimization (torch/gpytorch) otherwise draw from the global torch RNG,
        # which the per-call `random_state=self.seed` (split/DE) does not cover.
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
        self.n_ensemble_splits = n_ensemble_splits
        if batch_size is None:
            batch_size = 5
        self.batch_size = batch_size
        if n_epochs is None:
            n_epochs = 50
        self.n_epochs = n_epochs
        self.verbose = verbose
        (
            self.ridge_opt, 
            self.betas,
            self.y_dca_ttest,
            self.y_dca_ridge_ttest,
            self.y_llm_ttest,
            self.y_llm_lora_ttest,
            self.y_llm_ttrain,
            self.y_llm_lora_ttrain,
            self.y_gp_opt_ttest
        ) = None, None, None, None, None, None, None, None, None
        self.progress_cb = progress_cb
        self.abort_cb = abort_cb
        if self.gauss_opt and (
            self.sequences is None or self.wt_sequence is None or 
            self.pdb_struct is None
        ):
            raise RuntimeError(
                "Gaussian optimization requires variant and wt "
                "sequence inputs (`sequences` and `wt_sequence`) "
                "as well as the path to the wild-type protein "
                "structure file in PDB format (`pdb_struct`).")
        self.train_and_optimize()

    @staticmethod
    def spearmanr(
            y1: np.ndarray,
            y2: np.ndarray
    ) -> float:
        """
        Parameters
        ----------
        y1 : np.ndarray
            Array of target fitness values.
        y2 : np.ndarray
            Array of predicted fitness values.

        Returns
        -------
        Spearman's rank correlation coefficient.
        """
        return spearmanr(y1, y2)[0]

    @staticmethod
    def _standardize(
            x: np.ndarray,
            axis=0
    ) -> np.ndarray:
        """
        Standardizes the input array x by subtracting the mean
        and dividing it by the (sample) standard deviation.

        Parameters
        ----------
        x : np.ndarray
            Array to be standardized.
        axis : integer (default=0)
            Axis to exectute operations on.

        Returns
        -------
        Standardized version of 'x'.
        """
        return np.subtract(x, np.mean(x, axis=axis)) / np.std(x, axis=axis, ddof=1)

    def _delta_x(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Substracts for each variant the encoded wild-type sequence
        from its encoded sequence.
        
        Parameters
        ----------
        x : np.ndarray
            Array of encoded variant sequences (matrix X).

        Returns
        -------
        Array of encoded variant sequences with substracted encoded
        wild-type sequence.
        """
        return np.subtract(x, self.x_wild_type)

    def _delta_e(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the difference of the statistical energy 'dE'
        of the variant and wild-type sequence.

        dE = E (variant) - E (wild-type)
        with E = sum_{i} h_i (o_i) + sum_{i<j} J_{ij} (o_i, o_j)

        Parameters
        ----------
        X : np.ndarray
            Array of the encoded variant sequences.

        Returns
        -------
        Difference of the statistical energy between variant 
        and wild-type.
        """
        return np.sum(self._delta_x(x), axis=1)

    def _spearmanr_dca(self) -> float:
        """
        Returns
        -------
        Spearman's rank correlation coefficient of the full
        data and the statistical DCA predictions (difference
        of statistical energies). Used to adjust the sign
        of hybrid predictions, i.e.
            beta_1 * y_dca + beta_2 * y_ridge
        or
            beta_1 * y_dca - beta_2 * y_ridge.
        """
        y_dca = self._delta_e(self.x_train_dca)
        return self.spearmanr(self.y_train, y_dca)

    def ridge_predictor(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
    ) -> object:
        """
        Sets the parameter 'alpha' for ridge regression.

        Parameters
        ----------
        x_train : np.ndarray
            Array of the encoded sequences for training.
        y_train : np.ndarray
            Associated fitness values to the sequences present
            in 'x_train'.

        Returns
        -------
        Ridge object trained on 'x_train' and 'y_train' (cv=5)
        with optimized 'alpha'. 
        """
        grid = GridSearchCV(
            Ridge(), 
            {'alpha': self.alphas, 'random_state': [self.seed]}, 
            cv=5
        )
        grid.fit(x_train, y_train)
        return grid.best_estimator_
    
    def optimize_ensemble_weights_torch(
            self,
            y_true_np: np.ndarray, 
            *y_preds_np: np.ndarray | None, 
            method: str = "spearman-hybrid",
            alpha: float = 0.5
        ) -> np.ndarray:

            valid_indices = []
            valid_preds_list = []

            # Prepare and Flip Predictors
            for i, p in enumerate(y_preds_np):
                if p is not None:
                    # OPTIONAL: If a predictor is naturally anti-correlated, 
                    # one might want to flip it here: p = -p
                    p_std = (p - np.mean(p)) / (np.std(p) + 1e-8)
                    valid_preds_list.append(torch.from_numpy(p_std).float())
                    valid_indices.append(i)

            X = torch.stack(valid_preds_list, dim=-1)
            y_true_std = torch.from_numpy((y_true_np - np.mean(y_true_np)) / (np.std(y_true_np) + 1e-8)).float()

            # Leaf Tensor initialization
            betas = torch.ones(X.shape[1], requires_grad=True)
            with torch.no_grad():
                betas /= X.shape[1]

            optimizer = torch.optim.LBFGS([betas], lr=0.1, max_iter=20)

            def closure():
                optimizer.zero_grad()
                # SOFTPLUS: Forces weights to be positive, but allowed to be > 1
                weights = torch.nn.functional.softplus(betas)
                y_hat = X @ weights
                loss = hybrid_corr_mse_loss(y_true_std, y_hat, method=method, alpha=alpha)
                loss.backward()
                return loss

            for _ in range(25): # L-BFGS is efficient, 25 steps is usually plenty
                optimizer.step(closure)

            # Transform the raw betas into the EFFECTIVE weights
            with torch.no_grad():
                # You must apply softplus here too!
                effective_weights = torch.nn.functional.softplus(betas).numpy()

            # Map back to original predictor slots
            final_betas = np.zeros(len(y_preds_np))
            for idx, w in zip(valid_indices, effective_weights):
                final_betas[idx] = w

            # Internal validation check
            y_ensemble = np.zeros_like(y_true_np)
            for i, p in enumerate(y_preds_np):
                if p is not None:
                    p_scaled = (p - np.mean(p)) / (np.std(p) + 1e-8)
                    y_ensemble += final_betas[i] * p_scaled

            final_corr = self.spearmanr(y_true_np, y_ensemble)
            self.betas = final_betas
            self.betas_str += f" [{', '.join(f'{x:.2e}' for x in self.betas)}]"
            logger.info(
                f"Ensemble Opt. Spearman: {final_corr:.3f} (N_test={len(y_ensemble)}) | "
                f"Ensemble weights ({len(final_betas)}): {self.betas_str}"
            )
            return final_betas
    
    def optimize_ensemble_weights(self, y: np.ndarray, *predictions: np.ndarray | None) -> np.ndarray:
        """
        Find parameters that maximize the absolute Spearman rank
        correlation coefficient using differential evolution 
        (pos-hoc ensemble weighting).

        Parameters
        ----------
        y : np.ndarray
            Array of fitness values.
        *predictions : np.ndarray | None
            A variable number of prediction arrays to balance. 
            None values are safely filtered out.

        Returns
        -------
        np.ndarray
            Array of beta weights corresponding to each valid prediction input, 
            maximizing the absolute Spearman rank correlation coefficient.
        """
        valid_preds = [p for p in predictions if p is not None]
        if not valid_preds:
            raise ValueError("At least one valid prediction array must be provided.")

        num_preds = len(valid_preds)

        # Dynamically identify which predictors contain NaNs
        clean_indices = []
        clean_preds = []

        for i, p in enumerate(valid_preds):
            if np.any(np.isnan(p)):
                logger.warning(f"Predictor at index {i} contains NaNs. Weighting parameter with zero...")
            else:
                clean_indices.append(i)
                clean_preds.append(p)

        # If all predictors were full of NaNs, return an array of zeros
        if not clean_preds:
            return np.zeros(num_preds)

        # Stack clean predictions into a 2D matrix for fast vectorized multiplication
        # Shape: (n_samples, n_clean_predictors)
        X_clean = np.column_stack(clean_preds)

        # 4. Define the objective function using matrix multiplication (@)
        def loss(params):
            # This dynamically replaces params[0]*y0 + params[1]*y1 + ...
            y_pred = X_clean @ params
            return -np.abs(self.spearmanr(y, y_pred))

        # Handle bounds dynamically to match the number of clean parameters
        # If self.parameter_range is a list of bounds, map it to the clean indices.
        # Otherwise, duplicate the base tuple (e.g., (-1, 1)) for each parameter.
        if isinstance(self.parameter_range, list):
            bounds = [self.parameter_range[i] for i in clean_indices]
        else:
            bounds = [self.parameter_range] * len(clean_indices)

        # Run the optimizer
        try:
            minimizer = differential_evolution(
                loss, bounds=bounds, tol=1e-4, rng=self.seed
            )
        except TypeError:  # SciPy v. 1.15.0 change: `seed` -> `rng` keyword
            minimizer = differential_evolution(
                loss, bounds=bounds, tol=1e-4, seed=self.seed
            )

        # Reconstruct the final weights array, leaving NaN predictors as 0.0
        final_betas = np.zeros(num_preds)
        for clean_idx, opt_weight in zip(clean_indices, minimizer.x):
            final_betas[clean_idx] = opt_weight
        
        y_ensemble = np.zeros_like(y, dtype=np.float64)

        # Iterate through predictors and apply weights
        for i, p in enumerate(predictions):
            if p is not None:
                # Standardize the predictor to match the scale used during optimization
                # (Crucial if your loss function used MSE)
                p_mean = np.mean(p)
                p_std = np.std(p) + 1e-8
                p_scaled = (p - p_mean) / p_std

                # Add weighted contribution
                y_ensemble += final_betas[i] * p_scaled

        # Validate the final ensemble performance
        final_corr = self.spearmanr(y, y_ensemble)
        logger.info(f"Ensemble-optimized Spearman: {final_corr:.3f} (N_opt={len(y)}) "
                    f"| Weights: {final_betas} (N_predictors={len(predictions)})")
        self.betas = final_betas
        self.betas_str += f"[{', '.join(f'{x:.2e}' for x in self.betas)}]"
        return final_betas

    def get_subsplits_train(self, train_size_fit: float = 0.66):
        logger.info(
            "Getting subsplits for supervised (re-)training of models "
            "and for adjustment of hybrid component contribution "
            "weights (\"beta's\")..."
        )
        train_size_fit = int(train_size_fit * len(self.y_train))
        train_size_beta_adjustment = len(self.y_train) - train_size_fit
        logger.info(
            f"Splitting training data of size {len(self.y_train)} "
            f"into {train_size_fit} variants for model tuning and "
            f"{train_size_beta_adjustment} variants for hybrid model "
            f"beta adjustment using the {self.splitting_scheme} "
            f"splitting scheme..."
        )
        # Reduce sizes by batch modulo
        n_drop = train_size_fit % self.batch_size
        if n_drop > 0:
            train_size_fit = train_size_fit - n_drop
            train_size_beta_adjustment = len(self.y_train) - train_size_fit
            logger.info(
                  f"Shifting {n_drop} variants from training set to "
                  f"beta adjustment set to match batch requirements "
                  f"of batch size {self.batch_size} for PLM retraining "
                  f"resulting in {train_size_fit} variants for model "
                  f"tuning and {train_size_beta_adjustment} variants "
                  f"for determination of individual hybrid model weights "
                  f"(beta adjustment)..."
            )
        # Base arrays that are guaranteed to exist
        arrays_to_split = [self.x_train_dca, self.y_train]
        
        # Track if we are splitting variants to handle indexing dynamically
        has_sequences = self.sequences is not None
        if has_sequences:
            arrays_to_split.append(self.sequences)

        if self.llm_keys is not None:
            for llm_name in self.llm_keys:
                arrays_to_split.append(self.llm_data[llm_name]['x_llm'])
        if self.splitting_scheme == "random":
            splits = train_test_split(
                *arrays_to_split, 
                train_size=train_size_fit,
                random_state=self.seed
            )
        elif self.splitting_scheme == "positional":
            splits = positional_train_test_split(
                *arrays_to_split,
                wt_sequence=self.wt_sequence,
                variant_sequences=self.sequences,
                train_size=train_size_fit,
                random_state=self.seed,
                shared_fraction=self.shared_fraction,
                verbose=self.verbose
            )
        elif self.splitting_scheme == "modulo":
            splits = modulo_train_test_split(
                *arrays_to_split,
                wt_sequence=self.wt_sequence,
                variant_sequences=self.sequences,
                train_size=train_size_fit,
                random_state=self.seed,
                verbose=self.verbose
            )
        elif self.splitting_scheme == "contiguous":
            splits = contiguous_train_test_split(
                *arrays_to_split,
                wt_sequence=self.wt_sequence,
                variant_sequences=self.sequences,
                train_size=train_size_fit,
                random_state=self.seed,
                verbose=self.verbose
            )
        elif self.splitting_scheme == "block-random":
            splits = block_random_train_test_split(
                *arrays_to_split,
                wt_sequence=self.wt_sequence,
                variant_sequences=self.sequences,
                train_size=train_size_fit,
                random_state=self.seed,
                verbose=self.verbose
            )
        else:
            raise RuntimeError(
                f"Unknown splitting scheme '{self.splitting_scheme}' - "
                f"splitting scheme has to be 'random', 'block-random', "
                f"'positional', 'modulo', or 'contiguous'."
            )
        
        self.x_dca_ttrain = np.asarray(splits[0], dtype=float)
        self.x_dca_ttest = np.asarray(splits[1], dtype=float)
        self.y_ttrain = np.asarray(splits[2], dtype=float)
        self.y_ttest = np.asarray(splits[3], dtype=float)
        
        # Dynamically set index based on whether variants were included
        if has_sequences:
            self.sequences_ttrain = splits[4]
            self.sequences_ttest = splits[5]
            current_idx = 6
        else:
            self.sequences_ttrain = None
            self.sequences_ttest = None
            current_idx = 4
        
        if self.llm_keys is not None:
            for llm_name in self.llm_keys:
                self.llm_data[llm_name]['x_llm_ttrain'] = splits[current_idx]
                self.llm_data[llm_name]['x_llm_ttest'] = splits[current_idx + 1]
                current_idx += 2
        #except ValueError:
        """
        Not enough sequences to construct a sub-training and sub-testing 
        set when splitting the training set.
        Machine learning/adjusting the parameters 'beta_1' and 'beta_2' not 
        possible -> return parameter setting for 'EVmutation/GREMLIN' model.

        The sub-training set 'y_ttrain' is subjected to a five-fold cross 
        validation. This leads to the constraint that at least two sequences
        need to be in the 20 % of that set in order to allow a ranking. 
        If this is not given -> return parameter setting for 'EVmutation/GREMLIN' model.
        """


    def train_llm(self):
        # LoRA training on y_llm_ttrain --> Testing on y_llm_ttest 
        # Here, just getting the unsupervised scores and correlations on ttrain and ttest splits
        self.betas_str = "DCA, Ridge, "
        self.y_llm_preds = {}       # e.g., {'esm': array, 'prosst': array}
        self.y_llm_lora_preds = {} 
        self.embeddings = {}        # e.g., {'esm': emb, 'prosst': emb}
        # Initialize dictionaries for embeddings
        self.embs_ttrain, self.scores_ttrain = {}, {}
        self.embs_ttest, self.scores_ttest = {}, {}
        self.aa_cond_probs_dict = {}
        self.gp_models = {}
        self.gp_likelihoods = {}

        self.fold_predictions = {}

        # Loop through whatever models were passed in __init__
        for llm_name in self.llm_keys:
            logger.info(f"Processing PLM {llm_name}...")
            # Extract this specific model's data
            current_llm = self.llm_data[llm_name]
            base_model = current_llm['llm_base_model']
            lora_model = current_llm['llm_model']
            inference_fn = current_llm['llm_inference_function']
            training_fn = current_llm['llm_train_function']
            loss_fn = current_llm['llm_loss_function']
            optimizer = current_llm['llm_optimizer']
            tokenizer = current_llm['llm_tokenizer']
            x_tok_llm_ttrain = current_llm['x_llm_ttrain']
            x_tok_llm_ttest = current_llm['x_llm_ttest']
            wt_input_ids = current_llm['wt_input_ids']
            attention_mask = current_llm['llm_attention_mask']
            wt_struct_ids = current_llm.get('wt_structure_input_ids')
            
            y_llm_ttest = inference_fn(
                tokenized_sequences=x_tok_llm_ttest,
                model=base_model,  # Before training, LoRA and Base model yield the same results
                wt_input_ids=wt_input_ids,
                attention_mask=attention_mask,
                device=self.device,
                wt_structure_input_ids=wt_struct_ids,
                batch_size=self.batch_size,
                verbose=True
            )
            y_llm_ttrain = inference_fn(
                tokenized_sequences=x_tok_llm_ttrain,
                model=base_model,
                wt_input_ids=wt_input_ids,
                attention_mask=attention_mask,
                device=self.device,
                wt_structure_input_ids=wt_struct_ids,
                batch_size=self.batch_size,
                verbose=True
            )
            logger.info(
                f"{llm_name} unsupervised performance: "
                f"Train set = {spearmanr(self.y_ttrain, y_llm_ttrain.detach().cpu())[0]:.3f}"
                f" (N={len(self.y_ttrain)}), "
                f"Test set = {spearmanr(self.y_ttest, y_llm_ttest.detach().cpu())[0]:.3f}"
                f" (N={len(self.y_ttest)})"
            )
            self.y_llm_ttrain = y_llm_ttrain.detach().cpu().numpy()
            self.y_llm_ttest = y_llm_ttest.detach().cpu().numpy()
            self.fold_predictions[f"{llm_name}_base"] = self.y_llm_ttest
            self.betas_str += f"{llm_name}-ZS, "

            if self.lora_train:
                logger.info('Refining/training the model... gradient calculation adds a computational '
                      'graph that requires quite some memory - if you are facing an (out of memory) '
                      'error, try reducing the batch size or sticking to CPU device...')
                training_fn(
                    x_sequences=x_tok_llm_ttrain,
                    scores=self.y_ttrain,
                    loss_fn=loss_fn,
                    model=lora_model,
                    optimizer=optimizer,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    n_epochs=self.n_epochs,
                    device=self.device,
                    verbose=self.verbose,
                    raise_error_on_train_fail=False,
                    progress_cb=self.progress_cb,
                    abort_cb=self.abort_cb,
                    wt_structure_input_ids=wt_struct_ids,
                    batch_size=self.batch_size,
                    seed=self.seed,
                )
                y_llm_lora_ttrain = inference_fn(
                    tokenized_sequences=x_tok_llm_ttrain,
                    model=lora_model,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    device=self.device,
                    verbose=self.verbose,
                    wt_structure_input_ids=wt_struct_ids,
                    batch_size=self.batch_size,
                )
                y_llm_lora_ttest = inference_fn(
                    tokenized_sequences=x_tok_llm_ttest,
                    model=lora_model,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    device=self.device,
                    verbose=self.verbose,
                    wt_structure_input_ids=wt_struct_ids,
                    batch_size=self.batch_size,
                )
                logger.info(
                    f"{llm_name} supervised tuned performance: "
                    f"Train = {spearmanr(self.y_ttrain, y_llm_lora_ttrain.detach().cpu())[0]:.3f} "
                    f"(N={len(self.y_ttrain)}), "
                    f"Test = {spearmanr(self.y_ttest, y_llm_lora_ttest.detach().cpu())[0]:.3f} "
                    f"(N={len(self.y_ttest)})"
                )
                self.y_llm_lora_ttrain = y_llm_lora_ttrain.detach().cpu().numpy()
                self.y_llm_lora_ttest = y_llm_lora_ttest.detach().cpu().numpy()
                self.fold_predictions[f"{llm_name}_lora"] = self.y_llm_lora_ttest
                self.betas_str += f"{llm_name}-LoRA, "
        
            if self.gauss_opt:
                embs_ttrain = get_plm_embeddings(
                    x_tok_llm_ttrain, base_model, wt_input_ids, 
                    attention_mask, mode="mean", wt_structure_input_ids=wt_struct_ids,
                    device=self.device
                )

                embs_ttest = get_plm_embeddings(
                    x_tok_llm_ttest, base_model, wt_input_ids, attention_mask, 
                    mode="mean", wt_structure_input_ids=wt_struct_ids,
                    device=self.device
                )

                aa_cond_probs = plm_inference(
                    attention_mask=attention_mask,
                    wt_input_ids=wt_input_ids,
                    model=base_model,
                    tokenized_sequences=None,
                    extract_probs=True, 
                    wt_structure_input_ids=wt_struct_ids,
                    extract_conditional_aa_prob=True,
                    tokenizer=tokenizer,
                    device=self.device
                )

                self.embs_ttrain[llm_name] = embs_ttrain
                self.embs_ttest[llm_name] = embs_ttest
                self.scores_ttrain[llm_name] = y_llm_ttrain
                self.scores_ttest[llm_name] = y_llm_ttest
                self.aa_cond_probs_dict[llm_name] = aa_cond_probs

                train_inputs = prepare_kermut_inputs(
                    seqs=self.sequences_ttrain,
                    x_embed=embs_ttrain,
                    x_zero_shot=y_llm_ttrain,
                    device=self.device
                )

                struct_coords = extract_pdb_coords(
                    self.pdb_struct,
                    target_len=len(self.wt_sequence)
                )

                gp, likelihood = instantiate_gp(
                    train_inputs=train_inputs,
                    train_targets=torch.tensor(self.y_ttrain).to(self.device),
                    gp_inputs={
                        "aa_cond_probs": aa_cond_probs, 
                        "struct_coords": torch.tensor(struct_coords).to(self.device), 
                        "wt_seq": self.wt_sequence
                    },
                    use_structure_kernel=True,
                    use_sequence_kernel=True,
                    sequence_kernel_type="RBF",
                    use_zero_shot=True,
                    device=self.device
                )

                # Train GP
                gp, likelihood = optimize_gp(
                    gp, likelihood, train_inputs, torch.tensor(self.y_ttrain).to(self.device), 
                    lr=0.05, n_steps=150
                )
                self.gp_models[llm_name] = gp
                self.gp_likelihoods[llm_name] = likelihood

                test_inputs = prepare_kermut_inputs(
                    seqs=self.sequences_ttest,
                    x_embed=embs_ttest,
                    x_zero_shot=y_llm_ttest,
                    device=self.device
                )

                y_gp_opt_ttest, _test_variances = predict(gp, likelihood, test_inputs)
                self.y_gp_opt_ttest = y_gp_opt_ttest.detach().cpu().numpy()
                self.fold_predictions[f"{llm_name}_gp"] = self.y_gp_opt_ttest
                self.betas_str += f"{llm_name}-GP, "
        
        # Combined PLM Embedding GP
        if self.gauss_opt and self.gauss_comb_plm_emb:
            if len(self.llm_keys) >= 2:
                # Concatenate embeddings along feature dimension
                x_combined_embeddings_train = torch.cat(
                    list(self.embs_ttrain.values()), 
                    dim=-1
                )
                
                x_combined_embeddings_test = torch.cat(
                    list(self.embs_ttest.values()), 
                    dim=-1
                )

                # Concatenate zero-shot scores
                zs_train_tensors = [
                    v.unsqueeze(-1) if v.dim() == 1 else v 
                    for v in self.scores_ttrain.values()
                ]
                x_combined_zs_train = torch.cat(zs_train_tensors, dim=-1)

                zs_test_tensors = [
                    v.unsqueeze(-1) if v.dim() == 1 else v 
                    for v in self.scores_ttest.values()
                ]
                x_combined_zs_test = torch.cat(zs_test_tensors, dim=-1)
                
                train_inputs = prepare_kermut_inputs(
                    seqs=self.sequences_ttrain,
                    x_embed=x_combined_embeddings_train,
                    x_zero_shot=x_combined_zs_train,
                    device=self.device
                )

                struct_coords = extract_pdb_coords(
                    self.pdb_struct,
                    target_len=len(self.wt_sequence)
                )

                # Primary model conditional probabilities (or first loaded model)
                primary_llm = self.llm_keys[0]
                primary_aa_cond_probs = self.aa_cond_probs_dict[primary_llm]

                gp, likelihood = instantiate_gp(
                    train_inputs=train_inputs,
                    train_targets=torch.tensor(self.y_ttrain).to(self.device),
                    gp_inputs={
                        "aa_cond_probs": primary_aa_cond_probs.to(self.device), 
                        "struct_coords": torch.tensor(struct_coords).to(self.device), 
                        "wt_seq": self.wt_sequence
                    },
                    use_structure_kernel=True,
                    use_sequence_kernel=True,
                    sequence_kernel_type="RBF",
                    use_zero_shot=True,
                    device=self.device
                )

                # Train
                gp, likelihood = optimize_gp(
                    gp, likelihood, train_inputs, 
                    torch.tensor(self.y_ttrain).to(self.device), 
                    lr=0.05, n_steps=150
                )

                self.gp_models["combined"] = gp
                self.gp_likelihoods["combined"] = likelihood

                test_inputs = prepare_kermut_inputs(
                    seqs=self.sequences_ttest,
                    x_embed=x_combined_embeddings_test,
                    x_zero_shot=x_combined_zs_test,
                    device=self.device
                )

                combined_pred_ttest, _test_variances = predict(gp, likelihood, test_inputs)
                combined_pred_ttest = combined_pred_ttest.detach().cpu().numpy()
                
                # FIXED: Store under 'combined_gp' key and label
                self.fold_predictions["combined_gp"] = combined_pred_ttest
                self.betas_str += "Combined-GP, "

                logger.info(
                    f"Combined ({self.llm_keys}) supervised Gaussian process optimized performance: "
                    f"Test = {spearmanr(self.y_ttest, combined_pred_ttest)[0]:.3f} "
                    f"(N={len(self.y_ttest)})"
                )

    def _predict_on_current_split(self) -> tuple[np.ndarray, list]:
        """
        Generate predictions for all sub-components on the current cross-validation sub-split.
        Crucial for calculating multi-split ensemble weights (betas) without alignment drift.
        """
        # Base Unsupervised and Supervised Predictors
        y_dca_ttest = self._delta_e(self.x_dca_ttest)
        
        # Also fixed the triple 't' typo here: self.x_dca_tttest -> self.x_dca_ttest
        y_dca_ridge_ttest = self.ridge_opt.predict(self.x_dca_ttest)

        split_predictors = [y_dca_ttest, y_dca_ridge_ttest]

        embs_pred = {}
        zs_scores_pred = {}

        if self.llm_keys is not None:
            # Process all Base and LoRA PLM variants first
            for llm_name in self.llm_keys:
                current_llm = self.llm_data[llm_name]
                inference_fn = current_llm['llm_inference_function']
                
                # Read directly from the nested model data dict ---
                x_input = current_llm['x_llm_ttest']
                
                if isinstance(x_input, dict):
                    x_input_ids = x_input.get('input_ids', x_input)
                    x_attention_mask = x_input.get('attention_mask', current_llm['llm_attention_mask'])
                else:
                    x_input_ids = x_input
                    x_attention_mask = current_llm['llm_attention_mask']

                common_args = {
                    'wt_input_ids': current_llm['wt_input_ids'],
                    'attention_mask': x_attention_mask,
                    'device': self.device,
                    'verbose': False,
                    'wt_structure_input_ids': current_llm.get('wt_structure_input_ids')
                }

                y_base_ttest = inference_fn(
                    model=current_llm['llm_base_model'], 
                    tokenized_sequences=x_input_ids, 
                    **common_args
                )
                split_predictors.append(y_base_ttest.detach().cpu().numpy())
                
                if self.lora_train:
                    y_lora_ttest = inference_fn(
                        model=current_llm['llm_model'], 
                        tokenized_sequences=x_input_ids, 
                        **common_args
                    )
                    split_predictors.append(y_lora_ttest.detach().cpu().numpy())

                # Cache embedding dimensions if Gaussian Process optimization is enabled
                if self.gauss_opt:
                    llm_embs_pred = get_plm_embeddings(
                        x_input_ids, 
                        current_llm['llm_base_model'], 
                        current_llm['wt_input_ids'], 
                        x_attention_mask, 
                        mode="mean", 
                        wt_structure_input_ids=current_llm.get('wt_structure_input_ids'),
                        device=self.device
                    )
                    embs_pred[llm_name] = llm_embs_pred
                    zs_scores_pred[llm_name] = y_base_ttest

            # Process localized Gaussian Processes sequentially
            if self.gauss_opt:
                # Dynamic fallback tracking for active sub-split string sequences
                sequences_subsplit = getattr(self, 'sequences_ttest', getattr(self, 'seqs_ttest', None))
                
                eval_batch_size = getattr(self, 'eval_batch_size', 1000)  # Safe batch size for prediction
                
                for llm_name in self.llm_keys:
                    pred_inputs = prepare_kermut_inputs(
                        seqs=sequences_subsplit,
                        x_embed=embs_pred[llm_name],
                        x_zero_shot=zs_scores_pred[llm_name],
                        device=self.device
                    )
                    y_gp_opt_pred, _ = predict(
                        self.gp_models[llm_name], 
                        self.gp_likelihoods[llm_name], 
                        pred_inputs,
                        batch_size=eval_batch_size
                    )
                    split_predictors.append(y_gp_opt_pred.detach().cpu().numpy())
                
                # Process the Combined Multi-PLM Gaussian Process
                if self.gauss_comb_plm_emb and len(self.llm_keys) >= 2:
                    x_combined_embeddings_pred = torch.cat(
                        list(embs_pred.values()), 
                        dim=-1
                    )
                    zs_pred_tensors = [
                        v.unsqueeze(-1) if v.dim() == 1 else v 
                        for v in zs_scores_pred.values()
                    ]
                    x_combined_zs_pred = torch.cat(zs_pred_tensors, dim=-1)

                    pred_inputs = prepare_kermut_inputs(
                        seqs=sequences_subsplit,
                        x_embed=x_combined_embeddings_pred,
                        x_zero_shot=x_combined_zs_pred,
                        device=self.device
                    )
                    combined_gp_pred_mean, _ = predict(
                        self.gp_models["combined"], self.gp_likelihoods["combined"], pred_inputs
                    )
                    split_predictors.append(combined_gp_pred_mean.detach().cpu().numpy())

        return self.y_ttest, split_predictors

    def train_and_optimize(self) -> tuple:
        # Base Setup: Compute DCA and DCA-Ridge predictions
        self.get_subsplits_train()
        self.y_dca_ttrain = self._delta_e(self.x_dca_ttrain)
        self.y_dca_ttest = self._delta_e(self.x_dca_ttest)
        self.ridge_opt = self.ridge_predictor(self.x_dca_ttrain, self.y_ttrain)
        self.y_dca_ridge_ttrain = self.ridge_opt.predict(self.x_dca_ttrain)
        self.y_dca_ridge_ttest = self.ridge_opt.predict(self.x_dca_ttest)

        # 1. Start predictors AND feature_names with baseline DCA models
        predictors = [self.y_dca_ttest, self.y_dca_ridge_ttest]
        feature_names = ["y_dca", "y_ridge"]  # Matches self.hybrid_preds keys
        self.betas_str = "DCA, DCA-Ridge, "

        # Print baseline Spearman correlations
        logger.info(
            f"DCA unsupervised ensemble test set performance: "
            f"{spearmanr(self.y_ttest, self.y_dca_ttest)[0]:.3f} (N={len(self.y_ttest)})"
        )
        logger.info(
            f"DCA-Ridge supervised ensemble test set performance: "
            f"{spearmanr(self.y_ttest, self.y_dca_ridge_ttest)[0]:.3f} (N={len(self.y_ttest)})"
        )

        # Train and extract PLM predictions if applicable
        if len(self.parameter_range) >= 4:
            self.train_llm()
            
            if self.llm_keys:
                for llm_name in self.llm_keys:
                    # Unsupervised Base / Zero-Shot PLM
                    if f"{llm_name}_base" in self.fold_predictions:
                        base_preds = self.fold_predictions[f"{llm_name}_base"]
                        predictors.append(base_preds)
                        feature_names.append(f"{llm_name}_base")
                        logger.info(
                            f"{llm_name} zero-shot ensemble test set performance: "
                            f"Test set = {spearmanr(self.y_ttest, base_preds)[0]:.3f} (N={len(self.y_ttest)})"
                        )
                    
                    # Supervised LoRA Fine-tuned PLM
                    if self.lora_train and f"{llm_name}_lora" in self.fold_predictions:
                        lora_preds = self.fold_predictions[f"{llm_name}_lora"]
                        predictors.append(lora_preds)
                        feature_names.append(f"{llm_name}_lora")
                        logger.info(
                            f"{llm_name} LoRA tuned ensemble test set performance: "
                            f"{spearmanr(self.y_ttest, lora_preds)[0]:.3f} (N={len(self.y_ttest)})"
                        )
                    
                    # Supervised Gaussian Process Optimized PLM
                    if self.gauss_opt and f"{llm_name}_gp" in self.fold_predictions:
                        gp_preds = self.fold_predictions[f"{llm_name}_gp"]
                        predictors.append(gp_preds)
                        feature_names.append(f"{llm_name}_gp")
                        logger.info(
                            f"{llm_name} Gaussian process ensemble test set performance: "
                            f"{spearmanr(self.y_ttest, gp_preds)[0]:.3f} (N={len(self.y_ttest)})"
                        )
                
                # Combined Multi-PLM Gaussian Process
                if self.gauss_opt and self.gauss_comb_plm_emb and len(self.llm_keys) >= 2:
                    if "combined_gp" in self.fold_predictions:
                        comb_preds = self.fold_predictions["combined_gp"]
                        predictors.append(comb_preds)
                        feature_names.append("combined_gp")
                        logger.info(
                            f"Combined ({self.llm_keys}) Gaussian process ensemble test set performance: "
                            f"{spearmanr(self.y_ttest, comb_preds)[0]:.3f} (N={len(self.y_ttest)})"
                        )

        # Attach feature_names to self so it gets saved with the pickled model
        self.feature_names = feature_names

        # Clean up the trailing comma left behind by train_llm() on self.betas_str
        if hasattr(self, 'betas_str'):
            self.betas_str = self.betas_str.rstrip(", ") + ":"

        # Optimize ensemble weights on the initial split
        if self.ensemble_func == 'torch':
            first_betas = self.optimize_ensemble_weights_torch(self.y_ttest, *predictors)
        else:
            first_betas = self.optimize_ensemble_weights(self.y_ttest, *predictors)

        # Handle Cross-Validation Multi-Split Routine
        if self.n_ensemble_splits <= 1:
            self.all_betas = first_betas
        else:
            all_split_betas = [first_betas]
            original_seed = self.seed
            
            for split_i in range(1, self.n_ensemble_splits):
                self.seed = (original_seed + split_i) if original_seed is not None else split_i
                logger.info(
                    f"Multi-split ensemble: computing weights on split "
                    f"{split_i + 1}/{self.n_ensemble_splits} and splitting "
                    f"scheme {self.splitting_scheme} with new seed {self.seed}..."
                )
                self.get_subsplits_train()
                
                y_ttest_split, split_predictors = self._predict_on_current_split()
                
                if self.ensemble_func == 'torch':
                    split_betas = self.optimize_ensemble_weights_torch(y_ttest_split, *split_predictors)
                else:
                    split_betas = self.optimize_ensemble_weights(y_ttest_split, *split_predictors)
                all_split_betas.append(split_betas)

            self.seed = original_seed
            self.all_betas = np.mean(all_split_betas, axis=0)
            logger.info(
                f"Averaged ensemble weights across {self.n_ensemble_splits} splits: "
                f" [{', '.join(f'{x:.2e}' for x in self.all_betas)}]"
            )

        logger.info(f"Hybrid optimization done...")
        return (*self.all_betas, self.ridge_opt)

    def hybrid_prediction(
        self,
        x_dca: np.ndarray,
        x_llm_dict: dict | None = None,
        sequences: list[str] | None = None,
        verbose: bool = True
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        
        # Normalized lookup map for case-insensitive key matching
        x_llm_dict_norm = {}
        if x_llm_dict is not None:
            x_llm_dict_norm = {k.lower(): v for k, v in x_llm_dict.items()}

        betas_used = f" [{', '.join(f'{x:.2e}' for x in self.all_betas)}]"
        logger.info(
            f"Hybrid prediction with N_individual={len(self.all_betas)} model "
            f"weights ('betas') = {self.betas_str} -> used: {betas_used}..."
        )
        
        y_dca = self._delta_e(x_dca)
        y_ridge = self.ridge_opt.predict(x_dca) if self.ridge_opt is not None else np.zeros(len(y_dca))

        self.embs_pred = {}
        self.zs_scores_pred = {}

        # Intermediate storage maps
        base_ridge_preds = {"y_dca": y_dca, "y_ridge": y_ridge}
        llm_preds = {}
        gp_preds = {}

        if self.llm_keys is not None:
            # Complete all Base and LoRA PLM variants first
            for llm_name in self.llm_keys:
                current_llm = self.llm_data[llm_name]
                x_input = x_llm_dict_norm.get(llm_name.lower()) if x_llm_dict else None
                
                common_args = {
                    'wt_input_ids': current_llm['wt_input_ids'],
                    'attention_mask': current_llm['llm_attention_mask'],
                    'device': self.device,
                    'verbose': verbose,
                    'wt_structure_input_ids': current_llm.get('wt_structure_input_ids')
                }

                y_base = current_llm['llm_inference_function'](
                    model=current_llm['llm_base_model'], tokenized_sequences=x_input, **common_args
                )
                llm_preds[f"{llm_name}_base"] = y_base.detach().cpu().numpy()
                
                if self.lora_train:
                    y_lora = current_llm['llm_inference_function'](
                        model=current_llm['llm_model'], tokenized_sequences=x_input, **common_args
                    )
                    llm_preds[f"{llm_name}_lora"] = y_lora.detach().cpu().numpy()

                # Extract and store embedding dimensions if Gaussian optimization is active
                if self.gauss_opt:
                    llm_embs_pred = get_plm_embeddings(
                        x_input, 
                        current_llm['llm_base_model'], 
                        current_llm['wt_input_ids'], 
                        current_llm['llm_attention_mask'], 
                        mode="mean", 
                        wt_structure_input_ids=current_llm.get('wt_structure_input_ids'),
                        device=self.device
                    )
                    self.embs_pred[llm_name] = llm_embs_pred
                    self.zs_scores_pred[llm_name] = y_base

            # Complete all localized Gaussian Processes
            if self.gauss_opt:
                for llm_name in self.llm_keys:
                    pred_inputs = prepare_kermut_inputs(
                        seqs=sequences,
                        x_embed=self.embs_pred[llm_name],
                        x_zero_shot=self.zs_scores_pred[llm_name],
                        device=self.device
                    )

                    y_gp_opt_pred, _test_variances = predict(
                        self.gp_models[llm_name], self.gp_likelihoods[llm_name], pred_inputs
                    )
                    gp_preds[f"{llm_name}_gp"] = y_gp_opt_pred.detach().cpu().numpy()
                
                # Complete the Multi-PLM combined Gaussian Process
                if self.gauss_comb_plm_emb and len(self.llm_keys) >= 2:
                    # Concatenate embeddings in exact order of self.llm_keys
                    x_combined_embeddings_pred = torch.cat(
                        [self.embs_pred[k] for k in self.llm_keys], 
                        dim=-1
                    )

                    zs_pred_tensors = [
                        self.zs_scores_pred[k].unsqueeze(-1) 
                        if self.zs_scores_pred[k].dim() == 1 
                        else self.zs_scores_pred[k]
                        for k in self.llm_keys
                    ]
                    x_combined_zs_pred = torch.cat(zs_pred_tensors, dim=-1)

                    pred_inputs = prepare_kermut_inputs(
                        seqs=sequences,
                        x_embed=x_combined_embeddings_pred,
                        x_zero_shot=x_combined_zs_pred,
                        device=self.device
                    )
                    combined_gp_pred_mean, _test_variances = predict(
                        self.gp_models["combined"], self.gp_likelihoods["combined"], pred_inputs
                    )
                    gp_preds["combined_gp"] = combined_gp_pred_mean.detach().cpu().numpy()

        self.hybrid_preds = {}
        
        # Add baselines
        self.hybrid_preds["y_dca"] = base_ridge_preds["y_dca"]
        self.hybrid_preds["y_ridge"] = base_ridge_preds["y_ridge"]
        
        # Interleave variants per PLM key sequentially
        if self.llm_keys is not None:
            for llm_name in self.llm_keys:
                # Base
                if f"{llm_name}_base" in llm_preds:
                    self.hybrid_preds[f"{llm_name}_base"] = llm_preds[f"{llm_name}_base"]
                
                # LoRA
                if self.lora_train and f"{llm_name}_lora" in llm_preds:
                    self.hybrid_preds[f"{llm_name}_lora"] = llm_preds[f"{llm_name}_lora"]
                
                # GP
                if self.gauss_opt and f"{llm_name}_gp" in gp_preds:
                    self.hybrid_preds[f"{llm_name}_gp"] = gp_preds[f"{llm_name}_gp"]
            
            # Add Combined GP at the very end of the sequence
            if self.gauss_opt and self.gauss_comb_plm_emb and len(self.llm_keys) >= 2:
                if "combined_gp" in gp_preds:
                    self.hybrid_preds["combined_gp"] = gp_preds["combined_gp"]

        # Check feature key/count consistency against saved weights ---
        current_keys = list(self.hybrid_preds.keys())
        expected_count = len(self.all_betas)
        current_count = len(current_keys)

        if hasattr(self, "feature_names") and self.feature_names is not None:
            if set(self.feature_names) != set(current_keys):
                raise ValueError(
                    f"\n[PyPEF Mismatch Error] Loaded Hybrid Model feature names mismatch!\n"
                    f"  - Model trained with ({len(self.feature_names)} features): {list(self.feature_names)}\n"
                    f"  - Inference requested ({len(current_keys)} features): {current_keys}\n"
                    f"Fix: Match CLI flags (--gp, --params, --plm) to how the pickled model was trained."
                )

        if expected_count != current_count:
            raise ValueError(
                f"\n[PyPEF Mismatch Error] Number of model weights ('betas') does not match sub-models!\n"
                f"  - Trained weights vector length: {expected_count} ({self.all_betas})\n"
                f"  - Active sub-model predictions count: {current_count} {current_keys}\n"
                f"Fix: Re-fit the hybrid model or adjust CLI flags so weight dimensions match active sub-models."
            )

        predictions = np.array(list(self.hybrid_preds.values()), dtype=float)
        logger.info(f"Running hybrid prediction with: {current_keys}")
        
        # --- PREDICTION AGGREGATION ---
        self.y_hybrid = np.zeros_like(y_dca)
        
        feature_means = getattr(self, "feature_means", None)
        feature_stds = getattr(self, "feature_stds", None)

        for idx, (beta, p) in enumerate(zip(self.all_betas, predictions, strict=True)):
            # Scenario A: Use pre-calculated scaler parameters from model fitting
            if feature_means is not None and feature_stds is not None and idx < len(feature_means):
                p_mean = feature_means[idx]
                p_std_dev = feature_stds[idx]
                p_std = (p - p_mean) / (p_std_dev + 1e-8)
            # Scenario B: Single sequence evaluation (len(p) == 1, e.g. Directed Evolution step)
            elif len(p) == 1:
                p_std = p  # Avoid p - np.mean(p) which zero-out single sequence predictions
            # Scenario C: Batch evaluation fallback (len(p) > 1)
            else:
                std_val = np.std(p)
                p_std = (p - np.mean(p)) / (std_val + 1e-8) if std_val > 1e-8 else (p - np.mean(p))
                
            self.y_hybrid += beta * p_std

        return np.asarray(self.y_hybrid, dtype=float), self.hybrid_preds



""" 
###########################################################################################
# Below: Some helper functions that call or are dependent on the DCALLMHybridModel class. #
###########################################################################################
""" 


def check_model_type(model: dict | DCALLMHybridModel | PLMC | GREMLIN):
    """
    Checks type/instance of model.
    """
    if type(model) == dict:
        try:
            model = model['model']
        except KeyError:
            raise RuntimeError("Unknown model dictionary taken from Pickle file.")
    if type(model) == pypef.dca.plmc_encoding.PLMC:
        return 'PLMC'
    elif type(model) == pypef.hybrid.hybrid_model.DCALLMHybridModel:
        return 'Hybrid'
    elif type(model) == pypef.dca.gremlin_inference.GREMLIN:
        return 'GREMLIN'
    elif isinstance(model, sklearn.base.BaseEstimator):
        raise RuntimeError("Loaded an sklearn ML model. For pure ML-based modeling the "
                          "\'ml\' flag has to be used instead of the \'hybrid\' flag.")
    else:
        raise RuntimeError('Unknown model/unknown Pickle file.')


def get_model_path(model: str):
    """
    Checks if model Pickle files exits in CWD 
    and then in ./Pickles directory.
    """
    # Not capitalizing model names here as PLMC params file names are not capitalized
    # model = os.path.splitext(model)[0].upper() + os.path.splitext(model)[1]
    try:
        if isfile(model):
            model_path = model
        elif isfile(f'Pickles/{model}'):
            model_path = f'Pickles/{model}'
        else:
            raise RuntimeError(
                f"Did not find specified model file ({model}) in current "
                "working directory or /Pickles subdirectory. Make sure "
                "to train/save a model first (e.g., for saving a GREMLIN "
                "model, type \"pypef param_inference --msa TARGET_MSA.a2m\" "
                "or, for saving a plmc model, type \"pypef param_inference "
                "--params TARGET_PLMC.params\")."
            )
        return model_path
    except TypeError:
        raise RuntimeError(
            "No provided model. Specify a " \
            "model for DCA-based encoding."
        )


def get_model_and_type(
        params_file: str, 
        substitution_sep: str = '/'
):
    """
    Tries to load/unpickle model to identify the model type 
    and to load the model from the identified plmc pickle file 
    or from the loaded pickle dictionary.
    """
    file_path = get_model_path(params_file)
    logger.info(f"Unpickling file {os.path.abspath(file_path)}...")
    if type(params_file) == pypef.dca.gremlin_inference.GREMLIN:
        logger.info("Found GREMLIN model...")
        return params_file, 'GREMLIN'
    if type(params_file) == pypef.dca.plmc_encoding.PLMC:
        logger.info("Found PLMC model...")
        return params_file, 'PLMC'
    try:
        with open(file_path, 'rb') as read_pkl_file:
            model = pickle.load(read_pkl_file)
            model_type = check_model_type(model)
    except pickle.UnpicklingError:
        logger.info("Unpickling error... assuming PLMC parameters found...")
        model_type = 'PLMC_Params'

    if model_type == 'PLMC_Params':
        model = PLMC(
            params_file=params_file,
            separator=substitution_sep,
            verbose=False
        )
        model_type = 'PLMC'

    else:  # --> elif model_type in ['PLMC', 'GREMLIN', 'Hybrid']:
        model = model['model']
    if model_type == 'Hybrid':
        if not model.llm_keys:
            logger.info("Found hybrid model without PLM model...")
            return model, model_type
        # Reconstruct each stored PLM (base and LoRA) from its state dictionary
        for llm_name in model.llm_keys:
            current_llm = model.llm_data[llm_name]
            if llm_name == 'ESM':
                logger.info("Found hybrid model with ESM PLM model...")
                base_model, lora_model, tokenizer, _optimizer = get_esm_models()
                model_type += '_ESM'
            elif llm_name == 'PROSST':
                logger.info("Found hybrid model with ProSST PLM model...")
                base_model, lora_model, tokenizer, _optimizer = get_prosst_models()
                model_type += '_ProSST'
            else:
                logger.info(f"Found hybrid model with unknown PLM model {llm_name}...")
                continue
            base_model.load_state_dict(current_llm['llm_base_model'])
            lora_model.load_state_dict(current_llm['llm_model'])
            base_model.eval()
            lora_model.eval()
            current_llm['llm_base_model'] = base_model
            current_llm['llm_model'] = lora_model
            current_llm['llm_tokenizer'] = tokenizer

    return model, model_type


def save_model_to_dict_pickle(
        model: DCALLMHybridModel | PLMC | GREMLIN,
        model_type: str | None = None
):
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass

    if model_type is None:
        model_type = 'MODEL'
    
    # For hybrid PLM models save each PLM (base and LoRA) as state dictionaries
    if model_type.lower().startswith('hybrid'):
        if model.llm_keys is not None:
            for llm_name in model.llm_keys:
                logger.info(f"Storing PLM model {llm_name} "
                      f"of hybrid model as state dictionaries...")
                current_llm = model.llm_data[llm_name]
                current_llm['llm_model'] = current_llm['llm_model'].to('cpu').state_dict()
                current_llm['llm_base_model'] = current_llm['llm_base_model'].to('cpu').state_dict()
            model.progress_cb = None
            model.abort_cb = None
            model_type += ''.join(model.llm_keys)
    pkl_path = os.path.abspath(f'Pickles/{model_type.upper()}')
    pickle.dump(
        {
            'model': model,
            'model_type': model_type
        },
        open(pkl_path, 'wb')
    )
    logger.info(f'Saved model as Pickle file ({pkl_path})...')
    # Free up memory (needed when running from Qt GUI threads?) 
    del model
    torch.cuda.empty_cache()
    gc.collect()


global_model = None
global_model_type = None


def plmc_or_gremlin_encoding(
        variants,
        sequences,
        ys_true,
        params_file,
        substitution_sep='/',
        threads=1,
        verbose=True,
        use_global_model=False
):
    """
    Decides based on the params file input type which DCA encoding 
    to be performed, i.e., GREMLIN or PLMC.
    If use_global_model==True, to avoid each time pickle model 
    file getting loaded, which is quite inefficient when performing 
    directed evolution, i.e., encoding of single sequences, a 
    global model is stored at the first evolution step and used 
    in the subsequent steps.
    """
    global global_model, global_model_type
    if ys_true is None:
        ys_true = np.zeros(np.shape(sequences), dtype=np.float64)
    if use_global_model:
        if global_model is None:
            global_model, global_model_type = get_model_and_type(
                params_file, substitution_sep)
            model, model_type = global_model, global_model_type
        else:
            model, model_type = global_model, global_model_type
    else:
        model, model_type = get_model_and_type(
            params_file, substitution_sep)
    if model_type == 'PLMC':
        xs, x_wt, variants, sequences, ys_true = plmc_encoding(
            model, variants, sequences, ys_true, threads, verbose
        )
    elif model_type == 'GREMLIN':
        if verbose:
            logger.info(
                f"Following positions are frequent gap positions "
                f"in the MSA and cannot be considered for effective "
                f"modeling, i.e., substitutions at these positions "
                f"are removed or are predicted with wild-type fitness:"
                f"\n{[int(gap) + 1 for gap in model.gaps]}.\n"
                f"Effective positions (N={len(model.v_idx)}) are:\n"
                f"{[int(v_pos) + 1 for v_pos in model.v_idx]}"
            )
        xs, x_wt, variants, sequences, ys_true = gremlin_encoding(
            model, variants, sequences, ys_true,
            shift_pos=1, substitution_sep=substitution_sep
        )
    else:
        raise RuntimeError(
            f"Found a {model_type.lower()} model as input. Please "
            f"train a new hybrid model on the provided LS/TS datasets."
        )
    assert len(xs) == len(variants) == len(sequences) == len(ys_true)
    ys_true = np.asarray(ys_true, dtype=np.float64)
    return xs, variants, sequences, ys_true, x_wt, model, model_type


def gremlin_encoding(gremlin: GREMLIN, variants, sequences, ys_true, 
                     shift_pos=1, substitution_sep='/'):
    """
    Gets X and x_wt for DCA prediction: delta_Hamiltonian respectively
    delta_E = np.subtract(X, x_wt), with X = encoded sequences of variants.
    Also removes variants, sequences, and y_trues at MSA gap positions.
    """
    variants, sequences, ys_true = (
        np.atleast_1d(variants), 
        np.atleast_1d(sequences), 
        np.atleast_1d(ys_true)
    )
    variants, sequences, ys_true = remove_gap_pos(
        gremlin.gaps, variants, sequences, ys_true,
        shift_pos=shift_pos, substitution_sep=substitution_sep
    )
    if not sequences:
        xs = []
    else:
        try:
            xs = gremlin.get_scores(sequences, encode=True)
        except RuntimeError:
            xs = []
    x_wt = gremlin.get_scores(np.atleast_1d(gremlin.wt_seq), encode=True)
    return xs, x_wt, variants, sequences, ys_true


def plmc_encoding(plmc: PLMC, variants, sequences, ys_true, threads=1, verbose=False):
    """
    Gets X and x_wt for DCA prediction: delta_E = np.subtract(X, x_wt),
    with X = encoded sequences of variants.
    Also removes variants, sequences, and y_trues at MSA gap positions.
    """
    target_seq, index = plmc.get_target_seq_and_index()
    wt_name = target_seq[0] + str(index[0]) + target_seq[0]
    if verbose:
        logger.info(f"Using to-self-substitution '{wt_name}' as wild type reference. "
              f"Encoding variant sequences. This might take some time...")
    x_wt = get_encoded_sequence(wt_name, plmc)
    if threads > 1 and USE_RAY:
        # Hyperthreading, NaNs are already being removed by the called function
        variants, sequences, xs, ys_true = get_dca_data_parallel(
            variants, sequences, ys_true, plmc, threads, verbose=verbose)
    else:
        x_ = plmc.collect_encoded_sequences(variants)
        # NaNs must still be removed
        xs, variants, sequences, ys_true = remove_nan_encoded_positions(
            x_, variants, sequences, ys_true
        )
    return xs, x_wt, variants, sequences, ys_true


def remove_gap_pos(
        gaps,
        variants,
        sequences,
        fitnesses,
        shift_pos=1,
        substitution_sep='/'
):
    """
    Remove gap postions from input variants, sequences, and fitness values
    based on input gaps (gap positions).
    Note that by default, gap positions are shifted by +1 to match the input
    variant identifiers (e.g., variant A123C is removed if gap pos is 122; (122 += 1).

    Returns
    -----------
    variants_v
        Variants with substitutions at valid sequence positions, 
        i.e., at non-gap positions
    sequences_v
        Sequences of variants with substitutions at valid sequence positions, 
        i.e., at non-gap positions
    fitnesses_v
        Fitness values of variants with substitutions at valid sequence positions, 
        i.e., at non-gap positions
    """
    variants_v, sequences_v, fitnesses_v = [], [], []
    valid = []
    for i, variant in enumerate(variants):
        variant = variant.split(substitution_sep)
        for var in variant:
            if int(var[1:-1]) not in [gap + shift_pos for gap in gaps]:
                if i not in valid:
                    valid.append(i)
                    variants_v.append(variants[i])
                    sequences_v.append(sequences[i])
                    fitnesses_v.append(fitnesses[i])
    return variants_v, sequences_v, fitnesses_v


def parse_llm_flag(llm: str | None) -> list[str]:
    """
    Parse the `--plm` input into a list of (lower-case) PLM names.
    One or multiple PLMs can be specified, combining them via '+', ',',
    or whitespace, e.g. '--plm esm', '--plm esm+prosst', or
    '--plm "esm, prosst"' for combined DCA+ESM+ProSST hybrid modeling.
    """
    if llm is None:
        return []
    return [name.strip().lower() for name in re.split(r'[+,\s]+', llm) if name.strip()]


def setup_llm_input(
        llm_names: list[str],
        sequences: list[str],
        wt_seq: str | None = None,
        pdb_file: str | None = None
) -> dict:
    """
    Build the (merged) PLM input dictionary for one or multiple PLMs,
    e.g. {'esm': {...}, 'prosst': {...}}, used as `llm_model_input`
    for the DCALLMHybridModel.
    """
    llm_dict = {}
    for name in llm_names:
        if name.startswith('esm'):
            # ESM has no dedicated WT input; fall back to first sequence
            esm_wt_seq = wt_seq if wt_seq is not None else sequences[0]
            llm_dict.update(esm_setup(esm_wt_seq, sequences))
        elif name == 'prosst':
            llm_dict.update(prosst_setup(wt_seq, pdb_file, sequences=sequences))
        else:
            raise RuntimeError(
                f"Unknown --plm option '{name}'. Supported PLMs are 'esm' and 'prosst' "
                f"(combine multiple via '+', e.g. --plm esm+prosst)."
            )
    return llm_dict


def performance_ls_ts(
        ls_fasta: str | None,
        ts_fasta: str | None,
        threads: int,
        params_file: str,
        model_pickle_file: str | None = None,
        llm: str | None = None,
        pdb_file: str | None = None,
        wt_seq: str | None = None,
        substitution_sep: str = '/',
        label=False,
        lora_train: bool = False,
        gauss_opt: bool = False,
        gauss_comb: bool = False,
        seed: int | None = None,
        device: str | None = None,
        progress_cb=None,
        abort_cb=None
):
    if device is None:
        device = get_device()
    test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)

    if ls_fasta is not None and ts_fasta is not None:
        train_sequences, train_variants, y_train = get_sequences_from_file(
            ls_fasta
        )
        (
            x_train, train_variants, train_sequences, 
            y_train, x_wt, _, model_type 
        ) = plmc_or_gremlin_encoding(
            train_variants, train_sequences, y_train, 
            params_file, substitution_sep, threads
        )

        (
            x_test, test_variants, test_sequences, y_test, *_
        ) = plmc_or_gremlin_encoding(
            test_variants, test_sequences, y_test, params_file, 
            substitution_sep, threads, verbose=False
        )

        logger.info(f"Initial training set variants: {len(train_sequences)}. "
                    f"Remaining: {len(train_variants)} (after removing "
                    f"substitutions at gap positions).\nInitial test set "
                    f"variants: {len(test_sequences)}. Remaining: " 
                    f"{len(test_variants)} (after removing substitutions "
                    f"at gap positions)."
        )
        llm_names = parse_llm_flag(llm)
        if llm_names:
            logger.info(f"Setting up PLM(s) for hybrid modeling: {', '.join(llm_names)}...")
            llm_dict = setup_llm_input(llm_names, train_sequences, wt_seq, pdb_file)
            # hybrid_prediction expects a dict {PLM_NAME: tokenized_test_sequences}
            x_llm_test = {
                llm_name: tokenize_sequences(
                    test_sequences, llm_data['llm_tokenizer'])[0]
                for llm_name, llm_data in llm_dict.items()
            }
        else:
            llm_dict = None
            x_llm_test = None
            llm = ''
        hybrid_model = DCALLMHybridModel(
            x_train_dca=np.array(x_train),
            y_train=np.array(y_train),
            llm_model_input=llm_dict,
            x_dca_wt=x_wt,
            sequences=list(train_sequences),
            wt_sequence=wt_seq,
            pdb_struct=pdb_file,
            lora_train=lora_train,
            gauss_opt=gauss_opt,
            gauss_comb_plm_emb=gauss_comb,
            seed=seed,
            device=device,
            progress_cb=progress_cb,
            abort_cb=abort_cb
        )
        y_test_pred, _indiv_preds = hybrid_model.hybrid_prediction(
            x_dca=np.array(x_test), x_llm_dict=x_llm_test, sequences=list(test_sequences)
        )
        logger.info(f'Hybrid performance: {spearmanr(y_test, y_test_pred)[0]:.3f} N={len(y_test)}')
        save_model_to_dict_pickle(hybrid_model, f'HYBRID{model_type}')

    elif (
        ts_fasta is not None and 
        model_pickle_file is not None 
        and params_file is not None
    ):
        # no LS provided but hybrid model provided for 
        # individual beta contributed zero shot predictions
        logger.info(f'Taking model from saved model (Pickle file): {model_pickle_file}...')
        model, model_type = get_model_and_type(model_pickle_file)
        if not model_type.startswith('Hybrid'):  # same as below in next elif
            (
                x_test, test_variants, test_sequences, 
                y_test, x_wt, *_
            ) = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, model_pickle_file, 
                substitution_sep, threads, False
            )
            y_test_pred = get_delta_e_statistical_model(x_test, x_wt)
        else:  # Hybrid model input requires params from plmc or GREMLIN model
            (
                x_test, test_variants, test_sequences, 
                y_test, *_
            ) = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, params_file,
                substitution_sep, threads, False
            )
            if model.llm_keys is not None:
                logger.info(f"Found hybrid model with PLM(s) {', '.join(model.llm_keys)}...")
                x_llm_test = {
                    llm_name: tokenize_sequences(
                        test_sequences, model.llm_data[llm_name]['llm_tokenizer'])[0]
                    for llm_name in model.llm_keys
                }
                y_test_pred, _ = model.hybrid_prediction(
                    x_test, x_llm_test, sequences=list(test_sequences)
                )
            else:
                y_test_pred, _ = model.hybrid_prediction(x_test)
    
    elif ts_fasta is not None and model_pickle_file is None:
        # no LS and *no hybrid model* provided:
        # statistical modeling / no ML / zero-shot PLM predictions
        logger.info(
            f"No learning set provided, falling back to statistical DCA model: "
            f"no adjustments of individual hybrid model parameters (\"beta's\")."
        )
        test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)
        logger.info(
            f"Initial test set variants: {len(test_sequences)}. "
            f"Remaining: {len(test_variants)} (after removing "
            f"substitutions at gap positions)."
        )
        if params_file is not None:
            logger.info("DCA inference on test set...")
            (
                x_test, test_variants, test_sequences, 
                y_test, x_wt, model, model_type
            ) = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, 
                params_file, substitution_sep, threads
            )
            y_test_pred = get_delta_e_statistical_model(x_test, x_wt)
            save_model_to_dict_pickle(model, model_type)
            model_type = f'{model_type}_no_ML'
        else:
            model_type = 'PLM'
            # Zero-shot supports a single PLM
            plm_names = parse_llm_flag(llm)
            plm_name = plm_names[0] if plm_names else None
            if plm_name is not None and plm_name.startswith('esm'):
                logger.info("Zero-shot PLM inference using ESM...")
                plm_dict = esm_setup(wt_seq, test_sequences)
                y_test_pred = plm_inference(
                    tokenized_sequences=plm_dict['esm']['x_llm'],
                    wt_input_ids=plm_dict['esm']['wt_input_ids'],
                    attention_mask=plm_dict['esm']['llm_attention_mask'],
                    model=plm_dict['esm']['llm_base_model'],
                    device=device
                ).cpu()
            elif plm_name == 'prosst':
                logger.info("Zero-shot PLM inference using ProSST...")
                plm_dict = prosst_setup(wt_seq, pdb_file, test_sequences)
                y_test_pred = plm_inference(
                    tokenized_sequences=plm_dict['prosst']['x_llm'],
                    wt_input_ids=plm_dict['prosst']['wt_input_ids'],
                    attention_mask=plm_dict['prosst']['llm_attention_mask'],
                    model=plm_dict['prosst']['llm_base_model'],
                    wt_structure_input_ids=plm_dict['prosst']['wt_structure_input_ids'],
                    device=device
                ).cpu()
            else:
                raise RuntimeError(
                    f"Unknown or unset --plm option: '{llm}'. Expected 'esm' or 'prosst'."
                )
    else:
        raise RuntimeError('No test set given for performance estimation.')
    if llm is None or llm == '':
        llm = ''
    else:
        llm = '_' + '_'.join(parse_llm_flag(llm)).upper()
    plot_y_true_vs_y_pred(
        np.array(y_test, dtype=np.float64), 
        np.array(y_test_pred, dtype=np.float64), 
        np.array(test_variants), 
        label=label, 
        hybrid=True, 
        name=f'{model_type}{llm}'
    )


def predict_ps(
    prediction_dict: dict,
    threads: int,
    separator: str,
    model_pickle_file: str | None = None,
    params_file: str | None = None,
    prediction_set: str | None = None,
    llm: str | None = None,
    pdb_file: str | None = None,
    wt_seq: str | None = None,
    negative: bool = False,
    device: str | None = None
):
    """
    Predicting the fitness of sequences of a prediction set or multiple prediction 
    sets (e.g. created with 'pypef mkps') using DCA, Hybrid, or PLM Zero-Shot models.
    """
    if device is None:
        device = get_device()
    dca_modeling = False
    if model_pickle_file is None and params_file is not None:
        model_pickle_file = params_file
        logger.info(f'Trying to load model from saved parameters (Pickle file): {model_pickle_file}...')
        dca_modeling = True
    elif params_file is not None:
        logger.info(f'Loading model from saved model (Pickle file {model_pickle_file})...')
        dca_modeling = True

    model_type = None
    model = None
    if dca_modeling:
        model, model_type = get_model_and_type(model_pickle_file)
        if model_type in ('PLMC', 'GREMLIN'):
            logger.info(f'Found {model_type} model file. No hybrid model provided - '
                        f'falling back to a statistical DCA model...')

    pmult = [
        'Recomb_Double_Split', 'Recomb_Triple_Split', 'Recomb_Quadruple_Split',
        'Recomb_Quintuple_Split', 'Diverse_Double_Split', 'Diverse_Triple_Split',
        'Diverse_Quadruple_Split'
    ]

    # Pre-setup tokenizers for Hybrid model predictions if required.
    # Reuse the tokenizers restored while loading the model (see get_model_and_type);
    # do NOT call setup_llm_input here — it reloads the PLMs and, being handed
    # sequences=None at this point, would fail during tokenization. It also expects
    # lower-case PLM names while model.llm_keys are upper-case ('ESM'/'PROSST').
    llm_dict = None
    if dca_modeling and model_type.startswith('Hybrid') and model.llm_keys:
        logger.info(f"Using PLM tokenizer(s) from the loaded hybrid model: {', '.join(model.llm_keys)}...")
        llm_dict = {
            llm_name: {'llm_tokenizer': model.llm_data[llm_name]['llm_tokenizer']}
            for llm_name in model.llm_keys
        }

    # --- Mode 1: Multi-file directory prediction sets ---
    if True in prediction_dict.values():
        for ps, path in zip(prediction_dict.values(), pmult):
            if not ps:
                continue

            logger.info(f'Running predictions for variant-sequence files in directory {path}...')
            all_y_v_pred = []
            files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.fasta')]

            for i, file in enumerate(files):
                logger.info(f'Processing file ({i + 1}/{len(files)}) for prediction...')
                file_path = os.path.join(path, file)
                sequences, variants, _ = get_sequences_from_file(file_path)

                if not dca_modeling:  # Zero-shot PLM inference
                    plm_names = parse_llm_flag(llm)
                    plm_name = plm_names[0] if plm_names else None
                    if plm_name is None:
                        raise ValueError("No model or parameters provided. Specify --plm for zero-shot PLM inference.")
                    model_type = f'PLM_{plm_name.upper()}'
                    if plm_name.startswith('esm'):
                        logger.info("Zero-shot PLM inference using ESM...")
                        plm_dict = esm_setup(wt_seq, sequences)
                        ys_pred = plm_inference(
                            tokenized_sequences=plm_dict['esm']['x_llm'],
                            wt_input_ids=plm_dict['esm']['wt_input_ids'],
                            attention_mask=plm_dict['esm']['llm_attention_mask'],
                            model=plm_dict['esm']['llm_base_model'],
                            device=device
                        ).cpu()
                    elif plm_name == 'prosst':
                        logger.info("Zero-shot PLM inference using ProSST...")
                        plm_dict = prosst_setup(wt_seq, pdb_file, sequences)
                        ys_pred = plm_inference(
                            tokenized_sequences=plm_dict['prosst']['x_llm'],
                            wt_input_ids=plm_dict['prosst']['wt_input_ids'],
                            attention_mask=plm_dict['prosst']['llm_attention_mask'],
                            model=plm_dict['prosst']['llm_base_model'],
                            wt_structure_input_ids=plm_dict['prosst']['wt_structure_input_ids'],
                            device=device
                        ).cpu()
                    else:
                        raise RuntimeError(f"Unknown --plm flag option: '{llm}'. Expected 'esm' or 'prosst'.")
                else:
                    # plmc_or_gremlin_encoding drops variants at gap / NaN
                    # positions the DCA model does not cover, so it returns a
                    # *filtered* variant list. Bind it back into test_variants
                    # and use that (NOT the raw `variants`) below, otherwise the
                    # predictions get paired with the wrong variant names.
                    if not model_type.startswith('Hybrid'):  # Statistical DCA
                        x_test, test_variants, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, params_file, threads=threads, verbose=False,
                            substitution_sep=separator
                        )
                        ys_pred = get_delta_e_statistical_model(x_test, x_wt)
                    else:  # Hybrid model
                        x_test, test_variants, test_sequences, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, params_file,
                            threads=threads, verbose=False, substitution_sep=separator
                        )
                        if not model.llm_keys:
                            ys_pred, _ = model.hybrid_prediction(x_test)
                        else:
                            test_seqs = [str(seq) for seq in test_sequences]
                            x_llm_test = {
                                llm_name: tokenize_sequences(
                                    test_seqs, l_data['llm_tokenizer'])[0]
                                for llm_name, l_data in llm_dict.items()
                            }
                            ys_pred, _ = model.hybrid_prediction(
                                np.asarray(x_test), x_llm_test, sequences=test_seqs
                            )

                assert len(test_variants) == len(ys_pred), (
                    f"Mismatch: {len(test_variants)} variants vs {len(ys_pred)} predictions."
                )
                for k in range(len(ys_pred)):
                    all_y_v_pred.append((ys_pred[k], test_variants[k]))

            all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=not negative)
            predictions_out(
                predictions=all_y_v_pred,
                model=model_type,
                prediction_set=f'Top{path}',
                path=path
            )

    # --- Mode 2: Single FASTA file prediction set ---
    elif prediction_set is not None:
        sequences, variants, _ = get_sequences_from_file(prediction_set)

        if not dca_modeling:  # Zero-shot PLM inference
            plm_names = parse_llm_flag(llm)
            plm_name = plm_names[0] if plm_names else None
            if plm_name is None:
                raise ValueError("No model or parameters provided. Specify --plm for zero-shot PLM inference.")
            model_type = f'PLM_{plm_name.upper()}'
            if plm_name.startswith('esm'):
                logger.info("Zero-shot PLM inference using ESM...")
                plm_dict = esm_setup(wt_seq, sequences)
                ys_pred = plm_inference(
                    tokenized_sequences=plm_dict['esm']['x_llm'],
                    attention_mask=plm_dict['esm']['llm_attention_mask'],
                    wt_input_ids=plm_dict['esm']['wt_input_ids'],
                    model=plm_dict['esm']['llm_base_model']
                ).cpu()
            elif plm_name == 'prosst':
                logger.info("Zero-shot PLM inference using ProSST...")
                plm_dict = prosst_setup(wt_seq, pdb_file, sequences)
                ys_pred = plm_inference(
                    tokenized_sequences=plm_dict['prosst']['x_llm'],
                    attention_mask=plm_dict['prosst']['llm_attention_mask'],
                    wt_input_ids=plm_dict['prosst']['wt_input_ids'],
                    model=plm_dict['prosst']['llm_base_model'],
                    wt_structure_input_ids=plm_dict['prosst']['wt_structure_input_ids'],
                    device=device
                ).cpu()
            else:
                raise RuntimeError(f"Unknown --plm flag option: '{llm}'. Expected 'esm' or 'prosst'.")
        else:
            if not model_type.startswith('Hybrid'):  # Statistical DCA
                xs, variants, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                    variants, sequences, None, params_file,
                    threads=threads, verbose=False, substitution_sep=separator
                )
                ys_pred = get_delta_e_statistical_model(xs, x_wt)
            else:  # Hybrid model
                xs, variants, sequences, *_ = plmc_or_gremlin_encoding(
                    variants, sequences, None, params_file,
                    threads=threads, verbose=True, substitution_sep=separator
                )
                if not model.llm_keys:
                    ys_pred, _ = model.hybrid_prediction(xs)
                else:
                    test_seqs = [str(seq) for seq in sequences]
                    xs_llm = {
                        llm_name: tokenize_sequences(
                            test_seqs, l_data['llm_tokenizer'])[0]
                        for llm_name, l_data in llm_dict.items()
                    }
                    ys_pred, _ = model.hybrid_prediction(
                        np.asarray(xs), xs_llm, sequences=test_seqs
                    )

        assert len(variants) == len(ys_pred), f"Mismatch: {len(variants)} variants vs {len(ys_pred)} predictions."
        y_v_pred = zip(ys_pred, variants)
        y_v_pred = sorted(y_v_pred, key=lambda x: x[0], reverse=not negative)

        predictions_out(
            predictions=y_v_pred,
            model=model_type,
            prediction_set=f'Top{prediction_set}'
        )


global_hybrid_model = None
global_hybrid_model_type = None


def predict_directed_evolution(
        encoder: str,
        variant: str,
        variant_sequence: str,
        hybrid_model_data_pkl: None | str
) -> Union[str, list]:
    """
    Perform directed in silico evolution and predict the fitness of a
    (randomly) selected variant using the hybrid model. This function opens
    the stored DCALLMHybridModel and the model parameters to predict the fitness
    of the variant encoded herein using the PLMC class. If the variant
    cannot be encoded (based on the PLMC params file), returns 'skip'. Else,
    returning the predicted fitness value and the variant name.
    """
    global global_hybrid_model, global_hybrid_model_type
    if hybrid_model_data_pkl is not None:
        if global_hybrid_model is None:
            global_hybrid_model, global_hybrid_model_type = get_model_and_type(
                hybrid_model_data_pkl
            )
            model, model_type = global_hybrid_model, global_hybrid_model_type
        else:
            model, model_type = global_hybrid_model, global_hybrid_model_type
    else:
        model_type = 'StatisticalModel'  # any name != 'Hybrid'

    if not model_type.startswith('Hybrid'):  # statistical DCA model
        xs, variant, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
            variant, variant_sequence, None, encoder, 
            verbose=False, use_global_model=True)
        if not list(xs):
            return 'skip'
        y_pred = get_delta_e_statistical_model(xs, x_wt)
    else:  # model_type == 'Hybrid': Hybrid model input requires params 
        # from PLMC or GREMLIN model plus optional PLM input
        xs, variant, variant_sequence, *_ = plmc_or_gremlin_encoding(
            variant, variant_sequence, None, encoder, 
            verbose=False, use_global_model=True
        )
        if not list(xs):
            return 'skip'
        try:
            if model.llm_keys is None:
                y_pred, _ = model.hybrid_prediction(
                    np.atleast_2d(xs), verbose=False
                )
            else:
                x_llm = {
                    llm_name: tokenize_sequences(
                        variant_sequence,
                        model.llm_data[llm_name]['llm_tokenizer'],
                        verbose=False)[0]
                    for llm_name in model.llm_keys
                }
                y_pred, _y_pred_indiv = model.hybrid_prediction(
                    np.atleast_2d(xs), x_llm,
                    sequences=list(variant_sequence), verbose=False
                )
        except ValueError as e:
            raise RuntimeError(
                f"Error: {e}\nProbably a different model was used for encoding than "
                "for modeling; e.g. using a HYBRIDgremlin model in "
                "combination with parameters taken from a PLMC file."
            )
    y_pred = float(y_pred[0])
    return y_pred, variant[0][1:]
