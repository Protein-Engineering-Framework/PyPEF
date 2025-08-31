# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Contains Python code used for the approach presented in our 'hybrid modeling' paper
# Preprint available at: https://doi.org/10.1101/2022.06.07.495081
# Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

from __future__ import annotations

import logging
logger = logging.getLogger('pypef.hybrid.hybrid_model')

import os
import pickle
from os import listdir
from os.path import isfile, join
from typing import Union
import warnings
import gc
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
from pypef.utils.variant_data import get_sequences_from_file, remove_nan_encoded_positions
import pypef.dca.plmc_encoding
from pypef.dca.plmc_encoding import PLMC, get_dca_data_parallel, get_encoded_sequence
from pypef.utils.to_file import predictions_out
from pypef.utils.helpers import get_device
from pypef.utils.plot import plot_y_true_vs_y_pred
import pypef.dca.gremlin_inference
from pypef.dca.gremlin_inference import GREMLIN, get_delta_e_statistical_model
from pypef.llm.esm_lora_tune import esm_setup, get_esm_models
from pypef.llm.prosst_lora_tune import get_prosst_models, prosst_setup
from pypef.llm.inference import llm_embedder, inference
from pypef.llm.utils import get_batches

# sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and 
# will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function 
# becomes public and is part of the scikit-learn developer API.
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')


def reduce_by_batch_modulo(a: np.ndarray, batch_size=5) -> np.ndarray:
    """
    Cuts input array by batch size modulo.
    """
    reduce = len(a) - (len(a) % batch_size)
    return a[:reduce]


# TODO: Add meta-learning model (e.g., learn2learn MAML learning option on PGym dataset)?
class DCALLMHybridModel:
    def __init__(
            self,
            x_train_dca: np.ndarray,
            y_train: np.ndarray,
            llm_model_input: dict | None = None,
            x_wt: np.ndarray | None = None,
            alphas: np.ndarray | None = None,
            parameter_range: list[tuple] | None = None,
            batch_size: int | None = None,
            llm_train: bool = True,
            device: str | None = None,
            seed: int | None = None,
            verbose: bool = True
    ):
        if llm_model_input is not None:
            if type(llm_model_input) is not dict:
                raise RuntimeError(f"Model input must be in form of a dictionary.")
            else:
                logger.info("Using LLM as second model next to DCA for hybrid modeling...")
                if len(list(llm_model_input.keys())) == 1 and list(llm_model_input.keys())[0] == 'esm1v':
                    self.llm_key = 'esm1v'
                    self.llm_base_model = llm_model_input['esm1v']['llm_base_model']
                    self.llm_model = llm_model_input['esm1v']['llm_model']
                    self.llm_optimizer = llm_model_input['esm1v']['llm_optimizer']
                    self.llm_train_function = llm_model_input['esm1v']['llm_train_function']
                    self.llm_inference_function = llm_model_input['esm1v']['llm_inference_function']
                    self.llm_loss_function = llm_model_input['esm1v']['llm_loss_function']
                    self.x_train_llm = llm_model_input['esm1v']['x_llm']
                    self.llm_attention_mask = llm_model_input['esm1v']['llm_attention_mask']
                elif len(list(llm_model_input.keys())) == 1 and list(llm_model_input.keys())[0] == 'prosst':
                    self.llm_key = 'prosst'
                    self.llm_base_model = llm_model_input['prosst']['llm_base_model']
                    self.llm_model = llm_model_input['prosst']['llm_model']
                    self.llm_optimizer = llm_model_input['prosst']['llm_optimizer']
                    self.llm_train_function = llm_model_input['prosst']['llm_train_function']
                    self.llm_inference_function = llm_model_input['prosst']['llm_inference_function']
                    self.llm_loss_function = llm_model_input['prosst']['llm_loss_function']
                    self.x_train_llm = llm_model_input['prosst']['x_llm']
                    self.llm_attention_mask = llm_model_input['prosst']['llm_attention_mask']
                    self.input_ids = llm_model_input['prosst']['input_ids']
                    self.structure_input_ids = llm_model_input['prosst']['structure_input_ids']
                else:
                    raise RuntimeError("LLM input model dictionary not supported. Currently supported "
                                      "models are 'esm1v' or 'prosst'")
                self.llm_model_input = llm_model_input
            if parameter_range is None:
                parameter_range = [(0, 1), (0, 1), (0, 1), (0, 1)] 
        else:
            logger.info("No LLM inputs were defined for hybrid modelling. "
                  "Using only DCA for hybrid modeling...")
            self.llm_key = None
            self.llm_model_input = None
            self.llm_attention_mask = None
            if parameter_range is None:
                parameter_range = [(0, 1), (0, 1)]
        if alphas is None:
            alphas = np.logspace(-6, 6, 100)
        self.parameter_range = parameter_range
        self.alphas = alphas
        self.x_train_dca = x_train_dca
        self.y_train = y_train
        self.x_wild_type = x_wt
        if device is None:
            device = get_device()
        self.device = device
        logger.info(f'Using device {device.upper()} for hybrid modeling...')
        self.seed = seed
        if batch_size is None:
            batch_size = 5
        self.batch_size = batch_size
        self.llm_train = llm_train
        self.verbose = verbose
        (
            self.ridge_opt, 
            self.beta1, 
            self.beta2, 
            self.beta3, 
            self.beta4,
            self.y_dca_ttest,
            self.y_dca_ridge_ttest,
            self.y_llm_ttest,
            self.y_llm_lora_ttest
        ) = None, None, None, None, None, None, None, None, None
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
        return Ridge(**grid.best_params_).fit(x_train, y_train)

    def _adjust_betas(
            self,
            y: np.ndarray,
            y_dca: np.ndarray,
            y_ridge: np.ndarray,
            y_llm: np.ndarray| None = None,
            y_llm_lora: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Find parameters that maximize the absolut Spearman rank
        correlation coefficient using differential evolution.

        Parameters
        ----------
        y : np.ndarray
            Array of fitness values.
        y_dca : np.ndarray
            Difference of the statistical energies of variants
            and wild-type.
        y_ridge : np.ndarray
            (Ridge) predicted fitness values of the variants.

        Returns
        -------
        'beta_1' and 'beta_2' that maximize the absolut Spearman rank correlation
        coefficient.
        """
        if y_llm is None or y_llm_lora is None:
            loss = lambda params: -np.abs(
                self.spearmanr(
                    y, 
                    params[0] * y_dca + 
                    params[1] * y_ridge
                )
            )
        else:
            if np.any(np.isnan(y_llm_lora)):
                logger.warning("y_llm_lora contains NaN's, weighting lora llm "
                               "hybrid model weight parameters with zero...")
                loss = lambda params: -np.abs(
                    self.spearmanr(
                        y, 
                        params[0] * y_dca + 
                        params[1] * y_ridge + 
                        params[2] * y_llm
                    )
                )
            else:
                loss = lambda params: -np.abs(
                    self.spearmanr(
                        y, 
                        params[0] * y_dca + 
                        params[1] * y_ridge + 
                        params[2] * y_llm + 
                        params[3] * y_llm_lora 
                    )
                )
        try:
            minimizer = differential_evolution(
                loss, bounds=self.parameter_range, tol=1e-4, rng=self.seed)
        except TypeError:  # SciPy v. 1.15.0 change: `seed` -> `rng` keyword
            minimizer = differential_evolution(
                loss, bounds=self.parameter_range, tol=1e-4, seed=self.seed)
        if y_llm is not None or y_llm_lora is not None:
            if np.any(np.isnan(y_llm_lora)):
                minimizer.x[-1] = 0.0
        return minimizer.x

    def get_subsplits_train(self, train_size_fit: float = 0.66):
        logger.info("Getting subsplits for supervised (re-)training of models "
              "and for adjustment of hybrid component contribution "
              "weights (\"beta's\")..."
        )
        train_size_fit = int(train_size_fit * len(self.y_train))
        train_size_beta_adjustment = len(self.y_train) - train_size_fit
        logger.info(f"Splitting training data of size {len(self.y_train)} "
              f"into {train_size_fit} variants for model tuning and "
              f"{train_size_beta_adjustment} variants for hybrid model "
              f"beta adjustment...")
        if len(self.parameter_range) == 4:
            # Reduce sizes by batch modulo
            n_drop = train_size_fit % self.batch_size
            if n_drop > 0:
                train_size_fit = train_size_fit - n_drop
                train_size_beta_adjustment = len(self.y_train) - train_size_fit
                logger.info(f"Shifting {n_drop} variants from training set to "
                      f"beta adjustment set to match batch requirements "
                      f"of batch size {self.batch_size} for LLM retraining "
                      f"resulting in {train_size_fit} variants for model "
                      f"tuning and {train_size_beta_adjustment} variants "
                      f"for hybrid model beta adjustment...")
            (
                self.x_dca_ttrain, self.x_dca_ttest, 
                self.x_llm_ttrain, self.x_llm_ttest,
                self.y_ttrain, self.y_ttest
            ) = train_test_split(
                self.x_train_dca, 
                self.x_train_llm,
                self.y_train, 
                train_size=train_size_fit,
                random_state=self.seed
            )
        else:
            (
                self.x_dca_ttrain, self.x_dca_ttest, 
                self.y_ttrain, self.y_ttest
            ) = train_test_split(
                self.x_train_dca,
                self.y_train, 
                train_size=train_size_fit,
                random_state=self.seed
            )
            #except ValueError:
            """
            Not enough sequences to construct a sub-training and sub-testing 
            set when splitting the training set.
            Machine learning/adjusting the parameters 'beta_1' and 'beta_2' not 
            possible -> return parameter setting for 'EVmutation/GREMLIN' model.
            """
            #return 1.0, 0.0, 1.0, 0.0, None
            """
            The sub-training set 'y_ttrain' is subjected to a five-fold cross 
            validation. This leads to the constraint that at least two sequences
            need to be in the 20 % of that set in order to allow a ranking. 
            If this is not given -> return parameter setting for 'EVmutation/GREMLIN' model.
            """
            # int(0.2 * len(y_ttrain)) due to 5-fold-CV for adjusting the (Ridge) regressor
            #y_ttrain_min_cv = int(0.2 * len(y_ttrain))
            #if y_ttrain_min_cv < 5:
            #    return 1.0, 0.0, 1.0, 0.0, None

    def train_llm(self):
        # LoRA training on y_llm_ttrain --> Testing on y_llm_ttest 
        x_llm_ttrain_b, scores_ttrain_b = (
            get_batches(self.x_llm_ttrain, batch_size=self.batch_size, dtype=int), 
            get_batches(self.y_ttrain, batch_size=self.batch_size, dtype=float)
        )

        if self.llm_key == 'prosst':
            y_llm_ttest = self.llm_inference_function(
                xs=self.x_llm_ttest,
                model=self.llm_base_model,
                input_ids=self.input_ids,
                attention_mask=self.llm_attention_mask,
                structure_input_ids=self.structure_input_ids,
                device=self.device
            )
            y_llm_ttrain = self.llm_inference_function(
                xs=self.x_llm_ttrain,
                model=self.llm_base_model,
                input_ids=self.input_ids,
                attention_mask=self.llm_attention_mask,
                structure_input_ids=self.structure_input_ids,
                device=self.device
            )
        elif self.llm_key == 'esm1v':
            x_llm_ttest_b = get_batches(self.x_llm_ttest, batch_size=1, dtype=int)
            y_llm_ttest = self.llm_inference_function(
                xs=x_llm_ttest_b,
                model=self.llm_model,
                attention_mask=self.llm_attention_mask,
                device=self.device
            )
            y_llm_ttrain = self.llm_inference_function(
                xs=x_llm_ttrain_b,
                model=self.llm_model,
                attention_mask=self.llm_attention_mask,
                device=self.device
            )
        logger.info(
            f"{self.llm_key.upper()} unsupervised performance: "
            f"Train set = {spearmanr(self.y_ttrain, y_llm_ttrain.detach().cpu())[0]:.3f}"
            f" (N={len(self.y_ttrain)}), "
            f"Test set = {spearmanr(self.y_ttest, y_llm_ttest.detach().cpu())[0]:.3f}"
            f" (N={len(self.y_ttest)})"
        )
        logger.info('Refining/training the model... gradient calculation adds a computational '
              'graph that requires quite some memory - if you are facing an (out of memory) '
              'error, try reducing the batch size or sticking to CPU device...')
        
        # void function, training model in place
        if self.llm_key == 'prosst':
            self.llm_train_function(
                x_llm_ttrain_b, 
                scores_ttrain_b,
                self.llm_loss_function,
                self.llm_model,
                self.llm_optimizer, 
                self.input_ids,
                self.llm_attention_mask,  
                self.structure_input_ids,
                n_epochs=50,
                device=self.device,
                verbose=self.verbose,
                raise_error_on_train_fail=False
            )
            y_llm_lora_ttrain = self.llm_inference_function(
                xs=self.x_llm_ttrain,
                model=self.llm_model,
                input_ids=self.input_ids,
                attention_mask=self.llm_attention_mask,
                structure_input_ids=self.structure_input_ids,
                device=self.device,
                verbose=self.verbose
            )
            y_llm_lora_ttest = self.llm_inference_function(
                xs=self.x_llm_ttest,
                model=self.llm_model,
                input_ids=self.input_ids,
                attention_mask=self.llm_attention_mask,
                structure_input_ids=self.structure_input_ids,
                device=self.device,
                verbose=self.verbose
            )
        elif self.llm_key == 'esm1v':
            # xs, attns, scores, loss_fn, model, optimizer
            self.llm_train_function(
                x_llm_ttrain_b, 
                self.llm_attention_mask,
                scores_ttrain_b,
                self.llm_loss_function,
                self.llm_model,
                self.llm_optimizer,  
                n_epochs=5, 
                device=self.device,
                verbose=self.verbose
            )
            y_llm_lora_ttrain = self.llm_inference_function(
                xs=x_llm_ttrain_b,
                model=self.llm_model,
                attention_mask=self.llm_attention_mask,
                device=self.device,
                verbose=self.verbose
            )
            y_llm_lora_ttest = self.llm_inference_function(
                xs=x_llm_ttest_b,
                model=self.llm_model,
                attention_mask=self.llm_attention_mask,
                device=self.device,
                verbose=self.verbose
            )
        logger.info(
            f"{self.llm_key.upper()} supervised tuned performance: "
            f"Train = {spearmanr(self.y_ttrain, y_llm_lora_ttrain.detach().cpu())[0]:.3f}"
            f" (N={len(self.y_ttrain)}), "
            f"Test = {spearmanr(self.y_ttest, y_llm_lora_ttest.detach().cpu())[0]:.3f}"
            f" (N={len(self.y_ttest)})"
        )

        self.y_llm_ttest = y_llm_ttest.detach().cpu().numpy()
        self.y_llm_lora_ttest = y_llm_lora_ttest.detach().cpu().numpy()

    def train_and_optimize(self) -> tuple:
        """
        Get the adjusted parameters 'beta_1', 'beta_2', and the
        tuned regressor of the hybrid model.

        Parameters
        ----------
        x_train : np.ndarray
            Encoded sequences of the variants in the training set.
        y_train : np.ndarray
            Fitness values of the variants in the training set.
        train_size_fit : float [0,1] (default 0.66)
            Fraction to split training set into another
            training and testing set.
        random_state : int (default=224)
            Random state used to split.

        Returns
        -------
        Tuple containing the adjusted parameters 'beta_1' and 'beta_2',
        as well as the tuned regressor of the hybrid model.
        """
        self.get_subsplits_train()
        self.y_dca_ttest = self._delta_e(self.x_dca_ttest)
        self.ridge_opt = self.ridge_predictor(self.x_dca_ttrain, self.y_ttrain)
        self.y_dca_ridge_ttest = self.ridge_opt.predict(self.x_dca_ttest)

        if len(self.parameter_range) == 4:
            self.train_llm()
            self.beta1, self.beta2, self.beta3, self.beta4 = self._adjust_betas(
               self.y_ttest, self.y_dca_ttest, self.y_dca_ridge_ttest, 
               self.y_llm_ttest, self.y_llm_lora_ttest
            )
            return self.beta1, self.beta2, self.beta3, self.beta4, self.ridge_opt
        
        else:
            self.beta1, self.beta2 = self._adjust_betas(self.y_ttest, 
                self.y_dca_ttest, self.y_dca_ridge_ttest
            )
            return self.beta1, self.beta2, self.ridge_opt

    def hybrid_prediction(
            self,
            x_dca: np.ndarray,
            x_llm: None | np.ndarray = None,
            verbose: bool = True
    ) -> np.ndarray:
        """
        Use the regressor 'reg' and the parameters 'beta_1'
        and 'beta_2' for constructing a hybrid model and
        predicting the fitness associates of 'X'.

        Parameters
        ----------
        x : np.ndarray
            Encoded sequences X used for prediction.
        reg : object
            Tuned ridge regressor for the hybrid model.
        beta_1 : float
            Float for regulating EVmutation model contribution.
        beta_2 : float
            Float for regulating Ridge regressor contribution.

        Returns
        -------
        Predicted fitness associates of 'X' using the
        hybrid model.
        """
        y_dca = self._delta_e(x_dca)
        if self.ridge_opt is None:
            y_ridge = np.zeros(len(y_dca))  # in order to suppress error
        else:
            y_ridge = self.ridge_opt.predict(x_dca)

        if x_llm is None:
            if self.llm_attention_mask is not None:
                logger.info('No LLM input for hybrid prediction but the model '
                      'has been trained using an LLM model input.. '
                      'Using only DCA for hybridprediction.. This can lead '
                      'to unwanted prediction behavior if the hybrid model '
                      'is trained including an LLM...')
            return self.beta1 * y_dca + self.beta2 * y_ridge
        
        else:
            if self.llm_key == 'prosst':
                y_llm = self.llm_inference_function(
                    x_llm, 
                    self.llm_base_model, 
                    self.input_ids,
                    self.llm_attention_mask, 
                    self.structure_input_ids,
                    verbose=verbose,
                    device=self.device).detach().cpu().numpy()
                y_llm_lora = self.llm_inference_function(
                    x_llm, 
                    self.llm_model, 
                    self.input_ids,
                    self.llm_attention_mask, 
                    self.structure_input_ids,
                    verbose=verbose,
                    device=self.device).detach().cpu().numpy()
            elif self.llm_key == 'esm1v':
                x_llm_b = get_batches(x_llm, batch_size=1, dtype=int)
                y_llm = self.llm_inference_function(
                    x_llm_b, 
                    self.llm_attention_mask,
                    self.llm_base_model, 
                    verbose=verbose,
                    device=self.device).detach().cpu().numpy()
                y_llm_lora = self.llm_inference_function(
                    x_llm_b, 
                    self.llm_attention_mask,
                    self.llm_model, 
                    verbose=verbose,
                    device=self.device).detach().cpu().numpy()
            if np.any(np.isnan(y_llm)) or np.any(np.isnan(y_llm_lora)):
                logger.warning(
                    f"LLM predictions contains NaN's... replacing NaN's with "
                    f"zeros (optimized hybrid model weights: "
                    f"{self.beta1}, {self.beta2}, {self.beta3}, {self.beta4})..."
                )
                y_llm = np.nan_to_num(y_llm_lora, nan=0.0)
                y_llm_lora = np.nan_to_num(y_llm_lora, nan=0.0)
            return (
                self.beta1 * y_dca + self.beta2 * y_ridge + 
                self.beta3 * y_llm + self.beta4 * y_llm_lora
            )

    def ls_ts_performance(self):
        beta_1, beta_2, reg = self.settings(
            x_train=self.x_train,
            y_train=self.y_train
        )
        spearman_r = self.spearmanr(
            self.y_test,
            self.hybrid_prediction(self.x_test, reg, beta_1, beta_2)
        )
        self.beta_1, self.beta_2, self.regressor = beta_1, beta_2, reg
        return spearman_r, reg, beta_1, beta_2


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
        if model.llm_key == 'esm1v':
            logger.info("Found hybrid model with ESM1v LLM model...")
            base_model, lora_model, _tokenizer, _optimizer = get_esm_models()
            model_type += '_ESM1v'
        elif model.llm_key == 'prosst':
            logger.info("Found hybrid model with ProSST LLM model...")
            base_model, lora_model, _tokenizer, _optimizer = get_prosst_models()
            model_type += '_ProSST'
        else:
            logger.info("Found hybrid model without LLM model...")
            return model, model_type
        base_model.load_state_dict(model.llm_base_model)
        lora_model.load_state_dict(model.llm_model)
        model.llm_model = lora_model
        model.llm_base_model = base_model
        model.llm_model.eval()
        model.llm_base_model.eval()

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
    
    # For Hybrid LLM models save as model.state_dict()
    if model_type.lower().startswith('hybrid'):
        if model.llm_key is not None:
            logger.info(f"Storing LLM model {model.llm_key.upper()} "
                  f"of hybrid model as state dictionaries...")
            model.llm_model = model.llm_model.to('cpu')
            model.llm_model = model.llm_model.state_dict()
            model.llm_base_model = model.llm_base_model.to('cpu')
            model.llm_base_model = model.llm_base_model.state_dict()
            model.llm_model_input[model.llm_key]['llm_base_model'] = None
            model.llm_model_input[model.llm_key]['llm_model'] = None
            model_type += model.llm_key.upper()
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
        ys_true = np.zeros(np.shape(sequences))
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
                f"are removed as these would be predicted with "
                f"wild-type fitness:"
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
        device: str| None = None
):
    test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)

    if ls_fasta is not None and ts_fasta is not None:
        train_sequences, train_variants, y_train = get_sequences_from_file(
            ls_fasta)
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
        if llm is not None:
            if llm.lower().startswith('esm'):
                llm_dict = esm_setup(train_sequences)
                x_llm_test = llm_embedder(llm_dict, test_sequences)
            elif llm.lower() == 'prosst':
                llm_dict = prosst_setup(
                    wt_seq, pdb_file, sequences=train_sequences)
                x_llm_test = llm_embedder(llm_dict, test_sequences)
        else:
            llm_dict = None
            x_llm_test = None
            llm = ''
        hybrid_model = DCALLMHybridModel(
            x_train_dca=np.array(x_train),
            y_train=np.array(y_train),
            llm_model_input=llm_dict,
            x_wt=x_wt,
            device=device
        )
        y_test_pred = hybrid_model.hybrid_prediction(np.array(x_test), x_llm_test)
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
            if model.llm_model_input is not None:
                logger.info(f"Found hybrid model with LLM {list(model.llm_model_input.keys())[0]}...")
                x_llm_test = llm_embedder(model.llm_model_input, test_sequences)
                y_test_pred = model.hybrid_prediction(x_test, x_llm_test)
            else:
                y_test_pred = model.hybrid_prediction(x_test)
    
    elif ts_fasta is not None and model_pickle_file is None:
        # no LS and *no hybrid model* provided:
        # statistical modeling / no ML / zero-shot LLM predictions
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
            model_type = 'LLM'
            if llm == 'esm':
                logger.info("Zero-shot LLM inference on test set using ESM1v...")
                y_test_pred = inference(test_sequences, llm)
            elif llm == 'prosst':
                logger.info("Zero-shot LLM inference on test set using ProSST...")
                y_test_pred = inference(test_sequences, llm, pdb_file=pdb_file, wt_seq=wt_seq)
            else:
                raise RuntimeError("Unknown --llm flag option.")
    else:
        raise RuntimeError('No test set given for performance estimation.')
    if llm is None or llm == '':
        llm = ''
    else:
        llm = f"_{llm.upper()}"
    plot_y_true_vs_y_pred(
        np.array(y_test), np.array(y_test_pred), np.array(test_variants), 
        label=label, hybrid=True, name=f'{model_type}{llm}'
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
        negative: bool = False
):
    """
    Description
    -----------
    Predicting the fitness of sequences of a prediction set
    or multiple prediction sets that were exemplary created with
    'pypef mkps' based on single substitutional variant data
    provided in a CSV and the wild type sequence:
        pypef mkps --wt WT_SEQ --input CSV_FILE
        [--drop THRESHOLD] [--drecomb] [--trecomb] [--qarecomb] [--qirecomb]
        [--ddiverse] [--tdiverse] [--qdiverse]

    Parameters
    -----------
    prediction_dict: dict
        Contains arguments which directory to predict, e.g. {'drecomb': True},
        than predicts prediction files that are present in this directory, e.g.
        in directory './Recomb_Double_Split'.
    params_file: str
        PLMC/GREMLIN couplings parameter file
    threads: int
        Threads used for parallelization for DCA-based sequence encoding
    separator: str
        Separator of individual substitution of variants, default '/'
    model_pickle_file: str
        Pickle file containing the hybrid model and model parameters in
        a dictionary format
    test_set: str = None
        Test set for prediction and plotting of predictions (contains
        true fitness values of variants).
    prediction_set: str = None
        Prediction set for prediction, does not contain true fitness values.
    figure: str = None
        Plotting the test set predictions and the corresponding true fitness
        values.
    label: bool = False
        If True, plots associated variant names of predicted variants.
    negative: bool = False
        If true, negative defines improved variants having a reduced/negative
        fitness compared to wild type.


    Returns
    -----------
    ()
        Writes sorted predictions to files (for [--drecomb] [--trecomb]
        [--qarecomb] [--qirecomb] [--ddiverse] [--tdiverse] [--qdiverse]
        in the respective created folders).

    """
    dca_modeling = False
    if model_pickle_file is None and params_file is not None:
        model_pickle_file = params_file
        logger.info(f'Trying to load model from saved parameters (Pickle file): {model_pickle_file}...')
        dca_modeling = True
    elif params_file is not None:
        logger.info(f'Loading model from saved model (Pickle file {model_pickle_file})...')
        dca_modeling = True
    if dca_modeling:
        model, model_type = get_model_and_type(model_pickle_file)
        if model_type == 'PLMC' or model_type == 'GREMLIN':
            logger.info(f'Found {model_type} model file. No hybrid model provided - '
                        f'falling back to a statistical DCA model...')

    pmult = [
        'Recomb_Double_Split', 'Recomb_Triple_Split', 'Recomb_Quadruple_Split',
        'Recomb_Quintuple_Split', 'Diverse_Double_Split', 'Diverse_Triple_Split',
        'Diverse_Quadruple_Split'
    ]
    if True in prediction_dict.values():
        for ps, path in zip(prediction_dict.values(), pmult):
            if ps:  # if True, run prediction in this directory, e.g. for drecomb
                logger.info(f'Running predictions for variant-sequence files in directory {path}...')
                all_y_v_pred = []
                files = [f for f in listdir(path) if isfile(join(path, f)) if f.endswith('.fasta')]
                for i, file in enumerate(files):  # collect and predict for each file in the directory
                    logger.info(f'Encoding files ({i + 1}/{len(files)}) for prediction...')
                    file_path = os.path.join(path, file)
                    sequences, variants, _ = get_sequences_from_file(file_path)
                    if not model_type.startswith('Hybrid'):
                        x_test, _, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, model, threads=threads, verbose=False,
                            substitution_sep=separator)
                        ys_pred = get_delta_e_statistical_model(x_test, x_wt)
                    else:  # Hybrid model input requires params from plmc or GREMLIN model plus optional LLM input
                        x_test, _test_variants, test_sequences, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, params_file,
                            threads=threads, verbose=False, substitution_sep=separator
                        )
                        if model.llm_key is None:
                            ys_pred = model.hybrid_prediction(x_test)
                        else:
                            sequences = [str(seq) for seq in test_sequences]
                            x_llm_test = llm_embedder(model.llm_model_input, sequences)
                            ys_pred = model.hybrid_prediction(np.asarray(x_test), np.asarray(x_llm_test))
                    for k, y in enumerate(ys_pred):
                        all_y_v_pred.append((ys_pred[k], variants[k]))
                if negative:  # sort by fitness value
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=False)
                else:
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=True)
                predictions_out(
                    predictions=all_y_v_pred,
                    model=model_type,
                    prediction_set=f'Top{path}',
                    path=path
                )
            else:  # check next task to do, e.g., predicting triple substituted variants, e.g. trecomb
                continue

    elif prediction_set is not None:  # Predicting single FASTA file sequences
        sequences, variants, _ = get_sequences_from_file(prediction_set)
        # NaNs are already being removed by the called function
        if not dca_modeling:  # model_pickle_file is None and params_file is None:
            # *No hybrid model* and no DCA params provided:
            # Zero-shot LLM predictions
            if llm == 'esm':
                model_type = 'LLM_ESM1v'
                logger.info("Zero-shot LLM inference on test set using ESM1v...")
                ys_pred = inference(sequences, llm)
            elif llm == 'prosst':
                model_type = 'LLM_ProSST'
                logger.info("Zero-shot LLM inference on test set using ProSST...")
                ys_pred = inference(sequences, llm, pdb_file=pdb_file, wt_seq=wt_seq)
        else:
            if not model_type.startswith('Hybrid'):  # statistical DCA model
                xs, variants, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                    variants, sequences, None, params_file,
                    threads=threads, verbose=False, substitution_sep=separator
                )
                ys_pred = get_delta_e_statistical_model(xs, x_wt)
            else:  # Hybrid model input requires params from plmc or GREMLIN model plus optional LLM input
                xs, variants, sequences, *_ = plmc_or_gremlin_encoding(
                    variants, sequences, None, params_file,
                    threads=threads, verbose=True, substitution_sep=separator
                )
                if model.llm_key is None:
                    ys_pred = model.hybrid_prediction(xs)
                else:
                    sequences = [str(seq) for seq in sequences]
                    xs_llm = llm_embedder(model.llm_model_input, sequences)
                    ys_pred = model.hybrid_prediction(np.asarray(xs), np.asarray(xs_llm))
            assert len(xs) == len(variants) == len(ys_pred)
        y_v_pred = zip(ys_pred, variants)
        y_v_pred = sorted(y_v_pred, key=lambda x: x[0], reverse=True)
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
                hybrid_model_data_pkl)
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
        # from PLMC or GREMLIN model plus optional LLM input
        xs, variant, variant_sequence, *_ = plmc_or_gremlin_encoding(
            variant, variant_sequence, None, encoder, 
            verbose=False, use_global_model=True
        )
        if not list(xs):
            return 'skip'
        try:
            if model.llm_model_input is None:
                y_pred = model.hybrid_prediction(xs)
            else:
                x_llm = llm_embedder(model.llm_model_input, 
                                     variant_sequence, verbose=False)

                y_pred = model.hybrid_prediction(
                    np.atleast_2d(xs), 
                    np.atleast_2d(x_llm), verbose=False
                )[0]
        except ValueError as e:
            raise RuntimeError(
                f"Error: {e}\nProbably a different model was used for encoding than "
                "for modeling; e.g. using a HYBRIDgremlin model in "
                "combination with parameters taken from a PLMC file."
            )
    y_pred = float(y_pred)

    return [(y_pred, variant[0][1:])]
