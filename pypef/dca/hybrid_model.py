#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @authors: Niklas Siedhoff, Alexander-Maurice Illig
# @contact: <n.siedhoff@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

# Contains Python code used for the approach presented in our 'hybrid modeling' paper
# Preprint available at: https://doi.org/10.1101/2022.06.07.495081
# Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

from __future__ import annotations

import os
import pickle
from os import listdir
from os.path import isfile, join
from typing import Union
import logging
logger = logging.getLogger('pypef.dca.hybrid_model')

import numpy as np
import sklearn.base
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.optimize import differential_evolution

from pypef.utils.variant_data import (
    get_sequences_from_file, get_seqs_from_var_name,
    remove_nan_encoded_positions, get_wt_sequence, split_variants
)

from pypef.dca.plmc_encoding import PLMC, get_dca_data_parallel, get_encoded_sequence, EffectiveSiteError
from pypef.utils.to_file import predictions_out
from pypef.utils.plot import plot_y_true_vs_y_pred
import pypef.dca.gremlin_inference
from pypef.dca.gremlin_inference import GREMLIN


class DCAHybridModel:
    alphas = np.logspace(-6, 6, 100)  # Grid for the parameter 'alpha'.
    parameter_range = [(0, 1), (0, 1)]  # Parameter range of 'beta_1' and 'beta_2' with lb <= x <= ub
    # TODO: Implementation of other regression techniques (CVRegression models)

    def __init__(
            self,
            alphas=alphas,
            parameter_range=None,
            x_train: np.ndarray = None,
            y_train: np.ndarray = None,
            x_test: np.ndarray = None,  # not necessary for training
            y_test: np.ndarray = None,  # not necessary for training
            x_wt=None
    ):
        if parameter_range is None:
            parameter_range = parameter_range
        self._alphas = alphas
        self._parameter_range = parameter_range
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.X = np.concatenate((x_train, x_test), axis=0) if self.x_test is not None else self.x_train
        self.y = np.concatenate((y_train, y_test), axis=0) if self.y_test is not None else self.y_train
        self.x_wild_type = x_wt
        self._spearmanr_dca = self._spearmanr_dca()
        self.beta_1, self.beta_2, self.regressor = self.settings(self.x_train, self.y_train)

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
        with E = \sum_{i} h_i (o_i) + \sum_{i<j} J_{ij} (o_i, o_j)

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
        y_dca = self._delta_e(self.X)
        return self.spearmanr(self.y, y_dca)

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
        grid = GridSearchCV(Ridge(), {'alpha': self._alphas}, cv=5)
        grid.fit(x_train, y_train)
        return Ridge(**grid.best_params_).fit(x_train, y_train)

    def _y_hybrid(
            self,
            y_dca: np.ndarray,
            y_ridge: np.ndarray,
            beta_1: float,
            beta_2: float
    ) -> np.ndarray:
        """
        Chooses sign for connecting the parts of the hybrid model.

        Parameters
        ----------
        y_dca : np.ndarray
            Difference of the statistical energies of variants
            and wild-type.
        y_ridge : np.ndarray
            (Ridge) predicted fitness values of the variants.
        b1 : float
            Float between [0,1] coefficient for regulating DCA
            model contribution.
        b2 : float
            Float between [0,1] coefficient for regulating ML
            model contribution.

        Returns
        -------
        The predicted fitness value-representatives of the hybrid
        model.
        """
        # Uncomment lines below to see if correlation between
        # y_true and y_dca is positive or negative:
        # logger.info(f'Positive or negative correlation of (all data) y_true '
        #             f'and y_dca (+/-?): {self._spearmanr_dca:.3f}')
        if self._spearmanr_dca >= 0:
            return beta_1 * y_dca + beta_2 * y_ridge
        else:  # negative correlation
            return beta_1 * y_dca - beta_2 * y_ridge

    def _adjust_betas(
            self,
            y: np.ndarray,
            y_dca: np.ndarray,
            y_ridge: np.ndarray
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
        loss = lambda b: -np.abs(self.spearmanr(y, b[0] * y_dca + b[1] * y_ridge))
        minimizer = differential_evolution(loss, bounds=self.parameter_range, tol=1e-4)
        return minimizer.x

    def settings(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            train_size_fit=0.66,
            random_state=42
    ) -> tuple:
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
        try:
            X_ttrain, X_ttest, y_ttrain, y_ttest = train_test_split(
                x_train, y_train,
                train_size=train_size_fit,
                random_state=random_state
            )

        except ValueError:
            """
            Not enough sequences to construct a sub-training and sub-testing 
            set when splitting the training set.

            Machine learning/adjusting the parameters 'beta_1' and 'beta_2' not 
            possible -> return parameter setting for 'EVmutation' model.
            """
            return 1.0, 0.0, None

        """
        The sub-training set 'y_ttrain' is subjected to a five-fold cross 
        validation. This leads to the constraint that at least two sequences
        need to be in the 20 % of that set in order to allow a ranking. 

        If this is not given -> return parameter setting for 'EVmutation' model.
        """
        y_ttrain_min_cv = int(0.2 * len(y_ttrain))  # 0.2 because of five-fold cross validation (1/5)
        if y_ttrain_min_cv < 2:
            return 1.0, 0.0, None

        y_dca_ttest = self._delta_e(X_ttest)

        ridge = self.ridge_predictor(X_ttrain, y_ttrain)
        y_ridge_ttest = ridge.predict(X_ttest)

        beta1, beta2 = self._adjust_betas(y_ttest, y_dca_ttest, y_ridge_ttest)
        return beta1, beta2, ridge

    def hybrid_prediction(
            self,
            x: np.ndarray,
            reg: object,  # any regression-based estimator (from sklearn)
            beta_1: float,
            beta_2: float
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
        y_dca = self._delta_e(x)
        if reg is None:
            y_ridge = np.random.random(len(y_dca))  # in order to suppress error
        else:
            y_ridge = reg.predict(x)
        # adjusting: + or - on all data --> +-beta_1 * y_dca + beta_2 * y_ridge
        return self._y_hybrid(y_dca, y_ridge, beta_1, beta_2)

    def split_performance(
            self,
            train_size: float = 0.8,
            n_runs: int = 10,
            seed: int = 42,
            save_model: bool = False
    ) -> dict:
        """
        Estimates performance of the model.

        Parameters
        ----------
        train_size : int or float (default=0.8)
            Number of samples in the training dataset
            or fraction of full dataset used for training.
        n_runs : int (default=10)
            Number of different splits to perform.
        seed : int (default=42)
            Seed for random generator.
        save_model : bool (default=False)
            If True, model is saved using pickle, else not.

        Returns
        -------
        data : dict
            Contains information about hybrid model parameters
            and performance results.
        """
        data = {}
        np.random.seed(seed)

        for r, random_state in enumerate(np.random.randint(100, size=n_runs)):
            x_train, x_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=train_size, random_state=random_state)
            beta_1, beta_2, reg = self.settings(x_train, y_train)
            if beta_2 == 0.0:
                alpha = np.nan
            else:
                if save_model:
                    pickle.dumps(reg)
                alpha = reg.alpha
            data.update(
                {f'{len(y_train)}_{r}':
                    {
                        'no_run': r,
                        'n_y_train': len(y_train),
                        'n_y_test': len(y_test),
                        'rnd_state': random_state,
                        'spearman_rho': self.spearmanr(
                            y_test, self.hybrid_prediction(
                                x_test, reg, beta_1, beta_2
                            )
                        ),
                        'beta_1': beta_1,
                        'beta_2': beta_2,
                        'alpha': alpha
                    }
                }
            )

        return data

    def ls_ts_performance(self):
        beta_1, beta_2, reg = self.settings(
            x_train=self.x_train,
            y_train=self.y_train
        )
        spearman_r = self.spearmanr(
            self.y_test,
            self.hybrid_prediction(self.x_test, reg, beta_1, beta_2)
        )
        return spearman_r, reg, beta_1, beta_2

    def train_and_test(
            self,
            train_percent_fit: float = 0.66,
            random_state: int = 42
    ):
        """
        Description
        ----------
        Trains the hybrid model on a relative number of all variants
        and returns the individual model contribution weights beta_1 (DCA)
        and beta_2 (ML) as well as the hyperparameter-tuned regression model,
        e.g. to save all the hybrid model parameters for later loading as
        Pickle file.

        Parameters
        ----------
        train_percent_fit: float (default = 0.66)
            Relative number of variants used for model fitting (not
            hyperparameter validation. Default of 0.66 and overall train
            size of 0.8 means the total size for least squares fitting
            is 0.8 * 0.66 = 0.528, thus for hyperparameter validation
            the size is 0.8 * 0.33 = 0.264 and for testing the size is
            1 - 0.528 - 0.264 = 0.208.
        random_state: int (default = 42)
            Random state for splitting (and reproduction of results).

        Returns
        ----------
        beta_1: float
            DCA model contribution to hybrid model predictions.
        beta_2: float
            ML model contribution to hybrid model predictions.
        reg: object
            sklearn Estimator class, e.g. sklearn.linear_model.Ridge
            fitted and with optimized hyperparameters (e.g. alpha).
        self._spearmanr_dca: float
            To determine, if spearmanr_dca (i.e. DCA correlation) and measured
            fitness values is positive (>= 0) or negative (< 0).
        test_spearman_r : float
            Achieved performance in terms of Spearman's rank correlation
            between measured and predicted test set variant fitness values.
        """
        beta_1, beta_2, reg = self.settings(
            x_train=self.x_train,
            y_train=self.y_train,
            train_size_fit=train_percent_fit,
            random_state=random_state
        )

        if len(self.y_test) > 0:
            test_spearman_r = self.spearmanr(
                self.y_test,
                self.hybrid_prediction(
                    self.x_test, reg, beta_1, beta_2
                )
            )
        else:
            test_spearman_r = None
        return beta_1, beta_2, reg, self._spearmanr_dca, test_spearman_r

    def get_train_sizes(self) -> np.ndarray:
        """
        Generates a list of train sizes to perform low-n with.

        Returns
        -------
        Numpy array of train sizes up to 80% (i.e. 0.8 * N_variants).
        """
        eighty_percent = int(len(self.y) * 0.8)

        train_sizes = np.sort(np.concatenate([
            np.arange(15, 50, 5), np.arange(50, 100, 10),
            np.arange(100, 150, 20), [160, 200, 250, 300, eighty_percent],
            np.arange(400, 1100, 100)
        ]))

        idx_max = np.where(train_sizes >= eighty_percent)[0][0] + 1
        return train_sizes[:idx_max]

    def run(
            self,
            train_sizes: list = None,
            n_runs: int = 10
    ) -> dict:
        """

        Returns
        ----------
        data: dict
            Performances of the split with size of the
            training set = train_size and size of the
            test set = N_variants - train_size.
        """
        data = {}
        for t, train_size in enumerate(train_sizes):
            logger.info(f'{t + 1}/{len(train_sizes)}:{train_size}')
            data.update(self.split_performance(train_size=train_size, n_runs=n_runs))
        return data


"""
Below: Some helper functions that call or are dependent on the DCAHybridModel class.
"""


def check_model_type(model: dict | DCAHybridModel | PLMC | GREMLIN):
    """
    Checks type/instance of model.
    """
    if type(model) == dict:
        try:
            model = model['model']
        except KeyError:
            raise SystemError("Unknown model dictionary taken from Pickle file.")
    if type(model) == pypef.dca.plmc_encoding.PLMC:
        return 'PLMC'
    elif type(model) == pypef.dca.hybrid_model.DCAHybridModel:
        return 'Hybrid'
    elif type(model) == pypef.dca.gremlin_inference.GREMLIN:
        return 'GREMLIN'
    elif isinstance(model, sklearn.base.BaseEstimator):
        raise SystemError("Loaded an sklearn ML model. For pure ML-based modeling the "
                          "\'ml\' flag has to be used instead of the \'hybrid\' flag.")
    else:
        raise SystemError('Unknown model/unknown Pickle file.')


def get_model_path(model: str):
    try:
        if isfile(model):
            model_path = model
        elif isfile(f'Pickles/{model}'):
            model_path = f'Pickles/{model}'
        else:
            raise SystemError("Did not find specified model file.")
        return model_path
    except TypeError:
        raise SystemError("No provided model. "
                          "Specify a model for DCA-based encoding.")


def get_model_and_type(params_file: str, substitution_sep: str = '/'):
    file_path = get_model_path(params_file)
    try:
        with open(file_path, 'rb') as read_pkl_file:
            model = pickle.load(read_pkl_file)
            model_type = check_model_type(model)
    except pickle.UnpicklingError:
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

    return model, model_type


def save_model_to_dict_pickle(
        model: DCAHybridModel | PLMC | GREMLIN,
        model_type: str | None = None,
        beta_1: float | None = None,
        beta_2: float | None = None,
        spearman_r: float | None = None,
        regressor: sklearn.base.BaseEstimator = None
):
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass

    if model_type is None:
        model_type = 'MODEL'
    # else:
    #    model_type += '_MODEL'
    logger.info(f'Save model as Pickle file... {model_type}')
    pickle.dump(
        {
            'model': model,
            'model_type': model_type,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'spearman_rho': spearman_r,
            'regressor': regressor
        },
        open(f'Pickles/{model_type}', 'wb')
    )


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
    Decides based on the params file input type which DCA encoding to be performed, i.e.,
    GREMLIN or PLMC.
    If use_global_model==True, to avoid each time pickle model file getting loaded, which
    is quite inefficient when performing directed evolution, i.e., encoding of single
    sequences, a global model is stored at the first evolution step and used in the
    subsequent steps.
    """
    global global_model, global_model_type
    if ys_true is None:
        ys_true = np.zeros(np.shape(sequences))
    if use_global_model:
        if global_model is None:
            global_model, global_model_type = get_model_and_type(params_file, substitution_sep)
            model, model_type = global_model, global_model_type
        else:
            model, model_type = global_model, global_model_type
    else:
        model, model_type = get_model_and_type(params_file, substitution_sep)
    if model_type == 'PLMC':
        xs, x_wt, variants, sequences, ys_true = plmc_encoding(
            model, variants, sequences, ys_true, threads, verbose
        )
    elif model_type == 'GREMLIN':
        if verbose:
            logger.info(f"Following positions are frequent gap positions in the MSA "
                        f"and cannot be considered for effective modeling, i.e., "
                        f"substitutions at these positions are removed as these would be "
                        f"predicted as wild type:\n{[gap + 1 for gap in model.gaps]}.\n"
                        f"Effective positions (N={len(model.v_idx)}) are:\n"
                        f"{[v_pos + 1 for v_pos in model.v_idx]}")
        xs, x_wt, variants, sequences, ys_true = gremlin_encoding(
            model, variants, sequences, ys_true,
            shift_pos=1, substitution_sep=substitution_sep
        )
    else:
        raise SystemError(
            f"Found a {model_type.lower()} model as input. Please train a new "
            f"hybrid model on the provided LS/TS datasets."
        )
    assert len(xs) == len(variants) == len(sequences) == len(ys_true)
    return xs, variants, sequences, ys_true, x_wt, model, model_type


def gremlin_encoding(gremlin: GREMLIN, variants, sequences, ys_true, shift_pos=1, substitution_sep='/'):
    """
     Gets X and x_wt for DCA prediction: delta_Hamiltonian respectively
     delta_E = np.subtract(X, x_wt), with X = encoded sequences of variants.
     Also removes variants, sequences, and y_trues at MSA gap positions.
    """
    variants, sequences, ys_true = np.atleast_1d(variants), np.atleast_1d(sequences), np.atleast_1d(ys_true)
    variants, sequences, ys_true = remove_gap_pos(
        gremlin.gaps, variants, sequences, ys_true,
        shift_pos=shift_pos, substitution_sep=substitution_sep
    )
    try:
        xs = gremlin.get_score(sequences, encode=True)
    except SystemError:
        xs = []
    x_wt = gremlin.get_score(np.atleast_1d(gremlin.wt_seq), encode=True)
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
    if threads > 1:
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
        Variants with substitutions at valid sequence positions, i.e., at non-gap positions
    sequences_v
        Sequences of variants with substitutions at valid sequence positions, i.e., at non-gap positions
    fitnesses_v
        Fitness values of variants with substitutions at valid sequence positions, i.e., at non-gap positions
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


def get_delta_e_statistical_model(
        x_test: np.ndarray,
        x_wt: np.ndarray
):
    """
    Description
    -----------
    Delta_E means difference in evolutionary energy in plmc terms.
    In other words, this is the delta of the sum of Hamiltonian-encoded
    sequences of local fields and couplings of encoded sequence and wild-type
    sequence in GREMLIN terms.

    Parameters
    -----------
    x_test: np.ndarray [2-dim]
        Encoded sequences to be subtracted by x_wt to compute delta E.
    x_wt: np.ndarray [1-dim]
        Encoded wild-type sequence.

    Returns
    -----------
    delta_e: np.ndarray [1-dim]
        Summed subtracted encoded sequences.

    """
    delta_x = np.subtract(x_test, x_wt)
    delta_e = np.sum(delta_x, axis=1)
    return delta_e


def generate_model_and_save_pkl(
        variants,
        ys_true,
        params_file,
        wt,
        train_percent_fit: float = 0.66,  # percent of all data: 0.8 * 0.66
        test_percent: float = 0.2,
        random_state: int = 42,
        substitution_sep = '/',
        threads=1
):
    """
    Description
    -----------
    Save (Ridge) regression model (trained and with tuned alpha parameter)
    with betas (beta_1 and beta_2) as dictionary-structured pickle file.

    Parameters
    ----------
    test_percent: float
        Percent of DataFrame data used for testing. The remaining data is
        used for training (fitting and validation).
    train_percent_fit: float
        Percent of DataFrame data to train on.
        The remaining data is used for validation.
    random_state: int
        Random seed for splitting in train and test data for reproducing results.

    Returns
    ----------
    ()
        Just saving model parameters as pickle file.
    """
    wt_seq = get_wt_sequence(wt)
    variants_splitted = split_variants(variants, substitution_sep)
    variants, ys_true, sequences = get_seqs_from_var_name(wt_seq, variants_splitted, ys_true)

    xs, variants, sequences, ys_true, x_wt, model, model_type = plmc_or_gremlin_encoding(
        variants, sequences, ys_true, params_file, substitution_sep, threads)

    logger.info(
        f'Train size (fitting): {train_percent_fit * 100:.1f} % of training data '
        f'({((1 - test_percent) * train_percent_fit) * 100:.1f} % of all data)\n'
        f'Train size validation: {(1 - train_percent_fit) * 100:.1f} % of training data '
        f'({((1 - test_percent) * (1 - train_percent_fit)) * 100:.1f} % of all data)\n'
        f'Test size: {test_percent * 100:.1f} % ({test_percent * 100:.1f} % of all data)\n'
        f'Using random state: {random_state}...\n'
    )

    x_train, x_test, y_train, y_test = train_test_split(
        xs, ys_true, test_size=test_percent, random_state=random_state
    )

    hybrid_model = DCAHybridModel(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_wt=x_wt
    )

    beta_1, beta_2, reg, spearman_dca, test_spearman_r = hybrid_model.train_and_test(
        train_percent_fit=train_percent_fit,
        random_state=random_state
    )
    if reg is None:
        alpha_ = 'None'
    else:
        alpha_ = f'{reg.alpha:.3f}'
    logger.info(
        f'Individual model weights and regressor hyperparameters:\n'
        f'Hybrid model individual model contributions:\nBeta1 (DCA): '
        f'{beta_1:.3f}, Beta2 (ML): {beta_2:.3f} ('
        f'regressor: Ridge(alpha={alpha_}))\n'
        f'Test performance: Spearman\'s rho = {test_spearman_r:.3f}'
    )
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass
    model_name = f'HYBRID{model_type.lower()}'
    save_model_to_dict_pickle(hybrid_model, model_name, beta_1, beta_2, test_spearman_r, reg)


def performance_ls_ts(
        ls_fasta: str | None,
        ts_fasta: str | None,
        threads: int,
        params_file: str,
        model_pickle_file: str | None = None,
        substitution_sep: str = '/',
        label=False
):
    """
    Description
    -----------
    Computes performance based on a (linear) regression model trained
    on the training set by optimizing model hyperparameters based on
    validation performances on training subsets (default: 5-fold CV)
    and predicting test set entries using the hyperparmeter-tuned model
    to estimate performance for model generalization.

    Parameters
    -----------
    ls_fasta: str
        Fasta-like file with fitness values. Will be read and extracted
        for training the regressor.
    ts_fasta: str
        Fasta-like file with fitness values. Used for computing performance
        of the tuned regressor for test set entries (performance metric of
        measured and predicted fitness values).
    threads: int
        Number of threads to use for parallel computing using Ray.
    params_file: str
        PLMC parameter file (containing evolutionary, i.e. MSA-based local
        and coupling terms.
    model: str
        Model to load for TS prediction.
    separator: str
        Character to split the variant to obtain the single substitutions
        (default='/').

    Returns
    -----------
    None
        Just plots test results (predicted fitness vs. measured fitness)
        using def plot_y_true_vs_y_pred.
    """
    test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)

    if ls_fasta is not None and ts_fasta is not None:
        train_sequences, train_variants, y_train = get_sequences_from_file(ls_fasta)
        x_train, train_variants, train_sequences, y_train, x_wt, _, model_type = plmc_or_gremlin_encoding(
            train_variants, train_sequences, y_train, params_file, substitution_sep, threads
        )

        x_test, test_variants, test_sequences, y_test, *_ = plmc_or_gremlin_encoding(
            test_variants, test_sequences, y_test, params_file, substitution_sep, threads, verbose=False
        )

        logger.info(f"\nInitial training set variants: {len(train_sequences)}. "
                    f"Remaining: {len(train_variants)} (after removing "
                    f"substitutions at gap positions).\nInitial test set "
                    f"variants: {len(test_sequences)}. Remaining: {len(test_variants)} "
                    f"(after removing substitutions at gap positions)."
                    )

        hybrid_model = DCAHybridModel(
            x_train=np.array(x_train),
            y_train=np.array(y_train),
            x_test=np.array(x_test),
            y_test=np.array(y_test),
            x_wt=x_wt
        )
        model_name = f'HYBRID{model_type.lower()}'

        spearman_r, reg, beta_1, beta_2 = hybrid_model.ls_ts_performance()
        ys_pred = hybrid_model.hybrid_prediction(np.array(x_test), reg, beta_1, beta_2)

        if reg is None:
            alpha_ = 'None'
        else:
            alpha_ = f'{reg.alpha:.3f}'
        logger.info(
            f'Individual model weights and regressor hyperparameters:\n'
            f'Hybrid model individual model contributions: Beta1 (DCA): '
            f'{beta_1:.3f}, Beta2 (ML): {beta_2:.3f} (regressor: '
            f'Ridge(alpha={alpha_}))\nTesting performance...'
        )

        save_model_to_dict_pickle(hybrid_model, model_name, beta_1, beta_2, spearman_r, reg)

    elif ts_fasta is not None and model_pickle_file is not None and params_file is not None:
        logger.info(f'Taking model from saved model (Pickle file): {model_pickle_file}...')

        model, model_type = get_model_and_type(model_pickle_file)

        if model_type != 'Hybrid':  # same as below in next elif
            x_test, test_variants, test_sequences, y_test, x_wt, *_ = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, model_pickle_file, substitution_sep, threads, False)
            ys_pred = get_delta_e_statistical_model(x_test, x_wt)
        else:  # Hybrid model input requires params from plmc or GREMLIN model
            beta_1, beta_2, reg = model.beta_1, model.beta_2, model.regressor
            x_test, test_variants, test_sequences, y_test, *_ = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, params_file,
                substitution_sep, threads, False
            )
            ys_pred = model.hybrid_prediction(x_test, reg, beta_1, beta_2)

    elif ts_fasta is not None and model_pickle_file is None:  # no LS provided --> statistical modeling / no ML
        logger.info(f'No learning set provided, falling back to statistical DCA model: '
                    f'no adjustments of individual hybrid model parameters (beta_1 and beta_2).')
        test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)
        x_test, test_variants, test_sequences, y_test, x_wt, model, model_type = plmc_or_gremlin_encoding(
            test_variants, test_sequences, y_test, params_file, substitution_sep, threads
        )

        logger.info(f"Initial test set variants: {len(test_sequences)}. "
                    f"Remaining: {len(test_variants)} (after removing "
                    f"substitutions at gap positions).")

        ys_pred = get_delta_e_statistical_model(x_test, x_wt)

        save_model_to_dict_pickle(model, model_type, None, None, spearmanr(y_test, ys_pred)[0], None)

    else:
        raise SystemError('No Test Set given for performance estimation.')

    spearman_rho = spearmanr(y_test, ys_pred)
    logger.info(f'Spearman Rho = {spearman_rho[0]:.3f}')

    plot_y_true_vs_y_pred(
        np.array(y_test), np.array(ys_pred), np.array(test_variants), label=label, hybrid=True
    )


def predict_ps(  # also predicting "pmult" dict directories
        prediction_dict: dict,
        threads: int,
        separator: str,
        model_pickle_file: str,
        params_file: str = None,
        prediction_set: str = None,
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
        PLMC couplings parameter file
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
    logger.info(f'Taking model from saved model (Pickle file): {model_pickle_file}...')

    model, model_type = get_model_and_type(model_pickle_file)

    if model_type == 'PLMC':
        logger.info(f'No hybrid model provided – falling back to a statistical DCA model.')
    elif model_type == 'Hybrid':
        beta_1, beta_2, reg = model.beta_1, model.beta_2, model.regressor
        if reg is None:
            alpha_ = 'None'
        else:
            alpha_ = f'{reg.alpha:.3f}'
        logger.info(
            f'Individual model weights and regressor hyperparameters:\n'
            f'Hybrid model individual model contributions: Beta1 (DCA): {beta_1:.3f}, '
            f'Beta2 (ML): {beta_2:.3f} (regressor: Ridge(alpha={alpha_})).'
        )

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
                    logger.info(f'Encoding files ({i + 1}/{len(files)}) for prediction...\n')
                    file_path = os.path.join(path, file)
                    sequences, variants, _ = get_sequences_from_file(file_path)
                    if model_type != 'Hybrid':
                        x_test, test_variants, x_wt, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, model, threads=threads, verbose=False,
                            substitution_sep=separator)
                        ys_pred = get_delta_e_statistical_model(x_test, x_wt)
                    else:  # Hybrid model input requires params from plmc or GREMLIN model
                        ##encoding_model, encoding_model_type = get_model_and_type(params_file)
                        x_test, test_variants, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, params_file,
                            threads=threads, verbose=False, substitution_sep=separator
                        )
                        ys_pred = model.hybrid_prediction(x_test, reg, beta_1, beta_2)
                    for k, y in enumerate(ys_pred):
                        all_y_v_pred.append((ys_pred[k], variants[k]))
                if negative:  # sort by fitness value
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=False)
                else:
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=True)
                predictions_out(
                    predictions=all_y_v_pred,
                    model='Hybrid',
                    prediction_set=f'Top{path}',
                    path=path
                )
            else:  # check next task to do, e.g., predicting triple substituted variants, e.g. trecomb
                continue

    elif prediction_set is not None:
        sequences, variants, _ = get_sequences_from_file(prediction_set)
        # NaNs are already being removed by the called function
        if model_type != 'Hybrid':  # statistical DCA model
            xs, variants, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                variants, sequences, None, params_file,
                threads=threads, verbose=False, substitution_sep=separator)
            ys_pred = get_delta_e_statistical_model(xs, x_wt)
        else:  # Hybrid model input requires params from plmc or GREMLIN model
            xs, variants, *_ = plmc_or_gremlin_encoding(
                variants, sequences, None, params_file,
                threads=threads, verbose=True, substitution_sep=separator
            )
            ys_pred = model.hybrid_prediction(xs, reg, beta_1, beta_2)
        assert len(xs) == len(variants)
        y_v_pred = zip(ys_pred, variants)
        y_v_pred = sorted(y_v_pred, key=lambda x: x[0], reverse=True)
        predictions_out(
            predictions=y_v_pred,
            model='Hybrid',
            prediction_set=f'Top{prediction_set}'
        )


def predict_directed_evolution(
        encoder: str,
        variant: str,
        sequence: str,
        hybrid_model_data_pkl: str
) -> Union[str, list]:
    """
    Perform directed in silico evolution and predict the fitness of a
    (randomly) selected variant using the hybrid model. This function opens
    the stored DCAHybridModel and the model parameters to predict the fitness
    of the variant encoded herein using the PLMC class. If the variant
    cannot be encoded (based on the PLMC params file), returns 'skip'. Else,
    returning the predicted fitness value and the variant name.
    """
    if hybrid_model_data_pkl is not None:
        model, model_type = get_model_and_type(hybrid_model_data_pkl)
    else:
        model_type = 'StatisticalModel'  # any name != 'Hybrid'

    if model_type != 'Hybrid':  # statistical DCA model
        xs, variant, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
            variant, sequence, None, encoder, verbose=False, use_global_model=True)
        if not list(xs):
            return 'skip'
        y_pred = get_delta_e_statistical_model(xs, x_wt)
    else:  # model_type == 'Hybrid': Hybrid model input requires params from PLMC or GREMLIN model
        xs, variant, *_ = plmc_or_gremlin_encoding(
            variant, sequence, None, encoder, verbose=False, use_global_model=True
        )
        if not list(xs):
            return 'skip'
        try:
            y_pred = model.hybrid_prediction(np.atleast_2d(xs), model.regressor, model.beta_1, model.beta_2)[0]
        except ValueError:
            raise SystemError(
                "Probably a different model was used for encoding than for modeling; "
                "e.g. using a HYBRIDgremlin model in combination with parameters taken from a PLMC file."
            )
    y_pred = float(y_pred)

    return [(y_pred, variant[0][1:])]
