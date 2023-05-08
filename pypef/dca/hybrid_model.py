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

import pypef.dca.encoding
from pypef.utils.variant_data import get_sequences_from_file, remove_nan_encoded_positions
from pypef.dca.encoding import DCAEncoding, get_dca_data_parallel, get_encoded_sequence, EffectiveSiteError
from pypef.ml.regression import predictions_out, plot_y_true_vs_y_pred

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) # DEV


class DCAHybridModel:
    alphas = np.logspace(-6, 6, 100)  # Grid for the parameter 'alpha'.
    parameter_range = [(0, 1), (0, 1)]  # Parameter range of 'beta_1' and 'beta_2' with lb <= x <= ub

    def __init__(
            self,
            alphas=alphas,
            parameter_range=None,
            X_train: np.ndarray = None,
            y_train: np.ndarray = None,
            X_test: np.ndarray = None,  # not necessary for training
            y_test: np.ndarray = None,  # not necessary for training
            X_wt = None
    ):
        if parameter_range is None:
            parameter_range = parameter_range
        self._alphas = alphas
        self._parameter_range = parameter_range
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = np.concatenate((X_train, X_test), axis=0) if self.X_test is not None else self.X_train
        self.y = np.concatenate((y_train, y_test), axis=0) if self.y_test is not None else self.y_train
        self.x_wild_type = X_wt
        self._spearmanr_dca = self._spearmanr_dca()

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

    def _delta_X(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Substracts for each variant the encoded wild-type sequence
        from its encoded sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Array of encoded variant sequences.

        Returns
        -------
        Array of encoded variant sequences with substracted encoded
        wild-type sequence.
        """
        return np.subtract(X, self.x_wild_type)

    def _delta_E(
            self,
            X: np.ndarray
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
        return np.sum(self._delta_X(X), axis=1)

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
        y_dca = self._delta_E(self.X)
        return self.spearmanr(self.y, y_dca)

    def ridge_predictor(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
    ) -> object:
        """
        Sets the parameter 'alpha' for ridge regression.

        Parameters
        ----------
        X_train : np.ndarray
            Array of the encoded sequences for training.
        y_train : np.ndarray
            Associated fitness values to the sequences present
            in 'X_train'.

        Returns
        -------
        Ridge object trained on 'X_train' and 'y_train' (cv=5)
        with optimized 'alpha'. 
        """
        grid = GridSearchCV(Ridge(), {'alpha': self._alphas}, cv=5)
        grid.fit(X_train, y_train)
        return Ridge(**grid.best_params_).fit(X_train, y_train)

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
            X_train: np.ndarray,
            y_train: np.ndarray,
            train_size_fit=0.66,
            random_state=42
    ) -> tuple:
        """
        Get the adjusted parameters 'beta_1', 'beta_2', and the
        tuned regressor of the hybrid model.

        Parameters
        ----------
        X_train : np.ndarray
            Encoded sequences of the variants in the training set.
        y_train : np.ndarray
            Fitness values of the variants in the training set.
        X_test : np.ndarray
            Encoded sequences of the variants in the testing set.
        y_test : np.ndarray
            Fitness values of the variants in the testing set.
        train_size_train : float [0,1] (default 0.66)
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
                X_train, y_train,
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

        y_dca_ttest = self._delta_E(X_ttest)

        ridge = self.ridge_predictor(X_ttrain, y_ttrain)
        y_ridge_ttest = ridge.predict(X_ttest)

        beta1, beta2 = self._adjust_betas(y_ttest, y_dca_ttest, y_ridge_ttest)
        return beta1, beta2, ridge

    def hybrid_prediction(
            self,
            X: np.ndarray,
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
        X : np.ndarray
            Encoded sequences used for prediction.
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
        y_dca = self._delta_E(X)
        if reg is None:
            y_ridge = np.random.random(len(y_dca))  # in order to suppress error
        else:
            y_ridge = reg.predict(X)
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
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=train_size, random_state=random_state)
            beta_1, beta_2, reg = self.settings(X_train, y_train)
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
                                X_test, reg, beta_1, beta_2
                            )
                        ),
                        'beta_1': beta_1,
                        'beta_2': beta_2,
                        'alpha': alpha
                    }
                }
            )

        return data

    def ls_ts_performance(
            self,
            data=None
    ):
        if data is None:
            data = {}
        beta_1, beta_2, reg = self.settings(
            X_train=self.X_train,
            y_train=self.y_train
        )
        if reg is None:
            alpha_ = 'None'
        else:
            alpha_ = f'{reg.alpha:.3f}'
        logger.info(f'Beta 1 (DCA): {beta_1:.3f}, Beta 2 (ML): {beta_2:.3f} ( '
              f'regressor: Ridge(alpha={alpha_}))')
        if beta_2 == 0.0:
            alpha = np.nan
        else:
            alpha = reg.alpha
        data.update(
            {f'ls_ts':
                {
                    'n_y_train': len(self.y_train),
                    'n_y_test': len(self.y_test),
                    'spearman_rho': self.spearmanr(
                        self.y_test, self.hybrid_prediction(
                            self.X_test, reg, beta_1, beta_2
                        )
                    ),
                    'beta_1': beta_1,
                    'beta_2': beta_2,
                    'regressor': reg,
                    'alpha': alpha
                }
            }
        )

        return data, self.hybrid_prediction(self.X_test, reg, beta_1, beta_2)

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
            X_train=self.X_train,
            y_train=self.y_train,
            train_size_fit=train_percent_fit,
            random_state=random_state
        )

        if len(self.y_test) > 0:
            test_spearman_r = self.spearmanr(
                self.y_test,
                self.hybrid_prediction(
                    self.X_test, reg, beta_1, beta_2
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


def generate_model_and_save_pkl(
        xs: list,
        ys: list,
        dca_encoder: DCAEncoding,
        train_percent_fit: float = 0.66,  # percent of all data: 0.8 * 0.66
        test_percent: float = 0.2,
        random_state: int = 42
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
    None
        Just saving model parameters as pickle file.
    """

    # getting target (WT) sequence and encoding it to provide it as
    # relative value for pure DCA based predictions (difference in sums
    # of sequence encodings: variant - WT)
    target_seq, index = dca_encoder.get_target_seq_and_index()
    wt_name = target_seq[0] + str(index[0]) + target_seq[0]
    x_wt = get_encoded_sequence(wt_name, dca_encoder)

    logger.info(
        f'Train size (fitting): {train_percent_fit*100:.1f} % of training data '
        f'({((1 - test_percent)*train_percent_fit)*100:.1f} % of all data),\n'
        f'Train size validation: {(1 - train_percent_fit)*100:.1f} % of training data '
        f'({((1 - test_percent)*(1 - train_percent_fit))*100:.1f} % of all data),\n'
        f'Test size: {test_percent*100:.1f} % ({test_percent*100:.1f} % of all data),\n'
        f'(Random state: {random_state})...\n'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        xs, ys, test_size=test_percent, random_state=random_state
    )

    hybrid_model = DCAHybridModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_wt=x_wt
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
    logger.info(f'Save model as Pickle file... HYBRIDMODEL')
    pickle.dump(
        {
            'model': hybrid_model,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'spearman_rho': test_spearman_r,
            'regressor': reg
        },
        open('Pickles/HYBRIDMODEL', 'wb')
    )


def check_model_type(model):
    """
    Checks type/instance of loaded Pickle file.
    """
    if type(model) == pypef.dca.encoding.DCAEncoding:
        return 'DCAMODEL'
    elif type(model) == pypef.dca.hybrid_model.DCAHybridModel:
        return 'HYBRIDMODEL'
    elif isinstance(model, sklearn.base.BaseEstimator):
        raise SystemError("Loaded an sklearn ML model. For pure ML-based modeling the "
                          "\'ml\' flag has to be used instead of the \'hybrid\' flag.")
    else:
        raise SystemError('Unknown model/unknown Pickle file.')


def get_delta_e_statistical_model(
        x_test: np.ndarray,
        x_wt: np.ndarray
):
    """
    Description
    -----------


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



def performance_ls_ts(
        ls_fasta: str | None,
        ts_fasta: str | None,
        threads: int,
        params_file: str,
        separator: str
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
    separator: str
        Character to split the variant to obtain the single substitutions
        (default='/').


    Returns
    -----------
    None
        Just plots test results (predicted fitness vs. measured fitness)
        using def plot_y_true_vs_y_pred.
    """
    dca_encoder = DCAEncoding(
        params_file=params_file,
        separator=separator,
        verbose=False
    )
    # DCA prediction: delta E = np.subtract(X, self.x_wild_type),
    # with X = encoded sequence of any variant -->
    # getting wild-type name und subsequently x_wild_type
    # to provide it for the DCAHybridModel
    target_seq, index = dca_encoder.get_target_seq_and_index()
    wt_name = target_seq[0] + str(index[0]) + target_seq[0]
    logger.info(f'Using to-self-substitution \'{wt_name}\' as wild type reference.')
    x_wt = get_encoded_sequence(wt_name, dca_encoder)
    if ls_fasta is not None and ts_fasta is not None:
        _, train_variants, y_train = get_sequences_from_file(ls_fasta)
        _, test_variants, y_test = get_sequences_from_file(ts_fasta)

        if threads > 1:
            # Hyperthreading, NaNs are already being removed by the called function
            train_variants, x_train, y_train = get_dca_data_parallel(
                train_variants, y_train, dca_encoder, threads)
            test_variants, x_test, y_test = get_dca_data_parallel(
                test_variants, y_test, dca_encoder, threads)
        else:
            x_train_ = dca_encoder.collect_encoded_sequences(train_variants)
            x_test_ = dca_encoder.collect_encoded_sequences(test_variants)
            # NaNs must still be removed
            x_train, train_variants, y_train = remove_nan_encoded_positions(x_train_, train_variants, y_train)
            x_test, test_variants, y_test = remove_nan_encoded_positions(x_test_, test_variants, y_test)
        assert len(x_train) == len(train_variants) == len(y_train)
        assert len(x_test) == len(test_variants) == len(y_test)

        hybrid_model = DCAHybridModel(
            X_train=np.array(x_train),
            y_train=np.array(y_train),
            X_test=np.array(x_test),
            y_test=np.array(y_test),
            X_wt=x_wt
        )
        data, y_pred = hybrid_model.ls_ts_performance()
        result = data['ls_ts']
        test_spearman_r = result['spearman_rho']
        beta_1 = result['beta_1']
        beta_2 = result['beta_2']
        reg = result['regressor']
        if reg is None:
            alpha_ = 'None'
        else:
            alpha_ = f'{reg.alpha:.3f}'
        logger.info(
            f'Individual model weights and regressor hyperparameters:\n'
            f'Hybrid model individual model contributions: Beta1 (DCA): '
            f'{beta_1:.3f}, Beta2 (ML): {beta_2:.3f} (regressor: '
            f'Ridge(alpha={alpha_}))\nTesting performance...\nSpearman\'s '
            f'rho = {test_spearman_r:.3f}'
        )
        try:
            os.mkdir('Pickles')
        except FileExistsError:
            pass
        logger.info(f'Save model as Pickle file... HYBRIDMODEL')
        pickle.dump(
            {
                'model': hybrid_model,
                'beta_1': beta_1,
                'beta_2': beta_2,
                'spearman_rho': test_spearman_r,
                'regressor': reg
            },
            open('Pickles/HYBRIDMODEL', 'wb')
        )

    elif ts_fasta is not None:
        logger.info('No learning set provided, falling back to statistical DCA model: '
                    'no adjustments of individual hybrid model parameters (beta_1 and beta_2).')
        _, test_variants, y_test = get_sequences_from_file(ts_fasta)
        if threads > 1:
            test_variants, x_test, y_test = get_dca_data_parallel(
                test_variants, y_test, dca_encoder, threads)
        else:
            x_test_ = dca_encoder.collect_encoded_sequences(test_variants)
            x_test, y_test, test_variants = remove_nan_encoded_positions(x_test_, y_test, test_variants)

        delta_e = get_delta_e_statistical_model(x_test, x_wt)

        spearman_rho = spearmanr(y_test, delta_e)
        logger.info(f'Spearman Rho = {spearman_rho[0]:.3f}')

        logger.info(f'Save model as Pickle file... DCAMODEL')
        pickle.dump(
            {
                'model': dca_encoder,
                'beta_1': None,
                'beta_2': None,
                'spearman_rho': spearman_rho,
                'regressor': None
            },
            open('Pickles/DCAMODEL', 'wb')
        )

    else:
        logger.info('No Test Set given for performance estimation.')


def predict_ps(  # also predicting "pmult" dirs
        prediction_dict: dict,
        params_file: str,
        threads: int,
        separator: str,
        model_pickle_file: str,
        test_set: str = None,
        prediction_set: str = None,
        figure: str = None,
        label: bool = False,
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
    None
        Writes sorted predictions to files (for [--drecomb] [--trecomb]
        [--qarecomb] [--qirecomb] [--ddiverse] [--tdiverse] [--qdiverse]
        in the respective created folders).

    """
    if threads > 1:  # silent DCA encoding
        dca_encoder = DCAEncoding(params_file, separator=separator, verbose=False)
    else:
        dca_encoder = DCAEncoding(params_file, separator=separator)
    logger.info(f'Taking regression model from saved model (Pickle file): {model_pickle_file}...')
    model_data = pickle.load(open(f'Pickles/{model_pickle_file}', "rb"))
    model = model_data['model']
    test_spearman_r = model_data['spearman_rho']
    beta_1 = model_data['beta_1']
    beta_2 = model_data['beta_2']
    reg = model_data['regressor']

    if check_model_type(model) == 'DCAMODEL':
        pass
    elif check_model_type(model) == 'HYBRIDMODEL':
        if reg is None:
            alpha_ = 'None'
        else:
            alpha_ = f'{reg.alpha:.3f}'
        logger.info(
            f'Individual model weights and regressor hyperparameters:\n'
            f'Hybrid model individual model contributions: Beta1 (DCA): {beta_1:.3f}, '
            f'Beta2 (ML): {beta_2:.3f} (regressor: Ridge(alpha={alpha_})), '
            f'Train->Test performance: Spearman\'s rho = {test_spearman_r:.3f}.'
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
                    logger.info(f'Encoding files ({i+1}/{len(files)}) for prediction...\n')
                    file_path = os.path.join(path, file)
                    sequences, variants, _ = get_sequences_from_file(file_path)
                    if threads > 1:  # parallel execution
                        # NaNs are already being removed by the called function
                        variants, xs, _ = get_dca_data_parallel(
                            variants, list(np.zeros(len(variants))), dca_encoder, threads)
                    else:  # single thread execution
                        xs = dca_encoder.collect_encoded_sequences(variants)
                        # NaNs must still be removed
                        xs, variants = remove_nan_encoded_positions(xs, variants)
                    assert len(xs) == len(variants)
                    if check_model_type(model) == 'DCAMODEL':
                        target_seq, index = dca_encoder.get_target_seq_and_index()
                        wt_name = target_seq[0] + str(index[0]) + target_seq[0]
                        x_wt = get_encoded_sequence(wt_name, dca_encoder)
                        ys_pred = get_delta_e_statistical_model(xs, x_wt)
                    elif check_model_type(model) == 'HYBRIDMODEL':
                        ys_pred = model.hybrid_prediction(xs, reg, beta_1, beta_2)
                    for k, y in enumerate(ys_pred):
                        all_y_v_pred.append((ys_pred[k], variants[k]))
                if negative:  # sort by fitness value
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=False)
                else:
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=True)
                predictions_out(
                    predictions=all_y_v_pred,
                    model='hybrid',
                    prediction_set=prediction_set,
                    path=path
                )
            else:  # check next task to do, e.g. predicting triple substituted variants, e.g. trecomb
                continue

    elif prediction_set is not None:
        sequences, variants, _ = get_sequences_from_file(prediction_set)
        # NaNs are already being removed by the called function
        if threads > 1:
            # NaNs are already being removed by the called function
            variants, xs, _ = get_dca_data_parallel(
                variants, list(np.zeros(len(variants))), dca_encoder, threads)
        else:
            xs = dca_encoder.collect_encoded_sequences(variants)
            # NaNs must still be removed
            xs, variants = remove_nan_encoded_positions(xs, variants)
        assert len(xs) == len(variants)
        if prediction_set is not None:
            if check_model_type(model) == 'DCAMODEL':
                target_seq, index = dca_encoder.get_target_seq_and_index()
                wt_name = target_seq[0] + str(index[0]) + target_seq[0]
                x_wt = get_encoded_sequence(wt_name, dca_encoder)
                y_pred = get_delta_e_statistical_model(xs, x_wt)
            elif check_model_type(model) == 'HYBRIDMODEL':
                y_pred = model.hybrid_prediction(xs, reg, beta_1, beta_2)
            y_v_pred = zip(y_pred, variants)
            y_v_pred = sorted(y_v_pred, key=lambda x: x[0], reverse=True)
            predictions_out(
                predictions=y_v_pred,
                model='hybrid',
                prediction_set=prediction_set
            )
    elif test_set is not None or figure is not None:
        if test_set is not None:
            loaded_set = test_set
        else:
            loaded_set = figure
        sequences, variants, y_true = get_sequences_from_file(loaded_set)
        # NaNs are already being removed by the called function
        try:
            if threads > 1:
                variants, xs, y_test = get_dca_data_parallel(
                    variants, y_true, dca_encoder, threads)
            else:
                # NaNs must still be removed
                xs_ = dca_encoder.collect_encoded_sequences(variants)
                xs, variants, y_test = remove_nan_encoded_positions(xs_, variants, y_true)
            assert len(xs) == len(variants) == len(y_test)
        except IndexError:
            raise SystemError(
                "Potentially, you provided a prediction set for plotting the figure "
                "instead of a test set (including measured fitness values, i.e. y_true)."
            )
        if check_model_type(model) == 'DCAMODEL':
            target_seq, index = dca_encoder.get_target_seq_and_index()
            wt_name = target_seq[0] + str(index[0]) + target_seq[0]
            x_wt = get_encoded_sequence(wt_name, dca_encoder)
            y_pred = get_delta_e_statistical_model(xs, x_wt)
        elif check_model_type(model) == 'HYBRIDMODEL':
            y_pred = model.hybrid_prediction(xs, reg, beta_1, beta_2)
        logger.info('Testing performance...\n'
                    f'Spearman\'s rho = {spearmanr(y_test, y_pred)[0]:.3f} (N_test = {len(y_pred)})')
        if figure is not None:
            plot_y_true_vs_y_pred(
                np.array(y_test), np.array(y_pred), np.array(variants), label, hybrid=True
            )
    else:
        raise SystemError(
            'Define set(s) for prediction (e.g. \'-p PS.fasta\' or '
            'created prediction set folder, e.g. \'--pmult --drecomb\')'
        )


def predict_directed_evolution(
        encoder: DCAEncoding,
        variant: str,
        hybrid_model_data_pkl: str
) -> Union[str, list]:
    """
    Perform directed in silico evolution and predict the fitness of a
    (randomly) selected variant using the hybrid model. This function opens
    the stored DCAHybridModel and the model parameters to predict the fitness
    of the variant encoded herein using the DCAEncoding class. If the variant
    cannot be encoded (based on the PLMC params file), returns 'skip'. Else,
    returning the predicted fitness value and the variant name.
    """
    try:
        x = encoder.encode_variant(variant)
    except EffectiveSiteError:
        return 'skip'

    model_dict = pickle.load(open(os.path.join('Pickles', hybrid_model_data_pkl), "rb"))
    model = model_dict['model']
    reg = model_dict['regressor']
    beta_1 = model_dict['beta_1']
    beta_2 = model_dict['beta_2']
    if check_model_type(model) == 'DCAMODEL':
        target_seq, index = encoder.get_target_seq_and_index()
        wt_name = target_seq[0] + str(index[0]) + target_seq[0]
        x_wt = get_encoded_sequence(wt_name, encoder)
        y_pred = get_delta_e_statistical_model(np.atleast_2d(x), x_wt)[0]  # [0] only for unpacking from list
    elif check_model_type(model) == 'HYBRIDMODEL':
        y_pred = model.hybrid_prediction(  # 2d as only single variant
            X=np.atleast_2d(x),  # e.g., np.atleast_2d(3.0) --> array([[3.]])
            reg=reg,  # RidgeRegressor
            beta_1=beta_1,  # DCA model prediction weight
            beta_2=beta_2  # ML model prediction weight
        )[0]  # [0] only for unpacking from list

    return [(y_pred, variant[1:])]
