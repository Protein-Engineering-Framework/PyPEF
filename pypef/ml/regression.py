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


"""
Main modules for regression/ML including feature generation
(i.e. sequence encoding), cross-validation-based hyperparameter
tuning, prediction, and plotting routines.
"""


import os
from typing import Union

import logging
logger = logging.getLogger('pypef.ml.regression')
import matplotlib
matplotlib.use('Agg')  # no plt.show(), just save plot
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
from tqdm import tqdm  # progress bars
from adjustText import adjust_text
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  # default: refit=True

# import regression models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from pypef.utils.variant_data import (
    amino_acids, get_sequences_from_file,
    remove_nan_encoded_positions, get_basename
)
from pypef.dca.encoding import DCAEncoding, get_dca_data_parallel

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='numpy')
# ignoring warnings of PLS regression when using n_components
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
# FutureWarning: The default of 'normalize' will be set to False in version 1.2 and deprecated in version 1.4.
# If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage.
# sklearn: The default value of 'normalize' should be changed to False in linear models where now normalize=True
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')


def read_models(number):
    """
    reads the models found in the file Model_Results.txt.
    If no model was trained, the .txt file does not exist.
    """
    try:
        ls = ""
        with open('Model_Results.txt', 'r') as file:
            for i, lines in enumerate(file):
                if i == 0:
                    if lines[:6] == 'No FFT':
                        number += 2
                if i <= number + 1:
                    ls += lines
        return ls
    except FileNotFoundError:
        return "No Model_Results.txt found."


def full_aaidx_txt_path(filename):
    """
    returns the path of an index inside the folder /AAindex/,
    e.g. /path/to/pypef/ml/AAindex/FAUJ880104.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(modules_path, 'AAindex', filename)


def path_aaindex_dir():
    """
    returns the absolute path to the /AAindex folder,
    e.g. /path/to/AAindex/.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AAindex')


class AAIndex:
    """
    gets all the information that are given in each AAindex file.
    For the program routine it provides the library to enable translation
    of the alphabetical amino acid sequence to an array of numericals.
    """
    def __init__(self, filename):
        self.file = filename
        self.accession_number = None
        self.data_description = None
        self.pmid = None
        self.authors = None
        self.title_of_article = None
        self.journal_reference = None

    def general_information(self):
        """
        Gets and allocates general information based on the AAindex file
        format defined by file sections 'H', 'D', 'E', 'A', 'T', 'J'
        """
        with open(self.file, 'r') as f:
            for line in f:
                # try/ except "removes" empty lines.
                try:
                    words = line.split()
                    id_letter = words[0]
                except IndexError:
                    break

                # Extract some general information about AAindex file.
                if id_letter == 'H':
                    self.accession_number = words[1]
                elif id_letter == 'D':
                    self.data_description = words[1]
                elif id_letter == 'E':
                    self.pmid = words[1:]
                elif id_letter == 'A':
                    self.authors = ' '.join(words[1:])
                elif id_letter == 'T':
                    self.title_of_article = ' '.join(words[1:])
                elif id_letter == 'J':
                    self.journal_reference = ' '.join(words[1:])

    def encoding_dictionary(self):
        """
        Get numerical values of AAindex for each amino acid
        """
        try:
            with open(self.file, 'r') as f:
                for line in f:
                    # try/ except "removes" empty lines
                    try:
                        words = line.split()
                        id_letter = words[0]
                    except IndexError:
                        break

                    # Extract numerical values of AAindex.
                    if id_letter == 'I':

                        keys = []
                        for word in words[1:]:
                            keys.append(word[0])
                            keys.append(word[-1])

                        values = []
                        for row in range(2):
                            line = f.readline()
                            strings = line.split()
                            for idx, string in enumerate(strings):
                                # Some amino acids may have no value
                                try:
                                    strings[idx] = float(string)
                                except ValueError:
                                    strings[idx] = None
                            values.append(strings)
                        values = np.reshape(np.array(values).T, len(keys))

                        return dict(zip(keys, values))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Probably you used an encoding technique option in combination with a model "
                "that was created using another encoding option (e.g. pypef ml -e aaidx -m "
                "ONEHOT -p TS.fasta) which is not allowed."
            )


class AAIndexEncoding:
    """
    converts the string sequence into a list of numericals
    using the AAindex translation library; Fourier trans-
    forming the numerical array that was translated by
    get_numerical_sequence --> do_fourier,computing the input
    matrices X and Y for the regressor (get_x_and_y).
    Returns FFT-ed encoded sequences (amplitudes),
    and raw_encoded sequences (raw_numerical_sequences).
    """
    def __init__(
            self,
            aaindex_file=None,
            sequences: list = None,
    ):
        aaidx = AAIndex(aaindex_file)
        self.dictionary = aaidx.encoding_dictionary()
        self.sequences = sequences

    def get_numerical_sequence(self, sequence):
        return np.array([self.dictionary[aminoacid] for aminoacid in sequence])

    @staticmethod
    def do_fourier(sequence):
        """
        This static function does the Fast Fourier Transform.
        Since the condition

            len(Array) = 2^k -> k = log_2(len(Array))  , k in N

        must be satisfied, the array must be reshaped (zero padding)
        if k is no integer value. The verbose parameter prints also
        the real and imaginary part separately.
        """
        threshold = 1e-8  # errors due to computer uncertainties
        k = np.log2(sequence.size)  # get exponent k
        mean = np.mean(sequence, axis=0)  # calculate mean of numerical array
        sequence = np.subtract(sequence, mean)  # subtract mean to avoid artificial effects of FT

        if abs(int(k) - k) > threshold:  # check if length of array fulfills previous equation
            numerical_sequence_reshaped = np.zeros(pow(2, (int(k) + 1)))  # reshape array
            for index, value in enumerate(sequence):
                numerical_sequence_reshaped[index] = value
            sequence = numerical_sequence_reshaped

        fourier_transformed = np.fft.fft(sequence)  # FFT
        ft_real = np.real(fourier_transformed)
        ft_imag = np.imag(fourier_transformed)

        x = np.linspace(1, sequence.size, sequence.size)  # frequencies
        x = x / max(x)  # normalization of frequency

        amplitude = ft_real * ft_real + ft_imag * ft_imag

        if max(amplitude) != 0:
            amplitude = np.true_divide(amplitude, max(amplitude))  # normalization of amplitude

        return amplitude, x

    def aaidx_and_or_fft_encode_sequence(self, sequence):
        """
        getting the input matrices X (FFT amplitudes) and Y (variant labels)
        """
        num = self.get_numerical_sequence(sequence)
        # Numerical sequence gets expended by zeros so that also different
        # lengths of sequences can be processed using '--nofft' option
        k = np.log2(len(num))
        if abs(int(k) - k) > 1e-8:  # check if length of array fulfills previous equation
            raw_numerical_seq = np.append(num, np.zeros(pow(2, (int(k) + 1)) - len(num)))  # reshape array
        else:
            raw_numerical_seq = num

        if None not in num:  # Not all amino acids could be encoded with the corresponding AAindex
            amplitudes_, frequencies_ = self.do_fourier(num)  # --> None values in encoded sequence
        else:  # If None in encoded Sequence, do not further use encoding (and FFT not possible)
            return [None], [None]
        # Fourier spectra are mirrored at frequency = 0.5 -> No more information at higher frequencies
        half = len(frequencies_) // 2  # // for integer division
        amplitude = amplitudes_[:half]   # FFT-ed encoded amino acid sequences
        # Appended zeros of raw encoding allow also prediction of differently sizes sequences

        # return -> X_fft_encoding, X_raw_encoding
        return amplitude, raw_numerical_seq

    def collect_encoded_sequences(self):
        """
        Loop over all sequences to encode each and collect
        and return all encoded sequences
        """
        # There may be amino acids without a value in AAindex
        # Skip these indices
        fft_encoded_sequences, raw_encoded_sequences = [], []
        for sequence in self.sequences:
            fft_encoded_sequence, raw_encoded_sequence = self.aaidx_and_or_fft_encode_sequence(sequence)
            if None in raw_encoded_sequence:
                return 'skip', 'skip'  # skipping this AAindex
            else:
                fft_encoded_sequences.append(fft_encoded_sequence)
                raw_encoded_sequences.append(raw_encoded_sequence)

        return fft_encoded_sequences, raw_encoded_sequences


class OneHotEncoding:
    """
    Generates an one-hot encoding, i.e. represents
    the current amino acid at a position as 1 and
    the other (19) amino acids as 0. Thus, the encoding
    of a sequence has the length 20 x sequence length.
    E.g. 'ACDY' --> [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    """
    def __init__(
            self,
            sequences: list
    ):
        self.sequences = sequences
        self.amino_acids = amino_acids  # imported, 20 standard AAs

    def encoding_dict(self) -> dict[str, np.ndarray]:
        encoding_dict = {}
        for idx, amino_acid in enumerate(self.amino_acids):
            encoding_vector = np.zeros(20, dtype=int)
            encoding_vector[idx] = 1
            encoding_dict.update({amino_acid: encoding_vector})
        return encoding_dict

    def one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        encoded_sequence = []
        for aminoacid in sequence:
            encoded_sequence.append(self.encoding_dict()[aminoacid])
        return np.concatenate(encoded_sequence)

    def collect_encoded_sequences(self, silence=False) -> list:
        encoded_sequences = []
        for sequence in tqdm(self.sequences, disable=silence):
            encoded_sequences.append(self.one_hot_encode_sequence(sequence))
        return np.array(encoded_sequences)

def pls_loocv(
        x_train: np.ndarray,
        y_train: np.ndarray
) -> Union[tuple[str, str], tuple[PLSRegression, dict]]:
    """
    PLS regression with LOOCV n_components tuning as described by Cadet et al.
    https://doi.org/10.1186/s12859-018-2407-8
    https://doi.org/10.1038/s41598-018-35033-y
    Hyperparameter (N component) tuning of PLS regressor, can achieve slightly better
    """
    mean_squared_error_list = []
    for n_comp in range(1, 10):  # n_comp = 1, 2,..., 9
        try:
            pls = PLSRegression(n_components=n_comp)
            loo = LeaveOneOut()
            y_pred_loo = []
            y_test_loo = []
            for train, test in loo.split(x_train):
                x_learn_loo = []
                y_learn_loo = []
                x_test_loo = []
                for j in train:
                    x_learn_loo.append(x_train[j])
                    y_learn_loo.append(y_train[j])
                for k in test:
                    x_test_loo.append(x_train[k])
                    y_test_loo.append(y_train[k])
                x_learn_loo = np.array(x_learn_loo)
                x_test_loo = np.array(x_test_loo)
                y_learn_loo = np.array(y_learn_loo)
                try:
                    pls.fit(x_learn_loo, y_learn_loo)
                except ValueError:  # scipy/linalg/decomp_svd.py ValueError:
                    continue        # illegal value in %dth argument of internal gesdd
                y_pred_loo.append(pls.predict(x_test_loo)[0][0])
        except np.linalg.LinAlgError:  # numpy.linalg.LinAlgError: SVD did not converge
            continue
        try:
            mse = mean_squared_error(y_test_loo, y_pred_loo)
            mean_squared_error_list.append(mse)
        except ValueError:  # MSE could not be calculated (No values due to numpy.linalg.LinAlgErrors)
            return 'skip', 'skip'
    mean_squared_error_list = np.array(mean_squared_error_list)
    idx = np.where(mean_squared_error_list == np.min(mean_squared_error_list))[0][0] + 1
    # Model is fitted with best n_components (lowest MSE)
    best_params = {'n_components': idx}
    regressor_ = PLSRegression(n_components=best_params.get('n_components'))

    return regressor_, best_params


def cv_regression_options(regressor: str) -> GridSearchCV:
    """
    Returns the CVRegressor with the tunable regression-specific hyperparameter grid
    for training a regression model.
    Regression options are
        - Partial Least Squares Regression
        - Random Forest Regression
        - Support Vector Machines Regression
        - Multilayer Perceptron Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet Regression
    """
    if regressor == 'pls':
        params = {'n_components': list(np.arange(1, 10))}  # n_comp = 1, 2,..., 9
        regressor_ = GridSearchCV(PLSRegression(), param_grid=params, cv=5)  # iid in future versions redundant

    elif regressor == 'rf':
        params = {  # similar parameter grid as Xu et al., https://doi.org/10.1021/acs.jcim.0c00073
            'random_state': [42],  # state determined
            'n_estimators': [100, 250, 500, 1000],  # number of individual decision trees in the forest
            'max_features': ['auto', 'sqrt', 'log2']  # “auto” -> max_features=n_features,
            # “sqrt” -> max_features=sqrt(n_features) “log2” -> max_features=log2(n_features)
        }
        regressor_ = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5)

    elif regressor == 'svr':
        params = {  # similar parameter grid as Xu et al.
            'C': [2 ** 0, 2 ** 2, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12],  # Regularization parameter
            'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]  # often 1 / n_features or 1 / (n_features * X.var())
        }
        regressor_ = GridSearchCV(SVR(), param_grid=params, cv=5)

    elif regressor == 'mlp':
        params = {
            # feedforward network trained via backpropagation – here only using a single hidden layer
            'hidden_layer_sizes': [i for i in range(1, 12)],  # size of hidden layer [(1,), (2,), ..., (12,)]
            'activation': ['relu'],  # rectified linear unit
            'solver': ['adam', 'lbfgs'],  # ADAM: A Method for Stochastic Optimization , or Limited-memory BFGS
            'learning_rate': ['constant'],  # learning rate given by ‘learning_rate_init’
            'learning_rate_init': [0.001, 0.01, 0.1],  # only used when solver=’sgd’ or ‘adam’
            'max_iter': [1000, 200],  # for stochastic solvers (‘sgd’, ‘adam’) determines epochs
            'random_state': [42]
        }
        regressor_ = GridSearchCV(MLPRegressor(), param_grid=params, cv=5)

    elif regressor == 'elasticnet' or regressor == 'l1l2':
        # Linear regression with combined L1 and L2 priors as regularizer.
        # min(w):  ||y - Xw||^2_2 + alpha*l1_ratio*||w||_1 + 0.5*alpha*(1 - l1_ratio)*||w||^2_2
        params = {
            'alpha': np.logspace(-6, 6, num=100)  # {1.000E-06, 1.322E-06, 1.748E-06, ..., 1.000E06}
        }
        regressor_ = GridSearchCV(ElasticNet(), param_grid=params, cv=5)

    elif regressor == 'ridge' or regressor == 'l2':
        # Performs L2 regularization, i.e., adds penalty equivalent to square of the magnitude of coefficients
        # Majorly used to prevent overfitting, since it includes all the features
        # min(w): ||y - Xw||^2_2 + alpha*||w||^2_2
        # in case of exorbitantly high features, it will pose computational challenges.
        params = {
            # alpha = 0 is equivalent to an ordinary least square regression
            # higher values of alpha reduce overfitting, significantly high values can
            # cause underfitting as well (e.g., regularization strength alpha = 5)
            'alpha': np.logspace(-6, 6, num=100)  # {1.000E-06, 1.322E-06, 1.748E-06, ..., 1.000E06}
        }
        regressor_ = GridSearchCV(Ridge(), param_grid=params, cv=5)

    elif regressor == 'lasso' or regressor == 'l1':
        # Lasso model fit with Least Angle Regression a.k.a. Lars.
        # Performs L1 regularization, i.e., adds penalty equivalent to absolute value of the magnitude of coefficients
        # min(w): ||y - Xw||^2_2 + alpha*||w||_1
        # Provides sparse solutions: computationally efficient as features with zero coefficients can be ignored
        params = {
            # alpha = 0 is equivalent to an ordinary least square Regression
            'alpha': np.logspace(-6, 6, num=100)  # {1.000E-06, 1.322E-06, 1.748E-06, ..., 1.000E06}
        }
        regressor_ = GridSearchCV(Lasso(), param_grid=params, cv=5)

    else:
        raise SystemError("Did not find specified regression model as valid option. See '--help' for valid "
                          "regression model options.")

    return regressor_


def get_regressor_performances(
        x_learn: list,
        x_test: list,
        y_learn: list,
        y_test: list,
        regressor: str = 'pls',
        verbose: bool = False
):
    """
    The function get_regressor_performances takes features and labels from the
    learning and test set.

    When using 'pls_loocv' as regressor, the MSE is calculated for all LOOCV
    sets for predicted vs true labels (mse = mean_squared_error(y_test_loo, y_pred_loo)
    for a fixed number of components for PLS regression.
    In the next iteration, the number of components is increased by 1 (number_of_components += 1)
    and the MSE is calculated for this regressor. The loop breaks if i > 9.
    Finally, the model of the single AAindex model with the lowest MSE is chosen.

    When using other regressors the parameters are tuned using GridSearchCV.

    This function returnes performance (R2, (N)RMSE, Pearson's r) and model parameters.
    """
    regressor = regressor.lower()
    best_params = None

    if regressor == 'pls_loocv':  # PLS LOOCV tuning
        regressor_, best_params = pls_loocv(x_learn, y_learn)
        if regressor_ == 'skip':
            return [None, None, None, None, None, regressor, None]

    # other regression options (k-fold CV tuning)
    else:
        regressor_ = cv_regression_options(regressor)
    try:
        if verbose:
            logger.info('CV-based training of regression model...')
        regressor_.fit(x_learn, y_learn)  # fit model
    except ValueError:  # scipy/linalg/decomp_svd.py --> ValueError('illegal value in %dth argument of internal gesdd'
        return [None, None, None, None, None, regressor, None]

    if regressor != 'pls_loocv':  # take best parameters for the regressor and the AAindex
        best_params = regressor_.best_params_

    y_pred = []
    try:
        for y_p in regressor_.predict(x_test):  # predict validation entries with fitted model
            y_pred.append(float(y_p))
    except ValueError:
        raise ValueError("Above error message exception indicates that your test set may be empty.")

    r2, rmse, nrmse, pearson_r, spearman_rho = get_performances(y_test, y_pred)

    return r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params


def performance_list(
        train_set: str,
        test_set: str,
        encoding: str = 'aaidx',
        regressor: str = 'pls',
        no_fft: bool = False,
        sort: str = '1',
        couplings_file: str = None,
        threads: int = 1  # for parallelization of DCA-based encoding
):
    """
    returns the sorted list of all the model parameters and the
    performance values (R2 etc.) from function get_performances.
    """
    encoding = encoding.lower()
    performance_list = []
    train_sequences, train_variants, y_train = get_sequences_from_file(train_set)
    test_sequences, test_variants, y_test = get_sequences_from_file(test_set)
    if encoding == 'onehot':  # OneHot-based encoding
        x_onehot_train = OneHotEncoding(train_sequences)
        x_onehot_test = OneHotEncoding(test_sequences)
        x_train = x_onehot_train.collect_encoded_sequences()
        x_test = x_onehot_test.collect_encoded_sequences()
        r2, rmse, nrmse, pearson_r, spearman_rho, regression_model, \
            params = get_regressor_performances(x_train, x_test, y_train, y_test, regressor, verbose=True)
        if r2 is not None:  # get_regressor_performances() returns None for metrics if MSE can't be calculated
            performance_list.append([
                'ONEHOTMODEL', r2, rmse, nrmse, pearson_r,
                spearman_rho, regression_model, params
            ])
    elif encoding == 'dca':
        if threads > 1:  # NaNs are already being removed by the called function
            dca_encoder = DCAEncoding(couplings_file, verbose=False)
            logger.info('Parallel encoding (runs DCA encoding silent, '
                        'no information about non-encoded variant positions)...')
            train_variants, x_train, y_train = get_dca_data_parallel(train_variants, y_train, dca_encoder, threads)
            test_variants, x_test, y_test = get_dca_data_parallel(test_variants, y_test, dca_encoder, threads)
        else:
            dca_encoder = DCAEncoding(couplings_file)
            x_train_ = dca_encoder.collect_encoded_sequences(train_variants)
            x_test_ = dca_encoder.collect_encoded_sequences(test_variants)
            # NaNs must still be removed
            x_train, y_train = remove_nan_encoded_positions(x_train_, y_train)
            x_test, y_test = remove_nan_encoded_positions(x_test_, y_test)
        r2, rmse, nrmse, pearson_r, spearman_rho, regression_model, \
            params = get_regressor_performances(x_train, x_test, y_train, y_test, regressor, verbose=True)
        if r2 is not None:  # get_regressor_performances() returns None for metrics if MSE can't be calculated
            performance_list.append([
                'DCAMLMODEL', r2, rmse, nrmse, pearson_r,
                spearman_rho, regression_model, params
            ])

    else:  # AAindex-based encoding
        aa_indices = [file for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]
        # loop over the 566 AAindex entries, encode with each AAindex and test performance
        # can be seen as a AAindex hyperparameter search on the test set  --> also see CV performance
        # in created folder across all data to ensure a relatively well generalizable model
        for index, aaindex in enumerate(tqdm(aa_indices)):
            x_aaidx_train = AAIndexEncoding(full_aaidx_txt_path(aaindex), train_sequences)
            if not no_fft:  # X is FFT-ed of encoded alphabetical sequence
                x_train, _ = x_aaidx_train.collect_encoded_sequences()
            else:  # X is raw encoded of alphabetical sequence
                _, x_train = x_aaidx_train.collect_encoded_sequences()
            x_aaidx_test = AAIndexEncoding(full_aaidx_txt_path(aaindex), test_sequences)
            if not no_fft:  # X is FFT-ed of the encoded alphabetical sequence
                x_test, _ = x_aaidx_test.collect_encoded_sequences()
            else:  # X is the raw encoded of alphabetical sequence
                _, x_test = x_aaidx_test.collect_encoded_sequences()
            # If x_learn or x_test contains None, the sequence could not be (fully) encoded --> Skip
            if x_train == 'skip' or x_test == 'skip':
                continue  # skip the rest and do next iteration
            r2, rmse, nrmse, pearson_r, spearman_rho, regression_model, \
                params = get_regressor_performances(x_train, x_test, y_train, y_test, regressor)
            if r2 is not None:  # get_regressor_performances() returns None for metrics if MSE can't be calculated
                performance_list.append([
                    aaindex, r2, rmse, nrmse, pearson_r,
                    spearman_rho, regression_model, params
                ])

    try:
        sort = int(sort)
        if sort == 2 or sort == 3:
            performance_list.sort(key=lambda x: x[sort])
        else:
            performance_list.sort(key=lambda x: x[sort], reverse=True)

    except ValueError:
        raise ValueError("Choose between options 1 to 5 (R2, RMSE, NRMSE, Pearson's r, Spearman's rho.")

    return performance_list


def formatted_output(
        performance_list,
        no_fft=False,
        minimum_r2=0.0
):
    """
    Takes the sorted list from function r2_list and writes the model names with an R2 ≥ 0
    as well as the corresponding parameters for each model so that the user gets
    a list (Model_Results.txt) of the top ranking models for the given validation set.
    """

    index, value, value2, value3, value4, value5, regression_model, params = [], [], [], [], [], [], [], []

    for (idx, val, val2, val3, val4, val5, r_m, pam) in performance_list:
        if val >= minimum_r2:
            index.append(get_basename(idx))
            value.append('{:f}'.format(val))
            value2.append('{:f}'.format(val2))
            value3.append('{:f}'.format(val3))
            value4.append('{:f}'.format(val4))
            value5.append('{:f}'.format(val5))
            regression_model.append(r_m.upper())
            params.append(pam)

    if len(value) == 0:  # Criterion of not finding suitable model is defined by Minimum_R2
        raise ValueError('No model with positive R2.')

    data = np.array([index, value, value2, value3, value4, value5, regression_model, params]).T
    col_width = max(len(str(value)) for row in data for value in row[:-1]) + 5

    head = ['Index', 'R2', 'RMSE', 'NRMSE', 'Pearson\'s r', 'Spearman\'s rho', 'Regression', 'Model parameters']
    with open('Model_Results.txt', 'w') as f:
        if no_fft is not False:
            f.write("No FFT used in this model construction, performance"
                    " represents model accuracies on raw encoded sequence data.\n\n")

        heading = "".join(caption.ljust(col_width) for caption in head) + '\n'
        f.write(heading)

        row_length = []
        for row in data:
            row_ = "".join(str(value).ljust(col_width) for value in row) + '\n'
            row_length.append(len(row_))
        row_length_max = max(row_length)
        f.write(row_length_max * '-' + '\n')

        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')


def cross_validation(
        x: np.ndarray,
        y: np.ndarray,
        regressor_,  # Union[PLSRegression, Ridge, Lasso, ElasticNet, ...]
        n_samples: int = 5):
    """
    Perform k-fold cross-validation on the input data (encoded sequences and
    corresponding fitness values) with default k = 5. Returns all predicted
    fitness values of the length y (e.g. (1/5)*len(y) * 5 = 1*len(y)).
    """
    # perform k-fold cross-validation on all data
    # k = Number of splits, change for changing k in k-fold splitting, default: 5
    y_test_total = []
    y_predicted_total = []

    kf = KFold(n_splits=n_samples, shuffle=True)

    for train_index, test_index in kf.split(y):
        y = np.array(y)
        try:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for numbers in y_test:
                y_test_total.append(numbers)
            regressor_.fit(x_train, y_train)
            y_pred_test = regressor_.predict(x_test)

            for values in y_pred_test:
                y_predicted_total.append(float(values))
        except UserWarning:
            continue

    return y_test_total, y_predicted_total


def get_performances(
        y_true: list,
        y_pred: list
) -> tuple[float, float, float, float, float]:
    """
    Description
    -----------
    Gets performance metrics (R^2, RMSE, NRMSE, Pearson's r, Spearman's rho)
    between y_true and y_pred.

    Parameters
    -----------
    y_true: list
        Measured fitness values.
    y_pred: list
        Predicted fitness values.

    Returns
    -----------
    r_squared: float
    rmse: float
    nrmse: float
    pearson_r: float
    spearman_rho: float
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    r_squared = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    stddev = np.std(y_true, ddof=1)
    nrmse = rmse / stddev
    with warnings.catch_warnings():  # catching RunTime warning when there's no variance in an array,
        warnings.simplefilter("ignore")  # e.g. [2, 2, 2, 2] which would mean divide by zero
        pearson_r = np.corrcoef(y_true, y_pred)[0][1]
        spearman_rho = stats.spearmanr(y_true, y_pred)[0]

    return r_squared, rmse, nrmse, pearson_r, spearman_rho


def get_regressor(
        regressor: str,
        parameter: dict
):
    """
    Returns the tuned CVRegressor with the tuned hyperparameters.
    Regression options are
        - Partial Least Squares Regression
        - Random Forest Regression
        - Support Vector Machines Regression
        - Multilayer Perceptron Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet Regression
    """
    if regressor == 'pls' or regressor == 'pls_loocv':
        # n_components according to lowest MSE for validation set
        regressor_ = PLSRegression(n_components=parameter.get('n_components'))

    elif regressor == 'rf':
        regressor_ = RandomForestRegressor(
            random_state=parameter.get('random_state'),
            n_estimators=parameter.get('n_estimators'),
            max_features=parameter.get('max_features')
        )

    elif regressor == 'svr':
        regressor_ = SVR(C=parameter.get('C'), gamma=parameter.get('gamma'))

    elif regressor == 'mlp':
        regressor_ = MLPRegressor(
            hidden_layer_sizes=parameter.get('hidden_layer_sizes'),
            activation=parameter.get('activation'),
            solver=parameter.get('solver'),
            learning_rate=parameter.get('learning_rate'),
            learning_rate_init=parameter.get('learning_rate_init'),
            max_iter=parameter.get('max_iter'),
            random_state=parameter.get('random_state')
        )

    elif regressor == 'ridge' or regressor == 'l2':
        regressor_ = Ridge(
            alpha=parameter.get('alpha')
        )

    elif regressor == 'lasso' or regressor == 'l1':
        regressor_ = Lasso(
            alpha=parameter.get('alpha')
        )

    elif regressor == 'elasticnet' or regressor == 'l1l2':
        regressor_ = ElasticNet(
            alpha=parameter.get('alpha')
        )

    else:
        raise SystemError("Did not find specified regression model as valid option. "
                          "See '--help' for valid regression model options.")

    return regressor_


def save_model(
        path,
        performance_list,
        training_set,
        test_set,
        threshold=5,
        encoding='aaidx',
        regressor='pls',
        no_fft=False,
        train_on_all=False,
        couplings_file=None,
        threads: int = 1
):
    """
    Function Save_Model saves the best -s THRESHOLD models as 'Pickle'
    files (pickle.dump), which can be loaded again for doing predictions.
    Also, in Save_Model included is the def cross_validation-based computing
    of the k-fold CV performance of the n component-optimized model on all
    data (learning + test set); by default  k  is 5 (n_samples = 5).
    Plots of the CV performance for the t best models are stored inside the
    folder CV_performance.
    """
    logger.info('Encoding and cross validation on all data (creating folder CV_performance)...')
    regressor = regressor.lower()
    try:
        os.mkdir('CV_performance')
    except FileExistsError:
        pass
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass
    train_sequences, train_variants, y_train = get_sequences_from_file(training_set)
    test_sequences, test_variants, y_test = get_sequences_from_file(test_set)
    for i, t in enumerate(range(threshold)):
        try:
            idx = performance_list[t][0]
            parameter = performance_list[t][7]

            if encoding == 'onehot' or encoding == 'dca':
                name = idx
            else:
                name = get_basename(idx)
            cv_filename = os.path.join('CV_performance', f'{name}_{regressor.upper()}_CV_Results.txt')
            try:
                os.remove(cv_filename)
            except FileNotFoundError:
                pass
            file = open(cv_filename, 'w')
            file.write('5-fold cross-validated performance of top '
                       'models for validation set across all data.\n\n')
            if no_fft:
                file.write("No FFT used in this model construction, performance represents"
                           " model accuracies on raw encoded sequence data.\n\n")
            file.close()

            # Estimating the CV performance of the n_component-fitted model on all data
            if encoding == 'aaidx':  # AAindex encoding technique
                x_aaidx_train = AAIndexEncoding(full_aaidx_txt_path(idx), train_sequences)
                x_aaidx_test = AAIndexEncoding(full_aaidx_txt_path(idx), test_sequences)
                if no_fft is False:  # use FFT on encoded sequences (default)
                    x_train, _ = x_aaidx_train.collect_encoded_sequences()
                    x_test, _ = x_aaidx_test.collect_encoded_sequences()
                else:  # use raw encoding (no FFT used on encoded sequences)
                    _, x_train = x_aaidx_train.collect_encoded_sequences()
                    _, x_test = x_aaidx_test.collect_encoded_sequences()
            elif encoding == 'onehot':  # OneHot encoding technique
                x_onehot_train = OneHotEncoding(train_sequences)
                x_onehot_test = OneHotEncoding(test_sequences)
                x_train = x_onehot_train.collect_encoded_sequences()
                x_test = x_onehot_test.collect_encoded_sequences()
            else:  # DCA
                dca_encoder = DCAEncoding(couplings_file, verbose=False)
                if threads > 1:  # parallelization of encoding, NaNs are already being removed by the called function
                    train_variants, x_train, y_train = get_dca_data_parallel(train_variants, y_train, dca_encoder, threads)
                    test_variants, x_test, y_test = get_dca_data_parallel(test_variants, y_test, dca_encoder, threads)
                else:  # encode using a single thread
                    x_train_ = dca_encoder.collect_encoded_sequences(train_variants)
                    x_test_ = dca_encoder.collect_encoded_sequences(test_variants)

                    x_train, y_train, train_variants = remove_nan_encoded_positions(x_train_, y_train, train_variants)
                    x_test, y_test, test_variants = remove_nan_encoded_positions(x_test_, y_test, test_variants)

                    assert len(x_train) == len(y_train) == len(train_variants)
                    assert len(x_test) == len(y_test) == len(test_variants)

            x = np.concatenate([x_train, x_test])
            y = np.concatenate([y_train, y_test])

            regressor_ = get_regressor(regressor, parameter)
            # perform 5-fold cross-validation on all data (on X and Y)
            n_samples = 5
            y_test_total, y_predicted_total = cross_validation(x, y, regressor_, n_samples)

            r_squared, rmse, nrmse, pearson_r, spearman_rho = \
                get_performances(y_test_total, y_predicted_total)

            with open(cv_filename, 'a') as f:
                f.write('Regression type: {}; Parameter: {}; Encoding index: {}\n'.format(
                    regressor.upper(), parameter, name))
                f.write('R2 = {:.5f}; RMSE = {:.5f}; NRMSE = {:.5f}; Pearson\'s r = {:.5f};'
                        ' Spearman\'s rho = {:.5f}\n\n'.format(r_squared, rmse, nrmse, pearson_r, spearman_rho))

            figure, ax = plt.subplots()
            legend = r'$R^2$' + f' = {r_squared:.3f}' + f'\nRMSE = {rmse:.3f}' + f'\nNRMSE = {nrmse:.3f}' + \
                     f'\nPearson\'s ' + r'$r$'+f' = {pearson_r:.3f}' + f'\nSpearman\'s ' + \
                     fr'$\rho$ = {spearman_rho:.3f}' + '\n' + fr'($N$ = {len(y_test_total)})'
            ax.scatter(
                y_test_total, y_predicted_total,
                marker='o', s=20, linewidths=0.5, edgecolor='black', label=legend, alpha=0.8
            )
            ax.plot([min(y_test_total) - 1, max(y_test_total) + 1],
                    [min(y_predicted_total) - 1, max(y_predicted_total) + 1], 'k', lw=0.5)

            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            ax.legend(prop={'size': 8})
            plt.savefig(os.path.join('CV_performance', f'{name}_{regressor.upper()}_{n_samples}-fold-CV.png'), dpi=300)
            plt.close('all')

            if train_on_all:  # Train model hyperparameters based on all available data (training + test set)
                # But, no generalization performance can be estimated as the model also trained on the test set
                regressor_.fit(x, y)
            else:
                # fit (only) on full learning set (FFT or noFFT is defined already above)
                regressor_.fit(x_train, y_train)

            y_test_pred = []  # 2D prediction array output to 1D
            for value in regressor_.predict(x_test):
                y_test_pred.append(float(value))
            y_test_pred = np.array(y_test_pred)

            plot_y_true_vs_y_pred(
                y_true=y_test,
                y_pred=y_test_pred,
                variants=test_variants,
                label=True,
                hybrid=False,
                name=f'{name}_{regressor.upper()}_'
            )

            file = open(os.path.join(path, 'Pickles', name), 'wb')
            pickle.dump(regressor_, file)
            file.close()

        except IndexError:
            break

        if encoding == 'onehot' or encoding == 'dca':   # only 1 model/encoding  -->
            break                                       # no further iteration needed, thus break loop


def predict(
        path=None,
        prediction_set=None,
        model=None,
        encoding='aaidx',
        mult_path=None,
        no_fft=False,
        variants=None,
        sequences=None,
        couplings_file=None,
        dca_encoder: DCAEncoding = None,
        threads: int = 1  # for parallelization of DCA-based encoding
):
    """
    The function Predict is used to perform predictions.
    Saved pickle files of models will be loaded again:
      mod = pickle.load(file)
    and used for predicting the label y (y = mod.predict(x))
    of sequences given in the Prediction_Set.fasta.
    """
    if model is not None:
        # model defines pickle to load (and thus determines encoding AAidx)
        file = open(os.path.join(path, 'Pickles', str(model)), 'rb')
        loaded_model = pickle.load(file)
        file.close()
        aaidx = full_aaidx_txt_path(str(model) + '.txt')
    else:
        aaidx = None
        loaded_model = None
    if sequences is None and variants is None:  # File-based prediction with AAidx
        sequences, variants, _ = get_sequences_from_file(prediction_set, mult_path)

    if encoding == 'aaidx':  # AAidx
        # Directed evolution with AAidx, means sequences is only 1 sequence, thus
        x_aaidx = AAIndexEncoding(aaidx, np.atleast_1d(sequences).tolist())  # at least 1D if only 1 sequence inputted
        x, x_raw = x_aaidx.collect_encoded_sequences()
    elif encoding == 'onehot':  # OneHot
        x_onehot = OneHotEncoding(np.atleast_1d(sequences).tolist())
        if len(sequences) == 1:
            silence = True
        else:
            silence = False
        x = x_onehot.collect_encoded_sequences(silence=silence)
        x_raw = None
    else:  # DCA
        if dca_encoder is not None:  # use dca_encoder from directed evolution, single thread
            x_ = dca_encoder.collect_encoded_sequences(variants)
            x, variants = remove_nan_encoded_positions(x_, variants)
            x_raw = None

        else:
            if threads > 1:
                dca_encoder = DCAEncoding(couplings_file, verbose=False)
                logger.info('Parallel encoding (runs DCA encoding silent, '
                            'no information about non-encoded variant positions)...')
                # parallel encoding of variants, NaNs are already being removed by the called function
                variants, x, _ = get_dca_data_parallel(
                    variants, list(np.zeros(len(variants))), dca_encoder, threads
                )
            else:  # single thread running
                dca_encoder = DCAEncoding(couplings_file)
                x_ = dca_encoder.collect_encoded_sequences(variants)
                x, variants = remove_nan_encoded_positions(x_, variants)
                x_raw = None
        if type(x) == list:
            if not x:
                return 'skip'
        elif type(x) == np.ndarray:
            if not x.any():
                return 'skip'

    assert len(variants) == len(x)

    try:
        if no_fft and encoding == 'aaidx':  # predict using raw AAidx encoding (without FFT)
            ys = loaded_model.predict(x_raw)
        else:  # predict AAidx-FFTed or onehot or DCA-based encoding
            ys = loaded_model.predict(x)
    except ValueError:
        raise SystemError(
            "If you used an encoding such as onehot, make sure to use the correct model, e.g. -m ONEHOT. "
            "If you used an AAindex-encoded model you likely tried to predict using a model encoded with "
            "(or without) FFT featurization ('--nofft') while the model was trained without (or with) FFT "
            "featurization so check Model_Results.txt line 1, if the models were trained with or without FFT."
        )
    except AttributeError:
        raise SystemError(
            "The model specified is likely a hybrid or pure statistical DCA (and no pure ML model).\n"
            "Check the specified model provided via the \'-m\' flag."
        )

    predictions = [(float(ys[i]), variants[i]) for i in range(len(ys))]  # List of tuples

    # Pay attention if increased negative values would define a better variant --> use negative flag
    predictions.sort()
    predictions.reverse()
    # if predictions array too large? if Mult_Path is not None: predictions = predictions[:100000]

    return predictions


def predictions_out(
        predictions,
        model,
        prediction_set,
        path: str = ''
):
    """
    Writes predictions (of the new sequence space) to text file(s).
    """
    name, value = [], []
    for (val, nam) in predictions:
        name.append(nam)
        value.append('{:f}'.format(val))

    data = np.array([name, value]).T
    col_width = max(len(str(value)) for row in data for value in row) + 5

    head = ['Name', 'Prediction']
    path_ = os.path.join(path, 'Predictions_' + str(model) + '_' + str(prediction_set)[:-6] + '.txt')
    with open(path_, 'w') as f:
        f.write("".join(caption.ljust(col_width) for caption in head) + '\n')
        f.write(len(head)*col_width*'-' + '\n')
        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')


def predict_and_plot(
        path,
        fasta_file,
        model,
        encoding='aaidx',
        label=False,
        color=False,
        y_wt=1,
        no_fft=False,
        couplings_file=None,
        threads: int = None
):
    """
    Function Plot is used to make plots of the validation process and
    shows predicted (Y_pred) vs. measured/"true" (Y_true) protein fitness and
    calculates the corresponding model performance (R2, (N)RMSE, Pearson's r).
    Also allows colored version plotting to classify predictions in true or
    false positive or negative predictions.
    """
    sequences_test, variants_test, y_test = get_sequences_from_file(fasta_file)
    if encoding == 'aaidx':  # AAindex-based encoding
        aaidx = full_aaidx_txt_path(str(model) + '.txt')
        aaidx_encoded = AAIndexEncoding(aaidx, sequences_test)
        x, x_raw = aaidx_encoded.collect_encoded_sequences()
    elif encoding == 'onehot':  # OneHot-based encoding
        onehot_encoded = OneHotEncoding(sequences_test)
        x = onehot_encoded.collect_encoded_sequences()
        x_raw = None
    else:  # DCA
        if threads is None:
            threads = 1
        dca_encoder = DCAEncoding(couplings_file)
        x_raw = None
        if threads > 1:  # NaNs are already being removed by the called function, for ys just filling 0's
            variants_test, x, y_test = get_dca_data_parallel(variants_test, y_test, dca_encoder, threads)
        else:
            x_ = dca_encoder.collect_encoded_sequences(variants_test)
            x, variants_test, y_test = remove_nan_encoded_positions(x_, variants_test, y_test)
        assert len(x) == len(variants_test) == len(y_test)

    try:
        file = open(os.path.join(path, 'Pickles', str(model)), 'rb')
        model_ = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Did not find specified model: {str(model)}. You can define the "
            f"threshold of models to be saved; e.g. with pypef run -l LS.fasta "
            f"-v VS.fasta -s 10 and pay attention to capitalization of model "
            f"names (e.g. -m PONP930101 or -m ONEHOT."
        )
    y_pred = []
    try:
        # predicting (again) with (re-)loaded model (that was trained on training or full data)
        if no_fft and encoding == 'aaidx':  # predict using raw AAidx encoding
            y_pred_ = model_.predict(x_raw)
        else:  # predict AAidx-FFTed or onehot or DCA-based encoding
            y_pred_ = model_.predict(x)
    except ValueError:
        raise ValueError(
            "You likely tried to plot a test set with (or without) "
            "FFT featurization ('--nofft') while the model was trained "
            "without (or with) FFT featurization. Check the Model_Results.txt "
            "line 1, if the models were trained using FFT."
        )
    for y_p in y_pred_:
        y_pred.append(float(y_p))
    r_squared, rmse, nrmse, pearson_r, spearman_rho = \
        get_performances(y_test, y_pred)
    legend = '$R^2$ = {}\nRMSE = {}\nNRMSE = {}\nPearson\'s $r$ = {}\nSpearman\'s '.format(
        round(r_squared, 3), round(rmse, 3), round(nrmse, 3), round(pearson_r, 3)) + \
             r'$\rho$ = {}'.format(round(spearman_rho, 3)) + \
             '\n' + r'($N$ = {})'.format(len(y_test))
    x = np.linspace(min(y_pred), max(y_pred), 100)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred,
               label=legend, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.8)
    ax.legend(prop={'size': 8})
    ax.plot(x, x, color='black', linewidth=0.5)  # plot diagonal line
    if label is not False:
        logger.info('Adjusting variant labels for plotting can take some '
                    'time (the limit for labeling is 150 data points)...')
        if len(y_test) < 150:
            texts = [ax.text(y_test[i], y_pred[i], txt, fontsize=4) for i, txt in enumerate(variants_test)]
            adjust_text(texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5, lim=250)
        else:
            logger.info("Terminating label process. Too many variants "
                        "(> 150) for plotting (labels would overlap).")
    if color is not False:
        try:
            y_wt = float(y_wt)
        except TypeError:
            raise TypeError('Needs label value of WT (y_WT) when making color plot (e.g. --color --ywt 1.0)')
        if y_wt == 0:
            y_wt = 1E-9  # choose a value close to zero
        true_v, true_p, false_v, false_p = [], [], [], []
        for i, v in enumerate(y_test):
            if y_test[i] / y_wt >= 1 and y_pred[i] / y_wt >= 1:
                true_v.append(y_test[i]), true_p.append(float(y_pred[i]))
            elif y_test[i] / y_wt < 1 and y_pred[i] / y_wt < 1:
                true_v.append(y_test[i]), true_p.append(float(y_pred[i]))
            else:
                false_v.append(y_test[i]), false_p.append(float(y_pred[i]))
        try:
            ax.scatter(true_v, true_p, color='tab:blue', marker='o', s=20, linewidths=0.5, edgecolor='black')
        except IndexError:
            pass
        try:
            ax.scatter(false_v, false_p, color='tab:red', marker='o', s=20, linewidths=0.5, edgecolor='black')
        except IndexError:
            pass
        if (y_wt - min(y_test)) < (max(y_test) - y_wt):
            limit_y_true = float(max(y_test) - y_wt)
        else:
            limit_y_true = float(y_wt - min(y_test))
        limit_y_true = limit_y_true * 1.1
        if (y_wt - min(y_pred)) < (max(y_pred) - y_wt):
            limit_y_pred = float(max(y_pred) - y_wt)
        else:
            limit_y_pred = float(y_wt - min(y_pred))
        limit_y_pred = limit_y_pred * 1.1
        plt.vlines(x=(y_wt + limit_y_true) - (((y_wt + limit_y_true) - (y_wt - limit_y_true)) / 2),
                   ymin=y_wt - limit_y_pred, ymax=y_wt + limit_y_pred, color='grey', linewidth=0.5)
        plt.hlines(y=(y_wt + limit_y_pred) - (((y_wt + limit_y_pred) - (y_wt - limit_y_pred)) / 2),
                   xmin=y_wt - limit_y_true, xmax=y_wt + limit_y_true, color='grey', linewidth=0.5)
        crossline = np.linspace(y_wt - limit_y_true, y_wt + limit_y_true)
        plt.plot(crossline, crossline, color='black', linewidth=0.25)
        steps = float(abs(max(y_pred)))
        gradient = []
        for x in np.linspace(0, steps, 100):
            # arr = np.linspace(x/steps, 1-x/steps, steps)
            arr = 1 - np.linspace(x / steps, 1 - x / steps, 100)
            gradient.append(arr)
        gradient = np.array(gradient)
        plt.imshow(gradient, extent=[y_wt - limit_y_true, y_wt + limit_y_true,
                                     y_wt - limit_y_pred, y_wt + limit_y_pred],
                   aspect='auto', alpha=0.8, cmap='coolwarm')  # RdYlGn
        plt.xlim([y_wt - limit_y_true, y_wt + limit_y_true])
        plt.ylim([y_wt - limit_y_pred, y_wt + limit_y_pred])
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.legend(prop={'size': 8})
    plt.savefig(str(model) + '_' + str(fasta_file[:-6]) + '.png', dpi=500)


def plot_y_true_vs_y_pred(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        variants: np.ndarray,  # just required for labeling
        label=False,
        hybrid=False,
        name: str = ''
):
    """
    Plots predicted versus true values using the hybrid model for prediction.
    Function called by function predict_ps.
    """
    figure, ax = plt.subplots()
    if hybrid:
        spearman_rho = stats.spearmanr(y_true, y_pred)[0]
        ax.scatter(y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.7,
                   label=f'Spearman\'s ' + fr'$\rho$ = {spearman_rho:.3f}' + '\n' + fr'($N$ = {len(y_true)})')
        file_name = name + 'DCA_Hybrid_Model_LS_TS_Performance.png'
    else:
        r_squared, rmse, nrmse, pearson_r, spearman_rho = get_performances(
            y_true=y_true, y_pred=y_pred
        )
        ax.scatter(
            y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.7,
            label=r'$R^2$' + f' = {r_squared:.3f}' + f'\nRMSE = {rmse:.3f}' + f'\nNRMSE = {nrmse:.3f}' +
                  f'\nPearson\'s ' + r'$r$'+f' = {pearson_r:.3f}' + f'\nSpearman\'s ' +
                  fr'$\rho$ = {spearman_rho:.3f}' + '\n' + fr'($N$ = {len(y_true)})'
        )
        file_name = name + 'ML_Model_LS_TS_Performance.png'
        x = np.linspace(min(y_pred), max(y_pred), 100)
        ax.plot(x, x, color='black', linewidth=0.25)  # plot diagonal line
    ax.legend(prop={'size': 8})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    logger.info('Plotting...')
    if label:
        logger.info('Adjusting variant labels for plotting can take some '
                    'time (the limit for labeling is 150 data points)...')
        if len(y_true) < 150:
            texts = [ax.text(y_true[i], y_pred[i], txt, fontsize=4)
                     for i, txt in enumerate(variants)]
            adjust_text(
                texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5, lim=250)
        else:
            logger.info("Terminating label process. Too many variants "
                        "(> 150) for plotting (labels would overlap).")
    # Uncomment for renaming new plots
    # i = 1
    # while os.path.isfile(file_name):
    #     i += 1  # iterate until finding an unused file name
    #     file_name = f'DCA_Hybrid_Model_LS_TS_Performance({i}).png'
    plt.savefig(file_name, dpi=500)
