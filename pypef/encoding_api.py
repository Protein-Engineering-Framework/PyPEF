#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – an Integrated Framework for Data-driven Protein Engineering
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold  # default: refit=True
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import warnings
# ignoring warnings of PLS regression using n_components
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def tqdm_runtime(bool_, tqdm_input):
    if type(bool_) is bool:
        if bool_:
            return tqdm(tqdm_input)
        else:
            return tqdm_input
    else:
        raise SystemError("Please give Bool (True/False) for tqdm_ parameter.")


def full_path(filename):
    """
    returns the path of an index inside the folder /AAindex/,
    e.g. path/to/AAindex/FAUJ880109.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(modules_path, 'AAindex/' + filename)


def path_aaindex_dir():
    """
    returns the path to the /AAindex folder, e.g. path/to/AAindex/.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AAindex')


def aa_index_list():
    return [full_path(file) for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]


def pls_cv_regressor():
    params = {'n_components': list(np.arange(1, 10))}  # n_comp = 1, 2,..., 9
    regressor_ = GridSearchCV(PLSRegression(), param_grid=params, cv=5)
    return regressor_


def rf_cv_regressor():
    params = {  # similar parameter grid as Xu et al., https://doi.org/10.1021/acs.jcim.0c00073
        'random_state': [42],  # state determined
        'n_estimators': [100, 250, 500, 1000],  # number of individual decision trees in the forest
        'max_features': ['auto', 'sqrt', 'log2']  # “auto” -> max_features=n_features,
        # “sqrt” -> max_features=sqrt(n_features) “log2” -> max_features=log2(n_features)
    }
    regressor_ = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5)
    return regressor_


def svr_cv_regressor():
    params = {  # similar parameter grid as Xu et al.
        'C': [2 ** 0, 2 ** 2, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12],  # Regularization parameter
        'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]  # often 1 / n_features or 1 / (n_featrues * X.var())
    }
    regressor_ = GridSearchCV(SVR(), param_grid=params, cv=5)
    return regressor_


def mlp_cv_regressor():
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
    return regressor_


def cross_validation(x, y, regressor_, n_samples=5):
    # perform k-fold cross-validation on all data
    # k = Number of splits, change for changing k in k-fold split-up, default=5
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
            regressor_.fit(x_train, y_train)  # Fitting on a random subset for Final_Model
            # (and on a subset subset for Learning_Model)
            # Predictions for samples in the test_set during that iteration
            y_pred_test = regressor_.predict(x[test_index])

            for value in y_pred_test:
                y_predicted_total.append(float(value))
        except UserWarning:
            continue

    return y_test_total, y_predicted_total


class AAIndexDict:
    """
    gets all the information that are given in each AAindex file.
    For the program routine it provides the library to enable translation
    of the alphabetical amino acid sequence to an array of numericals.
    """
    def __init__(self, filename):
        self.File = filename
        self.Accession_Number = None
        self.Data_Description = None
        self.PMID = None
        self.Authors = None
        self.Title_Of_Article = None
        self.Journal_Reference = None

    def general_information(self):
        """
        Gets and allocates general information based on the AAindex file
        format defined by file sections 'H', 'D', 'E', 'A', 'T', 'J'
        """
        with open(self.File, 'r') as f:
            for line in f:
                # try/ except "removes" empty lines.
                try:
                    words = line.split()
                    id_letter = words[0]
                except IndexError:
                    break

                # Extract some general information about AAindex file.
                if id_letter == 'H':
                    self.Accession_Number = words[1]
                elif id_letter == 'D':
                    self.Data_Description = words[1]
                elif id_letter == 'E':
                    self.PMID = words[1:]
                elif id_letter == 'A':
                    self.Authors = ' '.join(words[1:])
                elif id_letter == 'T':
                    self.Title_Of_Article = ' '.join(words[1:])
                elif id_letter == 'J':
                    self.Journal_Reference = ' '.join(words[1:])

    def encoding_dictionary(self):
        """
        Get numerical values of AAindex for each amino acid
        """
        with open(self.File, 'r') as f:
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


class AAIndexEncoding:  # was class XY originally
    """
    converts the string sequence into a list of numericals using the AAindex translation library,
    Fourier transforming the numerical array that was translated by get_numerical_sequence --> do_fourier,
    computing the input matrices X and Y for the PLS regressor (get_x_and_y). Returns FFT-ed arrays (x),
    labels array (y), and raw_encoded sequences arrays (raw_numerical_sequences)
    """
    def __init__(self, aaindex_file, sequences):  # , value, mult_path=None, prediction=False):
        aaidx = AAIndexDict(aaindex_file)
        self.dictionary = aaidx.encoding_dictionary()
        # self.name, self.value = name, value  #ADD?
        self.sequences = sequences

    def get_numerical_sequence(self, sequence):
        return np.array([self.dictionary[aminoacid] for aminoacid in sequence])

    @staticmethod
    def do_fourier(sequence):
        """
        This static function does the Fast Fourier Transform. Since the condition

                    len(Array) = 2^k -> k = log_2(len(Array))
                    k in N

        must be satisfied, the array must be reshaped (zero padding) if k is no integer value.
        The verbose parameter prints also the real and imaginary part separately.
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

    def get_x_and_y(self):
        """
        getting the input matrices X (FFT amplitudes) and Y (variant labels)
        """
        frequencies = []
        amplitudes = []
        raw_numerical_seq = []

        for sequence in self.sequences:
            num = self.get_numerical_sequence(sequence)

            # There may be amino acids without a value in AAindex.
            # Skip these Indices.
            if None in num:
                break

            # Numerical sequence gets expended by zeros so that also different lengths of sequences
            # can be processed using '--nofft' option
            k = np.log2(len(num))
            if abs(int(k) - k) > 1e-8:  # check if length of array fulfills previous equation
                num_appended = np.append(num, np.zeros(pow(2, (int(k) + 1)) - len(num)))  # reshape array
            else:
                num_appended = num

            amplitudes_, frequencies_ = self.do_fourier(num)

            # Fourier spectra are mirrored at frequency = 0.5. No more information at higher frequencies.
            half = len(frequencies_) // 2  # // for integer division
            frequencies.append(frequencies_[:half])
            amplitudes.append(amplitudes_[:half])    # FFT-ed encoded amino acid sequences
            raw_numerical_seq.append(num_appended)  # Raw encoded amino acid sequences

        amplitudes = np.array(amplitudes)
        # frequencies = np.array(frequencies)  # not used
        raw_numerical_seq = np.array(raw_numerical_seq)

        x = amplitudes

        return x, raw_numerical_seq


def get_performance(
        seq_learn, y_learn, seq_valid, y_valid, fft=True,
        regressor=pls_cv_regressor(), tqdm_=False, kfold_cv_on_all=False,
        finally_train_on_all=False, save_models=True, sort='1'
):

    performances = []
    cv_performances = []

    for aaindex in tqdm_runtime(tqdm_, aa_index_list()):

        if fft:
            x_learn, _ = AAIndexEncoding(aaindex, seq_learn).get_x_and_y()
            x_valid, _ = AAIndexEncoding(aaindex, seq_valid).get_x_and_y()
        else:
            _, x_learn = AAIndexEncoding(aaindex, seq_learn).get_x_and_y()
            _, x_valid = AAIndexEncoding(aaindex, seq_valid).get_x_and_y()

        try:
            regressor.fit(x_learn, y_learn)
        except ValueError:
            continue
        # also here try-except as only CrossCVRegressor has .best_params_
        try:
            best_params = regressor.best_params_
        except AttributeError:
            best_params = regressor.get_params()

        y_pred, y_pred_learn = [], []
        for y_p in regressor.predict(x_valid):  # predict validation entries with fitted model
            y_pred.append(float(y_p))
        for y_p in regressor.predict(x_learn):  # predict validation entries with fitted model
            y_pred_learn.append(float(y_p))

        # order: (y_true, y_pred)
        r2 = r2_score(y_valid, y_pred)
        r2_train = r2_score(y_learn, y_pred_learn)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_train = np.sqrt(mean_squared_error(y_learn, y_pred_learn))
        nrmse = rmse / np.std(y_valid, ddof=1)
        nrmse_train = rmse_train / np.std(y_valid, ddof=1)

        with warnings.catch_warnings():  # catching RunTime warning when there's
            # no variance in an array, e.g. [2, 2, 2, 2]
            warnings.simplefilter("ignore")  # which would mean divide by zero
            pearson_r = np.corrcoef(y_valid, y_pred)[0][1]
            pearson_r_train = np.corrcoef(y_learn, y_pred_learn)[0][1]
            spearman_rho = spearmanr(y_valid, y_pred)[0]
            spearman_rho_train = spearmanr(y_learn, y_pred_learn)[0]
        try:
            best_estimator = regressor.estimator
        except AttributeError:
            best_estimator = regressor
        best_estimator = str(best_estimator.set_params(**best_params))

        performances.append([
            aaindex[-14:-4], r2, rmse, nrmse, pearson_r,
            spearman_rho, best_params, best_estimator, aaindex,
            r2_train, rmse_train, nrmse_train, pearson_r_train, spearman_rho_train,
            y_pred_learn, y_pred
        ])

        if kfold_cv_on_all:
            try:
                kfold_cv_on_all = int(kfold_cv_on_all)
                n_samples = kfold_cv_on_all
                if n_samples == 1:  # min 2 splits
                    raise SystemError("Increase kFold splits at least to the minimum (2 splits).")
            except ValueError:
                n_samples = 5
            x_all = np.concatenate((x_learn, x_valid), axis=0)
            y_all = np.concatenate((y_learn, y_valid), axis=0)

            # perform k-fold cross-validation on all data (on X and Y)
            y_test_total, y_predicted_total = cross_validation(x_all, y_all, regressor, n_samples)

            r2 = r2_score(y_test_total, y_predicted_total)
            rmse = np.sqrt(mean_squared_error(y_test_total, y_predicted_total))
            stddev = np.std(y_test_total, ddof=1)
            nrmse = rmse / stddev
            pearson_r = np.corrcoef(y_test_total, y_predicted_total)[0][1]
            spearman_rho = spearmanr(y_test_total, y_predicted_total)[0]
            cv_performances.append([aaindex[-14:-4], r2, rmse, nrmse, pearson_r, spearman_rho])

    try:
        sort = int(sort)
        if sort == 2 or sort == 3:
            performances.sort(key=lambda x: x[sort])
            cv_performances.sort(key=lambda x: x[sort])
        else:
            performances.sort(key=lambda x: x[sort], reverse=True)
            cv_performances.sort(key=lambda x: x[sort], reverse=True)
    except ValueError:
        raise ValueError("Choose between options 1 to 5 (R2, RMSE, NRMSE, Pearson's r, Spearman's rho.")

    if save_models:
        try:
            save_model = int(save_models)
            threshold = save_model
        except ValueError:
            threshold = 5
        for t in range(threshold):
            try:
                idx = performances[t][0]
                estimator = eval(performances[t][7])
                idx_path = performances[t][8]

                if fft:
                    x_learn, _ = AAIndexEncoding(idx_path, seq_learn).get_x_and_y()
                    x_valid, _ = AAIndexEncoding(idx_path, seq_valid).get_x_and_y()
                else:
                    _, x_learn = AAIndexEncoding(idx_path, seq_learn).get_x_and_y()
                    _, x_valid = AAIndexEncoding(idx_path, seq_valid).get_x_and_y()

                if finally_train_on_all:
                    x_all = np.concatenate((x_learn, x_valid), axis=0)
                    y_all = np.concatenate((y_learn, y_valid), axis=0)
                    estimator.fit(x_all, y_all)
                else:
                    estimator.fit(x_learn, y_learn)

                try:
                    os.mkdir('Models')
                except FileExistsError:
                    pass

                pickle.dump(estimator, open('Models/' + idx + '.sav', 'wb'))

            except IndexError:
                break

    return performances, cv_performances
