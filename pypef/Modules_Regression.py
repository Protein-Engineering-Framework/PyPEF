#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
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
Main modules for regression/ML including featurization,
validation, tuning, prediction, and plotting routines.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # no plt.show(), just save plot
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm  # progress bars
# import adjust_text  # only locally imported for labeled validation plots and in silico directed evolution
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  # default: refit=True

# import regression models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import warnings
# ignoring warnings of PLS regression using n_components
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


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


class AAIndex:
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


def get_sequences(fasta, mult_path=None, prediction=False):
    """
    "Get_Sequences" reads (learning and validation).fasta format files and extracts the name,
    the target value and the sequence of the peptide. See example directory for required fasta file format.
    Make sure every marker (> and ;) is seperated by an space ' ' from the value/ name.
    """
    if mult_path is not None:
        os.chdir(mult_path)

    sequences = []
    values = []
    names_of_mutations = []

    with open(fasta, 'r') as f:
        for line in f:
            if '>' in line:
                words = line.split()
                names_of_mutations.append(words[1])
                # words[1] is appended so make sure there is a space in between > and the name!

            elif '#' in line:
                pass  # are Comments

            elif ';' in line:
                words = line.split()
                values.append(float(words[1]))
                # words[1] is appended so make sure there is a space in between ; and the value!

            else:
                try:
                    words = line.split()
                    sequences.append(words[0])
                except IndexError:
                    raise IndexError("Learning or Validation sets (.fasta) likely have emtpy lines at end of file")

    # Check consistency
    if not prediction:
        if len(sequences) != len(values):
            print('Error: Number of sequences does not fit with number of target values!')
            print('Number of sequences: {}, Number of target values: {}.'.format(str(len(sequences)), str(len(values))))
            sys.exit()

    return sequences, names_of_mutations, values


class XY:
    """
    converts the string sequence into a list of numericals using the AAindex translation library,
    Fourier transforming the numerical array that was translated by get_numerical_sequence --> do_fourier,
    computing the input matrices X and Y for the PLS regressor (get_x_and_y). Returns FFT-ed arrays (x),
    labels array (y), and raw_encoded sequences arrays (raw_numerical_sequences)
    """
    def __init__(self, aaindex_file, fasta_file, mult_path=None, prediction=False):
        aaidx = AAIndex(aaindex_file)
        self.dictionary = aaidx.encoding_dictionary()
        self.sequences, self.names, self.values = get_sequences(fasta_file, mult_path, prediction)

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
        y = self.values                             # Fitness values (sequence labels)

        return x, y, raw_numerical_seq


def get_r2(x_learn, x_valid, y_learn, y_valid, regressor='pls'):
    """
    The function Get_R2 takes features and labels from the learning and validation set.

    When using 'pls' as regressor, the MSE is calculated for all LOOCV sets for predicted vs true labels
    (mse = mean_squared_error(y_test_loo, y_pred_loo) for a fixed number of components for PLS regression.
    In the next iteration, the number of components is increased by 1 (number_of_components += 1)
    and the MSE is calculated for this regressor. The loop breaks if i > 9.
    Finally, the model of the single AAindex model with the lowest MSE is chosen.

    When using other regressors the parameters are tuned using GridSearchCV.

    This function returnes performance (R2, (N)RMSE, Pearson's r) and model parameters.
    """
    regressor = regressor.lower()
    mean_squared_error_list = []

    if regressor == 'pls':
        # PLS regression with LOOCV n_components tuning as described by Cadet et al.
        # https://doi.org/10.1186/s12859-018-2407-8
        # https://doi.org/10.1038/s41598-018-35033-y
        # Hyperparameter (N component) tuning of PLS regressor
        for n_comp in range(1, 10):  # n_comp = 1, 2,..., 9
            pls = PLSRegression(n_components=n_comp)
            loo = LeaveOneOut()

            y_pred_loo = []
            y_test_loo = []

            for train, test in loo.split(x_learn):
                x_learn_loo = []
                y_learn_loo = []
                x_test_loo = []

                for j in train:
                    x_learn_loo.append(x_learn[j])
                    y_learn_loo.append(y_learn[j])

                for k in test:
                    x_test_loo.append(x_learn[k])
                    y_test_loo.append(y_learn[k])

                pls.fit(x_learn_loo, y_learn_loo)
                y_pred_loo.append(pls.predict(x_test_loo)[0][0])

            mse = mean_squared_error(y_test_loo, y_pred_loo)

            mean_squared_error_list.append(mse)

        mean_squared_error_list = np.array(mean_squared_error_list)
        # idx = np.where(...) finds best number of components
        idx = np.where(mean_squared_error_list == np.min(mean_squared_error_list))[0][0] + 1

        # Model is fitted with best n_components (lowest MSE)
        best_params = {'n_components': idx}
        regressor_ = PLSRegression(n_components=best_params.get('n_components'))

    # other regression options (CV tuning)
    elif regressor == 'pls_cv':
        params = {'n_components': list(np.arange(1, 10))}  # n_comp = 1, 2,..., 9
        regressor_ = GridSearchCV(PLSRegression(), param_grid=params, iid=False, cv=5)  # iid in future
                                                                                        # versions redundant
    elif regressor == 'rf':
        params = {                 # similar parameter grid as Xu et al., https://doi.org/10.1021/acs.jcim.0c00073
            'random_state': [42],  # state determined
            'n_estimators': [100, 250, 500, 1000],  # number of individual decision trees in the forest
            'max_features': ['auto', 'sqrt', 'log2']  # “auto” -> max_features=n_features,
            # “sqrt” -> max_features=sqrt(n_features) “log2” -> max_features=log2(n_features)
        }
        regressor_ = GridSearchCV(RandomForestRegressor(), param_grid=params, iid=False, cv=5)

    elif regressor == 'svr':
        params = {                      # similar parameter grid as Xu et al.
            'C': [2 ** 0, 2 ** 2, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12],  # Regularization parameter
            'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]  # often 1 / n_features or 1 / (n_features * X.var())
        }
        regressor_ = GridSearchCV(SVR(), param_grid=params, iid=False, cv=5)

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
        regressor_ = GridSearchCV(MLPRegressor(), param_grid=params, iid=False, cv=5)

    else:
        raise SystemError("Did not find specified regression model as valid option. See '--help' for valid "
                          "regression model options.")

    regressor_.fit(x_learn, y_learn)  # fit model

    if regressor != 'pls':      # take best parameters for the regressor and the AAindex
        best_params = regressor_.best_params_

    y_pred = []
    try:
        for y_p in regressor_.predict(x_valid):  # predict validation entries with fitted model
            y_pred.append(float(y_p))
    except ValueError:
        raise ValueError("Above error message exception indicates that your validation set may be empty.")

    r2 = r2_score(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    nrmse = rmse / np.std(y_valid, ddof=1)
    # ranks for Spearman's rank correlation
    y_val_rank = np.array(y_valid).argsort().argsort()
    y_pred_rank = np.array(y_pred).argsort().argsort()
    with warnings.catch_warnings():  # catching RunTime warning when there's no variance in an array, e.g. [2, 2, 2, 2]
        warnings.simplefilter("ignore")  # which would mean divide by zero
        pearson_r = np.corrcoef(y_valid, y_pred)[0][1]
        spearman_rho = np.corrcoef(y_val_rank, y_pred_rank)[0][1]

    return r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params


def r2_list(learning_set, validation_set, regressor='pls', no_fft=False, sort='1'):
    """
    returns the sorted list of all the model parameters and
    the performance values (r2 etc.) from function get_r2.
    """
    aa_indices = [file for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]

    aaindex_r2_list = []
    for index, aaindex in enumerate(tqdm(aa_indices)):
        xy_learn = XY(full_path(aaindex), learning_set)
        if not no_fft:  # X is FFT-ed of encoded alphabetical sequence
            x_learn, y_learn, _ = xy_learn.get_x_and_y()
        else:               # X is raw encoded of alphabetical sequence
            _, y_learn, x_learn = xy_learn.get_x_and_y()

        # If x_learn (or y_learn) is an empty array, the sequence could not be encoded,
        # because of NoneType value. -> Skip
        if len(x_learn) != 0:
            xy_test = XY(full_path(aaindex), validation_set)
            if not no_fft:  # X is FFT-ed of the encoded alphabetical sequence
                x_test, y_test, _ = xy_test.get_x_and_y()
            else:               # X is the raw encoded of alphabetical sequence
                _, y_test, x_test = xy_test.get_x_and_y()
            r2, rmse, nrmse, pearson_r, spearman_rho, regression_model, params = get_r2(x_learn, x_test, y_learn,
                                                                                        y_test, regressor)
            aaindex_r2_list.append([aaindex, r2, rmse, nrmse, pearson_r, spearman_rho, regression_model, params])

    try:
        sort = int(sort)
        if sort == 2 or sort == 3:
            aaindex_r2_list.sort(key=lambda x: x[sort])
        else:
            aaindex_r2_list.sort(key=lambda x: x[sort], reverse=True)

    except ValueError:
        raise ValueError("Choose between options 1 to 5 (R2, RMSE, NRMSE, Pearson's r, Spearman's rho.")

    return aaindex_r2_list


def formatted_output(aaindex_r2_list, no_fft=False, minimum_r2=0.0):
    """
    takes the sorted list from function r2_list and writes the model names with an R2 ≥ 0
    as well as the corresponding parameters for each model so that the user gets
    a list (Model_Results.txt) of the top ranking models for the given validation set.
    """

    index, value, value2, value3, value4, value5, regression_model, params = [], [], [], [], [], [], [], []

    for (idx, val, val2, val3, val4, val5, r_m, pam) in aaindex_r2_list:
        if val >= minimum_r2:
            index.append(idx[:-4])
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

    return ()


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
            y_pred_test = regressor_.predict(x_test)

            for values in y_pred_test:
                y_predicted_total.append(float(values))
        except UserWarning:
            continue

    return y_test_total, y_predicted_total


def save_model(path, aaindex_r2_list, learning_set, validation_set, threshold=5, regressor='pls',
               no_fft=False, train_on_all=False):
    """
    Function Save_Model saves the best -s THRESHOLD models as 'Pickle' files (pickle.dump),
    which can be loaded again for doing predictions. Also, in Save_Model included is the def cross_validation
    -based computing of the k-fold CV performance of the n component-optimized model on all data
    (learning + validation set); by default  k  is 5 (n_samples = 5).
    Plots of the CV performance for the t best models are stored inside the folder CV_performance.
    """
    regressor = regressor.lower()
    try:
        os.mkdir('CV_performance')
    except FileExistsError:
        pass
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass

    try:
        os.remove('CV_performance/_CV_Results.txt')
    except FileNotFoundError:
        pass
    file = open('CV_performance/_CV_Results.txt', 'w')
    file.write('5-fold cross-validated performance of top models for validation set across all data.\n\n')
    if no_fft:
        file.write("No FFT used in this model construction, performance represents"
                   " model accuracies on raw encoded sequence data.\n\n")
    file.close()

    for t in range(threshold):
        try:
            idx = aaindex_r2_list[t][0]
            parameter = aaindex_r2_list[t][7]

            # Estimating the CV performance of the n_component-fitted model on all data
            xy_learn = XY(full_path(idx), learning_set)
            xy_test = XY(full_path(idx), validation_set)
            if no_fft is False:
                x_test, y_test, _ = xy_test.get_x_and_y()
                x_learn, y_learn, _ = xy_learn.get_x_and_y()
            else:
                _, y_test, x_test = xy_test.get_x_and_y()
                _, y_learn, x_learn = xy_learn.get_x_and_y()

            x = np.concatenate([x_learn, x_test])
            y = np.concatenate([y_learn, y_test])

            if regressor == 'pls' or regressor == 'pls_cv':
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

            else:
                raise SystemError("Did not find specified regression model as valid option. "
                                  "See '--help' for valid regression model options.")

            # perform 5-fold cross-validation on all data (on X and Y)
            n_samples = 5
            y_test_total, y_predicted_total = cross_validation(x, y, regressor_, n_samples)

            r_squared = r2_score(y_test_total, y_predicted_total)
            rmse = np.sqrt(mean_squared_error(y_test_total, y_predicted_total))
            stddev = np.std(y_test_total, ddof=1)
            nrmse = rmse / stddev
            pearson_r = np.corrcoef(y_test_total, y_predicted_total)[0][1]
            # ranks for Spearman correlation
            y_test_total_rank = np.array(y_test_total).argsort().argsort()
            y_predicted_total_rank = np.array(y_predicted_total).argsort().argsort()
            spearman_rho = np.corrcoef(y_test_total_rank, y_predicted_total_rank)[0][1]

            with open('CV_performance/_CV_Results.txt', 'a') as f:
                f.write('Regression type: {}; Parameter: {}; Encoding index: {}\n'.format(
                    regressor.upper(), parameter, idx[:-4]))
                f.write('R2 = {:.5f}; RMSE = {:.5f}; NRMSE = {:.5f}; Pearson\'s r = {:.5f};'
                        ' Spearman\'s rho = {:.5f}\n\n'.format(r_squared, rmse, nrmse, pearson_r, spearman_rho))

            figure, ax = plt.subplots()
            ax.scatter(y_test_total, y_predicted_total, marker='o', s=20, linewidths=0.5, edgecolor='black')
            ax.plot([min(y_test_total) - 1, max(y_test_total) + 1],
                    [min(y_predicted_total) - 1, max(y_predicted_total) + 1], 'k', lw=2)
            ax.legend([
                '$R^2$ = {}\nRMSE = {}\nNRMSE = {}\nPearson\'s $r$ = {}\nSpearman\'s '.format(
                    round(r_squared, 3), round(rmse, 3), round(nrmse, 3), round(pearson_r, 3))
                + r'$\rho$ = {}'.format(str(round(spearman_rho, 3)))
            ])
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            plt.savefig('CV_performance/' + idx[:-4] + '_' + str(n_samples) + '-fold-CV.png', dpi=250)
            plt.close('all')

            if train_on_all:
                # fit on all available data (learning + validation set; FFT or noFFT is defined already above)
                regressor_.fit(x, y)
            else:
                # fit (only) on full learning set (FFT or noFFT is defined already above)
                regressor_.fit(x_learn, y_learn)

            file = open(os.path.join(path, 'Pickles/'+idx[:-4]), 'wb')
            pickle.dump(regressor_, file)
            file.close()

        except IndexError:
            break

    return ()


def predict(path, prediction_set, model, mult_path=None, no_fft=False, print_matrix=False):
    """
    The function Predict is used to perform predictions.
    Saved pickle files of models will be loaded again (mod = pickle.load(file))
    and used for predicting the label y (y = mod.predict(x)) of sequences given in the Prediction_Set.fasta.
    """
    aaidx = full_path(str(model) + '.txt')
    xy = XY(aaidx, prediction_set, mult_path, prediction=True)
    x, _, x_raw = xy.get_x_and_y()

    file = open(os.path.join(path, 'Pickles/'+str(model)), 'rb')
    mod = pickle.load(file)
    file.close()

    try:
        y_list = []
        if no_fft is False:
            y = mod.predict(x)
            for y_ in y:
                # just make sure predicted Y is nested (list of list) [[Y_1], [Y_2], ..., [Y_N]]
                y_list.append([float(y_)])
        else:
            y = mod.predict(x_raw)
            for y_ in y:
                # just make sure predicted Y is nested (list of list) [[Y_1], [Y_2], ..., [Y_N]]
                y_list.append([float(y_)])

    except ValueError:
        raise ValueError("You likely tried to predict using a model with (or without) FFT featurization ('--nofft')"
                         " while the model was trained without (or with) FFT featurization. Check the Model_Results.txt"
                         " line 1, if the models were trained using FFT.")

    _, names_of_mutations, _ = get_sequences(prediction_set, mult_path, prediction=True)

    predictions = [(y_list[i][0], names_of_mutations[i]) for i in range(len(y_list))]

    # Pay attention if more negative values would define a better variant --> use negative flag
    predictions.sort()
    predictions.reverse()
    # if predictions array too large?  if Mult_Path is not None: predictions = predictions[:100000]

    # Print FFT-ed and raw sequence vectors for directed evolution if desired
    if print_matrix:
        print('X (FFT):\n{} len(X_raw): {}\nX_raw (noFFT):\n{} len(X): {}\n(Predicted value, Variant): {}\n\n'.format(
            x, len(x[0]), x_raw, len(x_raw[0]), predictions))

    return predictions


def predictions_out(predictions, model, prediction_set):
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
    with open('Predictions_' + str(model) + '_' + str(prediction_set)[:-6] + '.txt', 'w') as f:
        f.write("".join(caption.ljust(col_width) for caption in head) + '\n')
        f.write(len(head)*col_width*'-' + '\n')
        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')

    return ()


def plot(path, fasta_file, model, label, color, y_wt, no_fft=False):
    """
    Function Plot is used to make plots of the validation process and
    shows predicted (Y_pred) vs. measured/"true" (Y_true) protein fitness and
    calculates the corresponding model performance (R2, (N)RMSE, Pearson's r).
    Also allows colored version plotting to classify predictions in true or
    false positive or negative predictions.
    """
    aaidx = full_path(str(model) + '.txt')
    xy = XY(aaidx, fasta_file, prediction=False)
    x, y_true, x_raw = xy.get_x_and_y()

    try:
        file = open(os.path.join(path, 'Pickles/'+str(model)), 'rb')
        mod = pickle.load(file)
        file.close()

        y_pred = []

        try:
            # predicting (again) with (re-)loaded model (that was trained on training or full data)
            if no_fft is False:
                y_pred_ = mod.predict(x)
            else:
                y_pred_ = mod.predict(x_raw)
        except ValueError:
            raise ValueError("You likely tried to plot a validation set with (or without) FFT featurization ('--nofft')"
                             " while the model was trained without (or with) FFT featurization. Check the"
                             " Model_Results.txt line 1, if the models were trained using FFT.")

        for y_p in y_pred_:
            y_pred.append(float(y_p))

        _, names_of_mutations, _ = get_sequences(fasta_file)

        r_squared = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        stddev = np.std(y_true, ddof=1)
        nrmse = rmse / stddev
        pearson_r = np.corrcoef(y_true, y_pred)[0][1]
        # ranks for Spearman
        y_true_total_rank = np.array(y_true).argsort().argsort()
        y_pred_total_rank = np.array(y_pred).argsort().argsort()
        spearman_rho = np.corrcoef(y_true_total_rank, y_pred_total_rank)[0][1]
        legend = '$R^2$ = {}\nRMSE = {}\nNRMSE = {}\nPearson\'s $r$ = {}\nSpearman\'s '.format(
            round(r_squared, 3), round(rmse, 3), round(nrmse, 3), round(pearson_r, 3)) \
            + r'$\rho$ = {}'.format(round(spearman_rho, 3))
        x = np.linspace(min(y_pred) - 1, max(y_pred) + 1, 100)

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, label=legend, marker='o', s=20, linewidths=0.5, edgecolor='black')
        ax.plot(x, x, color='black', linewidth=0.5)  # plot diagonal line

        if label is not False:
            from adjustText import adjust_text
            texts = [ax.text(y_true[i], y_pred[i], txt, fontsize=4) for i, txt in enumerate(names_of_mutations)]
            adjust_text(texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5)

        if color is not False:
            try:
                y_wt = float(y_wt)
            except TypeError:
                raise TypeError('Needs label value of WT (y_WT) when making color plot (e.g. --color --ywt 1.0)')
            if y_wt == 0:
                y_wt = 1E-9  # choose a value close to zero
            true_v, true_p, false_v, false_p = [], [], [], []
            for i, v in enumerate(y_true):
                if y_true[i] / y_wt >= 1 and y_pred[i] / y_wt >= 1:
                    true_v.append(y_true[i]), true_p.append(float(y_pred[i]))
                elif y_true[i] / y_wt < 1 and y_pred[i] / y_wt < 1:
                    true_v.append(y_true[i]), true_p.append(float(y_pred[i]))
                else:
                    false_v.append(y_true[i]), false_p.append(float(y_pred[i]))
            try:
                ax.scatter(true_v, true_p, color='tab:blue', marker='o', s=20, linewidths=0.5, edgecolor='black')
            except IndexError:
                pass
            try:
                ax.scatter(false_v, false_p, color='tab:red', marker='o', s=20, linewidths=0.5, edgecolor='black')
            except IndexError:
                pass

            if (y_wt - min(y_true)) < (max(y_true) - y_wt):
                limit_y_true = float(max(y_true) - y_wt)
            else:
                limit_y_true = float(y_wt - min(y_true))
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
            plt.plot(crossline, crossline, color='black', linewidth=0.5)

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
    except FileNotFoundError:
        raise FileNotFoundError("Did not find specified model: {}. You can define the threshold of models to be saved;"
                                " e.g. with run_pypef.py run -l LS.fasta -v VS.fasta -t 10.".format(str(model)))

    return ()
