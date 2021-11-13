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
from sklearn.linear_model import LinearRegression  # Ordinary least squares Linear Regression, likely bad CV performance
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, LassoLars

from pypef.api.encoding import aa_index_list, AAIndexEncoding
from pypef.api.cv_regression_options import pls_cv_regressor

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def tqdm_runtime(bool_, tqdm_input):
    if type(bool_) is bool:
        if bool_:
            return tqdm(tqdm_input)
        else:
            return tqdm_input
    else:
        raise SystemError("Please give Boolean (True/False) for tqdm_ parameter.")


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
