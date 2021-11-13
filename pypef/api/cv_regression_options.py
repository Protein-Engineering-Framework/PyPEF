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


"""
k-fold CV regression options for tuning some Scikit-learn
BaseEstimator regression models using GridSearchCV
"""

from sklearn.model_selection import GridSearchCV, KFold  # default: refit=True
from sklearn.linear_model import LinearRegression  # Ordinary least squares Linear Regression, likely bad CV performance
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, LassoLars


# diverse regression options
def pls_cv_regressor():
    params = {'n_components': list(range(1, 10))}  # n_comp = 1, 2,..., 9
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
        'random_state': [42]  # state determined
    }
    regressor_ = GridSearchCV(MLPRegressor(), param_grid=params, cv=5)
    return regressor_


def ridge_cv_regressor():
    # Performs L2 regularization, i.e., adds penalty equivalent to square of the magnitude of coefficients
    # Majorly used to prevent overfitting, since it includes all the features
    # in case of exorbitantly high features, it will pose computational challenges.
    params = {
        # alpha = 0 is equivalent to an ordinary least square Regression
        # higher values of alpha reduce overfitting, significantly high values can
        # cause underfitting as well (e.g., alpha = 5)
        'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 0.001, 0.1, 0.3, 0.5, 1.0, 10.0]
    }
    regressor_ = GridSearchCV(Ridge(), param_grid=params, cv=5)
    return regressor_


def lasso_lars_cv_regressor():
    # Lasso model fit with Least Angle Regression a.k.a. Lars.
    # Performs L1 regularization, i.e., adds penalty equivalent to absolute value of the magnitude of coefficients
    # Provides sparse solutions: computationally efficient as features with zero coefficients can be ignored
    params = {
        # alpha = 0 is equivalent to an ordinary least square Regression
        'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 0.001, 0.1, 0.3, 0.5, 1.0, 10.0]
    }
    regressor_ = GridSearchCV(LassoLars(), param_grid=params, cv=5)
    return regressor_