#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @authors: Niklas Siedhoff, Alexander-Maurice Illig
# @contact: <niklas.siedhoff@rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF
# Licensed under Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099

import warnings
import numpy as np
from scipy import stats
from sklearnex import patch_sklearn
patch_sklearn(verbose=False)
from sklearn.metrics import (
    root_mean_squared_error, r2_score,
    precision_score, accuracy_score, recall_score, 
    balanced_accuracy_score, f1_score, matthews_corrcoef,
    average_precision_score, roc_curve, auc
)
from sklearn.preprocessing import normalize, Binarizer


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
    rmse = root_mean_squared_error(y_true, y_pred)
    stddev = np.std(y_true, ddof=1)
    nrmse = rmse / stddev
    with warnings.catch_warnings():  # catching RunTime warning when there's no variance in an array,
        warnings.simplefilter("ignore")  # e.g. [2, 2, 2, 2] which would mean divide by zero
        pearson_r = np.corrcoef(y_true, y_pred)[0][1]
        spearman_rho = stats.spearmanr(y_true, y_pred)[0]
    return r_squared, rmse, nrmse, pearson_r, spearman_rho


def setpositives_bin(y_true, y_pred, positives_percentage: float = 0.1):
    """
    1 - Given a percentage of positives
    2 - sort the values descending
    3 - set the top as 1, rest as 0
    4 - return array of 0s and 1s in the initial order

    Using sklearn.preprocessing.Binarizer for binarization

    `y_pred_proba` is the imaginary probability that a variant is improved
    and here simply represents the normalized predicted score ([0, 1]).
    This pseudo-probability is required to plot a ROC-AUC curve.
    """
    assert len(y_true) == len(y_pred)
    n_positives = max(1, int(positives_percentage * len(y_true)))
    y_true_sorted = np.sort(y_true)[::-1]
    y_threshold = y_true_sorted[n_positives]
    y_true_binned = Binarizer(threshold=y_threshold).transform(np.atleast_2d(y_true))
    y_pred_sorted = np.sort(y_pred)[::-1]
    y_pred_threshold = y_pred_sorted[n_positives]
    y_pred_binned = Binarizer(threshold=y_pred_threshold).transform(np.atleast_2d(y_pred))
    y_pred_proba = normalize(np.atleast_2d(y_pred))
    return y_true_binned[0], y_pred_binned[0], y_pred_proba[0]


def binary_classification_metrics(y_true_binned, y_pred_binned, y_pred_proba):
    """
    returns 
    --------
    prec:float, acc: float, bacc: float, rec: float, f1: float, 
    mcc: float, auroc: float, aps: float, fpr: list, tpr: list
    """
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://doi.org/10.1186/s12864-019-6413-7
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    # https://www.analyticsvidhya.com/blog/2020/09/precision-recall-machine-learning/
    fpr, tpr, _ = roc_curve(y_true_binned, y_pred_proba)
    prec = precision_score(y_true_binned, y_pred_binned)
    acc = accuracy_score(y_true_binned, y_pred_binned)
    bacc = balanced_accuracy_score(y_true_binned, y_pred_binned)
    rec = recall_score(y_true_binned, y_pred_binned)
    f1 = f1_score(y_true_binned, y_pred_binned)
    mcc = matthews_corrcoef(y_true_binned, y_pred_binned)
    auroc = auc(fpr, tpr)
    aps = average_precision_score(y_true_binned, y_pred_proba)
    return prec, acc, bacc, rec, f1, mcc, auroc, aps, fpr, tpr


def get_binarized_classification_performances(
        y_true: list,
        y_pred: list
) -> tuple[float, float, float, float, float]:
    """
    returns 
    --------
    precision: float, 
    accuracy: float, 
    balanced accuracy: float, 
    recall: float, 
    f1 score: float, 
    Matthews correlation coefficent: float, 
    Area under receiver operating curve: float, 
    Average precision score: float
    """
    y_true_binned, y_pred_binned, y_pred_proba = setpositives_bin(y_true, y_pred)
    prec, acc, bacc, rec, f1, mcc, auroc, aps, _fpr, _tpr = binary_classification_metrics(
        y_true_binned, y_pred_binned, y_pred_proba)
    return prec, acc, bacc, rec, f1, mcc, auroc, aps
