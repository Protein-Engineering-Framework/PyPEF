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

import warnings
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


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