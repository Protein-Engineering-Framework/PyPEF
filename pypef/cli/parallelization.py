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
Validation modules from Modules_Regression.py
modified for parallelization with Ray.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import ray
import warnings

from pypef.cli.regression import full_path, path_aaindex_dir, XY, get_r2

# to handle UserWarning for PLS n_components as error and general regression module warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def formatted_output_parallel(aaindex_r2_list, minimum_r2=0.0, no_fft=False):
    """
    takes the sorted list from function R2_List and writes the model names with an R2 ≥ 0
    as well as the corresponding number of components for each model so that the user gets
    a list (Model_Results.txt) of the top ranking models for the given validation set.
    """
    index, value, value2, value3, value4, value5, regr_models, parameters = [], [], [], [], [], [], [], []

    for (idx, val, val2, val3, val4, val5, r_m, pam) in aaindex_r2_list:
        if val >= minimum_r2:
            index.append(idx[:-4])
            value.append('{:f}'.format(val))
            value2.append('{:f}'.format(val2))
            value3.append('{:f}'.format(val3))
            value4.append('{:f}'.format(val4))
            value5.append('{:f}'.format(val5))
            regr_models.append(r_m)
            parameters.append(pam)

    if len(value) == 0:
        raise ValueError('No model with positive R2.')

    data = np.array([index, value, value2, value3, value4, value5, regr_models, parameters]).T
    col_width = max(len(str(value)) for row in data for value in row[:-1]) + 5

    head = ['Index', 'R2', 'RMSE', 'NRMSE', 'Pearson_r', 'Regression', 'Model parameters']
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


@ray.remote
def parallel(d, core, aa_indices, learning_set, validation_set, regressor='pls', no_fft=False):
    """
    Parallelization of running using the user-defined number of cores.
    Defining the task for each core.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    aaindex_r2_list = []
    for i in range(d[core][0], d[core][1]):
        aaindex = aa_indices[i]  # Parallelization of AAindex iteration
        xy_learn = XY(full_path(aaindex), learning_set)

        if not no_fft:  # X is FFT-ed of encoded alphabetical sequence
            x_learn, y_learn, _ = xy_learn.get_x_and_y()
        else:  # X is raw encoded of alphabetical sequence
            _, y_learn, x_learn = xy_learn.get_x_and_y()

        # If x_learn (or y_learn) is an empty array, the sequence could not be encoded,
        # because of NoneType value. -> Skip
        if len(x_learn) != 0:
            xy_test = XY(full_path(aaindex), validation_set)

            if not no_fft:  # X is FFT-ed of the encoded alphabetical sequence
                x_test, y_test, _ = xy_test.get_x_and_y()
            else:  # X is the raw encoded of alphabetical sequence
                _, y_test, x_test = xy_test.get_x_and_y()

            r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params = get_r2(
                x_learn, x_test, y_learn, y_test, regressor
            )
            if r2 is not None:  # get_r2() returns None for metrics if MSE can't be calculated
                aaindex_r2_list.append([aaindex, r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params])

    return aaindex_r2_list


def r2_list_parallel(learning_set, validation_set, cores, regressor='pls', no_fft=False, sort='1'):
    """
    Parallelization of running using the user-defined number of cores.
    Calling function Parallel to execute the parallel running and
    getting the results from each core each being defined by a result ID.
    """
    aa_indices = [file for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]

    split = int(len(aa_indices) / cores)
    last_split = int(len(aa_indices) % cores) + split

    d = {}
    for i in range(cores - 1):
        d[i] = [i * split, i * split + split]

    d[cores - 1] = [(cores - 1) * split, (cores - 1) * split + last_split]

    result_ids = []
    for j in range(cores):  # Parallel running
        result_ids.append(parallel.remote(d, j, aa_indices, learning_set, validation_set, regressor, no_fft))

    results = ray.get(result_ids)

    aaindex_r2_list = []
    for core in range(cores):
        for j, _ in enumerate(results[core]):
            aaindex_r2_list.append(results[core][j])

    try:
        sort = int(sort)
        if sort == 2 or sort == 3:
            aaindex_r2_list.sort(key=lambda x: x[sort])
        else:
            aaindex_r2_list.sort(key=lambda x: x[sort], reverse=True)

    except ValueError:
        raise ValueError("Choose between options 1 to 5 (R2, RMSE, NRMSE, Pearson's r, Spearman's rho.")

    return aaindex_r2_list
