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
Validation modules from regression.py for AAindex-based encoding
modified for parallelization of the 566 AAindices used for encoding
with Ray, see https://docs.ray.io/en/latest/index.html.
"""

import matplotlib
matplotlib.use('Agg')
import os
import ray
import warnings

from pypef.utils.variant_data import get_sequences_from_file
from pypef.ml.regression import (
    full_aaidx_txt_path, path_aaindex_dir, AAIndexEncoding, get_regressor_performances
)

# to handle UserWarning for PLS n_components as error and general regression module warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


@ray.remote
def parallel(
        d,
        core,
        aa_indices,
        train_set,
        test_set,
        regressor='pls',
        no_fft=False
):
    """
    Parallelization of running using the user-defined number of cores.
    Defining the task for each core.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    performances = []
    for i in range(d[core][0], d[core][1]):
        aaindex = aa_indices[i]  # Parallelization of AAindex iteration
        sequences_train, _, y_train = get_sequences_from_file(train_set)
        sequences_test, _, y_test = get_sequences_from_file(test_set)
        x_aaidx_train = AAIndexEncoding(full_aaidx_txt_path(aaindex), sequences_train)
        x_aaidx_test = AAIndexEncoding(full_aaidx_txt_path(aaindex), sequences_test)

        if not no_fft:  # X is FFT-ed of encoded alphabetical sequence
            x_train, _ = x_aaidx_train.collect_encoded_sequences()
            x_test, _ = x_aaidx_test.collect_encoded_sequences()
        else:  # X is raw encoded of alphabetical sequence
            _, x_train = x_aaidx_train.collect_encoded_sequences()
            _, x_test = x_aaidx_test.collect_encoded_sequences()

        # If x_learn (or y_learn) is an empty array, the sequence could not be encoded,
        # because of NoneType value. -> Skip
        if x_train == 'skip' or x_test == 'skip':
            continue  # skip the rest and do next iteration
        r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params = \
            get_regressor_performances(x_train, x_test, y_train, y_test, regressor)
        if r2 is not None:  # None for metrics if MSE can't be calculated
            performances.append([
                aaindex, r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params
            ])

    return performances


def aaindex_performance_parallel(
        train_set,
        test_set,
        cores,
        regressor='pls',
        no_fft=False,
        sort='1'
):
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
        result_ids.append(parallel.remote(
            d, j, aa_indices, train_set, test_set, regressor, no_fft))

    results = ray.get(result_ids)

    performances = []
    for core in range(cores):
        for j, _ in enumerate(results[core]):
            performances.append(results[core][j])

    try:
        sort = int(sort)
        if sort == 2 or sort == 3:
            performances.sort(key=lambda x: x[sort])
        else:
            performances.sort(key=lambda x: x[sort], reverse=True)

    except ValueError:
        raise ValueError(
            "Choose between options 1 to 5 (R2, RMSE, NRMSE, "
            "Pearson's r, Spearman's rho."
        )

    return performances
