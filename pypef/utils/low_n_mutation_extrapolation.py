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


import os
import random
import logging
logger = logging.getLogger('pypef.utils.low_n_mutation_extrapolation')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from pypef.ml.regression import cv_regression_options
from pypef.dca.hybrid_model import DCAHybridModel
from pypef.utils.variant_data import process_df_encoding, get_basename


def get_train_sizes(number_variants) -> np.ndarray:
    """
    Generates a list of train sizes to perform low-n with.
    Returns
    -------
    Numpy array of train sizes up to 80% (i.e. 0.8 * N_variants).
    """
    eighty_percent = int(number_variants * 0.8)
    train_sizes = np.sort(np.concatenate([
        np.arange(15, 50, 5), np.arange(50, 100, 10),
        np.arange(100, 150, 20), [160, 200, 250, 300, eighty_percent],
        np.arange(400, 1100, 100)
    ]))
    idx_max = np.where(train_sizes >= eighty_percent)[0][0] + 1
    return train_sizes[:idx_max]


def plot_low_n(
        train_sizes: list,
        avg_spearmanr: list,
        stddev_spearmanr: list,
        plt_name: str = ''
):
    """
    Plot the performance results of the low N engineering task.
    """
    logger.info('Plotting...')
    plt.plot(train_sizes, avg_spearmanr, 'ko--', linewidth=1, markersize=1.5)
    plt.fill_between(
        np.array(train_sizes),
        np.array(avg_spearmanr) + np.array(stddev_spearmanr),
        np.array(avg_spearmanr) - np.array(stddev_spearmanr),
        alpha=0.5
    )
    plt.ylim(0, max(np.array(avg_spearmanr) + np.array(stddev_spearmanr)))
    plt.xlabel('Train sizes')
    plt.ylabel(r"Spearman's $\rho$")
    plt.savefig(plt_name + '.png', dpi=500)
    plt.clf()


def low_n(
        encoded_csv: str,
        cv_regressor: str = None,
        n_runs: int = 10,
        hybrid_modeling: bool = False,
        train_size_train: float = 0.66
):
    """
    Performs the "low N protein engineering task" learning on distinct
    numbers of encoded_variant_sequences-fitness data to predict the
    left out data (full dataset - train dataset). Maximum sizes of
    learning sets is 0.8 * full dataset (and thus maximal size of test
    set 0.2 * full dataset).
    """
    df = pd.read_csv(encoded_csv, sep=';', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(encoded_csv, sep=',', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(encoded_csv, sep='\t', comment='#')
    if cv_regressor:
        name = 'ml_' + cv_regressor
        if cv_regressor == 'pls_loocv':
            raise SystemError(
                'PLS LOOCV is not (yet) implemented for the extrapolation task. '
                'Please choose another CV regression option.'
            )
        regressor = cv_regression_options(cv_regressor)
    elif hybrid_modeling:
        name = 'hybrid_ridge'
    n_variants = df.shape[0]
    train_sizes = get_train_sizes(n_variants).tolist()
    variants, X, y = process_df_encoding(df)

    avg_spearmanr, stddev_spearmanr = [], []
    # test_sizes = [n_variants - size for size in train_sizes]
    if hybrid_modeling:
        logger.info('Using first CSV row/entry as wild type reference...')
    for size in tqdm(train_sizes):
        spearmanr_nruns = []
        for _ in range(n_runs):
            train_idxs = random.sample(range(n_variants - 1), int(size))
            test_idxs = []
            for n in range(n_variants - 1):
                if n not in train_idxs:
                    test_idxs.append(n)
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_test, y_test = X[test_idxs], y[test_idxs]

            if hybrid_modeling:
                x_wt = X[0]  # WT should be first CSV variant entry
                hybrid_model = DCAHybridModel(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,  # only used for adjusting +/- sign of y_dca, can also be None
                    y_test=y_test,  # only used for adjusting +/- sign of y_dca, can also be None
                    X_wt=x_wt
                )
                beta_1, beta_2, reg = hybrid_model.settings(
                    X_train, y_train, train_size_fit=train_size_train
                )
                spearmanr_nruns.append(
                    hybrid_model.spearmanr(
                        y_test,
                        hybrid_model.hybrid_prediction(
                            X_test, reg, beta_1, beta_2
                        )
                    )
                )

            else:  # ML
                regressor.fit(X_train, y_train)
                # Best CV params: best_params = regressor.best_params_
                y_pred = regressor.predict(X_test)
                spearmanr_nruns.append(stats.spearmanr(y_test, y_pred)[0])
        avg_spearmanr.append(np.mean(spearmanr_nruns))
        stddev_spearmanr.append(np.std(spearmanr_nruns, ddof=1))

    plot_low_n(
        train_sizes,
        avg_spearmanr,
        stddev_spearmanr,
        'low_N_' + str(encoded_csv[:-4] + '_' + name)
    )

    return train_sizes, avg_spearmanr, stddev_spearmanr


def count_mutation_levels_and_get_dfs(df_encoding) -> tuple:
    """
    The input dataframe (from the sequence encoding CSV file) is split
    according to levels of variant substitutions. Substitution seperator
    is '/'.
    """
    single_variants_index, all_higher_variants_index = [], []
    double_i, triple_i, quadruple_i, quintuple_i, sextuple_i, \
    septuple_i, octuple_i, nonuple_i, higher_nine_i = [], [], [], [], [], [], [], [], []
    for i, row in enumerate(df_encoding.iloc[:, 0]):  # iterate over variant column
        if '/' in row:  # TypeError: argument of type 'float' is not iterable if empty columns are (at end of) CSV
            all_higher_variants_index.append(i)
            if row.count('/') == 1:
                double_i.append(i)
            elif row.count('/') == 2:
                triple_i.append(i)
            elif row.count('/') == 3:
                quadruple_i.append(i)
            elif row.count('/') == 4:
                quintuple_i.append(i)
            elif row.count('/') == 5:
                sextuple_i.append(i)
            elif row.count('/') == 6:
                septuple_i.append(i)
            elif row.count('/') == 7:
                octuple_i.append(i)
            elif row.count('/') == 8:
                nonuple_i.append(i)
            elif row.count('/') >= 9:
                higher_nine_i.append(i)
        else:
            single_variants_index.append(i)
    logger.info(f'\nNo. Singles: {len(single_variants_index)}\nNo. All higher: {len(all_higher_variants_index)}\n'
                f'2: {len(double_i)}\n3: {len(triple_i)}\n4: {len(quadruple_i)}\n'
                f'5: {len(quintuple_i)}\n6: {len(sextuple_i)}\n7: {len(septuple_i)}\n'
                f'8: {len(octuple_i)}\n9: {len(nonuple_i)}\n>=10: {len(higher_nine_i)}')
    return (
        df_encoding.iloc[single_variants_index, :],
        df_encoding.iloc[double_i, :],
        df_encoding.iloc[triple_i, :],
        df_encoding.iloc[quadruple_i, :],
        df_encoding.iloc[quintuple_i, :],
        df_encoding.iloc[sextuple_i, :],
        df_encoding.iloc[septuple_i, :],
        df_encoding.iloc[octuple_i, :],
        df_encoding.iloc[nonuple_i, :],
        df_encoding.iloc[higher_nine_i, :],
        df_encoding.iloc[all_higher_variants_index, :],
    )


def performance_mutation_extrapolation(
        encoded_csv: str,
        cv_regressor: str = None,
        train_size: float = 0.66,
        conc: bool = False,
        save_model: bool = True,
        hybrid_modeling: bool = False
) -> dict:
    """
    Train on distinct mutation levels, e.g. only single-substituted samples
    of encoded_variant_sequences-fitness data to predict distinct levels
    of higher substituted variants (i.e. 1->2, 1->3, 1->4 etc.). Also can
    train on concatenated levels of substitution-fitness data using the flag
    --conc, i.e. conc = True (i.e. 1->2, 1+2->3, 1+2+3->4, etc.).
    """
    df = pd.read_csv(encoded_csv, sep=';', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(encoded_csv, sep=',', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(encoded_csv, sep='\t', comment='#')

    df_mut_lvl = count_mutation_levels_and_get_dfs(df)
    name = ''
    if save_model:
        try:
            os.mkdir('Pickles')
        except FileExistsError:
            pass
    if hybrid_modeling:
        regressor = None
        name = 'hybrid_ridge_' + get_basename(encoded_csv)
    elif cv_regressor:
        name = 'ml_' + cv_regressor + '_' + get_basename(encoded_csv)
        if cv_regressor == 'pls_loocv':
            raise SystemError(
                'PLS LOOCV is not implemented for the extrapolation '
                'task. Please choose another CV regressor.'
            )
        regressor = cv_regression_options(cv_regressor)
        beta_1, beta_2 = None, None
    else:
        regressor = None
        hybrid_model = None
    data = {}
    collected_levels = []
    for i_m, mutation_level_df in enumerate(df_mut_lvl):
        if mutation_level_df.shape[0] != 0:
            collected_levels.append(i_m)
    train_idx_appended = []
    if len(collected_levels) > 1:
        train_idx = collected_levels[0]
        train_df = df_mut_lvl[train_idx]
        train_variants, X_train, y_train = process_df_encoding(train_df)
        all_higher_df = df_mut_lvl[-1]  # only used for adjusting +/- of y_dca
        all_higher_variants, X_all_higher, y_all_higher = process_df_encoding(all_higher_df)
        if hybrid_modeling:
            x_wt = X_train[0]
            hybrid_model = DCAHybridModel(
                X_train=X_train,
                y_train=y_train,
                X_test=X_all_higher,  # only used for adjusting +/- of y_dca, can also be None but
                y_test=y_all_higher,  # higher risk of wrong sign assignment of beta_1 (y_dca)
                X_wt=x_wt
            )
            beta_1, beta_2, reg = hybrid_model.settings(
                X_train, y_train, train_size_fit=train_size)
            pickle.dump(
                {'hybrid_model': hybrid_model, 'beta_1': beta_1, 'beta_2': beta_2,
                 'spearman_rho': float('nan'), 'regressor': reg},
                open(os.path.join('Pickles', 'HYBRID_LVL_1'), 'wb')
            )
        elif cv_regressor:
            logger.info('Fitting regressor on lvl 1 substitution data...')
            regressor.fit(X_train, y_train)
            if save_model:
                logger.info(f'Saving model as Pickle file: ML_LVL_1')
                pickle.dump(regressor, open(os.path.join('Pickles', 'ML_LVL_1'), 'wb'))
        for i, _ in enumerate(tqdm(collected_levels)):
            if i < len(collected_levels) - 1:  # not last i else error, last entry is: lvl 1 --> all higher variants
                test_idx = collected_levels[i + 1]
                test_df = df_mut_lvl[test_idx]
                test_variants, X_test, y_test = process_df_encoding(test_df)
                if not conc:
                    # For training on distinct iterated level i, uncomment lines below:
                    # train_idx = collected_levels[i]
                    # train_df = self.mutation_level_dfs[train_idx]
                    # train_variants, X_train, y_train = self._process_df_encoding(train_df)
                    if hybrid_modeling:
                        data.update({
                            test_idx + 1:
                                {
                                    'hybrid_model': hybrid_model,
                                    'max_train_lvl': train_idx + 1,
                                    'n_y_train': len(y_train),
                                    'test_lvl': test_idx + 1,
                                    'n_y_test': len(y_test),
                                    'spearman_rho': hybrid_model.spearmanr(
                                        y_test,
                                        hybrid_model.hybrid_prediction(
                                            X_test, reg, beta_1, beta_2
                                        )
                                    ),
                                    'beta_1': beta_1,
                                    'beta_2': beta_2,
                                    'regressor': reg
                                }
                        })
                    else:  # ML
                        data.update({
                            test_idx + 1:
                                {
                                    'regressor': regressor,
                                    'max_train_lvl': train_idx + 1,
                                    'n_y_train': len(y_train),
                                    'test_lvl': test_idx + 1,
                                    'n_y_test': len(y_test),
                                    'spearman_rho': stats.spearmanr(
                                        y_test,                    # Call predict on the BaseSearchCV estimator
                                        regressor.predict(X_test)  # with the best found parameters
                                    )[0]
                                }
                        })

                else:  # conc mode, training on mutational levels i: 1, ..., max(i)-1
                    train_idx_appended.append(collected_levels[i])
                    if i < len(collected_levels) - 2:  # -2 as not the last (all_higher)  ## i != 0 and
                        train_df_appended_conc = pd.DataFrame()
                        for idx in train_idx_appended:
                            train_df_appended_conc = pd.concat(
                                [train_df_appended_conc, df_mut_lvl[idx]])
                        train_variants_conc, X_train_conc, y_train_conc = \
                            process_df_encoding(train_df_appended_conc)
                        if hybrid_modeling:  # updating hybrid model params with newly inputted concatenated train data
                            beta_1_conc, beta_2_conc, reg_conc = hybrid_model.settings(
                                X_train_conc,
                                y_train_conc,
                                train_size_fit=train_size
                            )
                            data.update({
                                test_idx + 1:
                                    {
                                        'hybrid_model': hybrid_model,
                                        'max_train_lvl': train_idx_appended[-1] + 1,
                                        'n_y_train': len(y_train_conc),
                                        'test_lvl': test_idx + 1,
                                        'n_y_test': len(y_test),
                                        'spearman_rho': hybrid_model.spearmanr(
                                            y_test,
                                            hybrid_model.hybrid_prediction(
                                                X_test, reg_conc, beta_1_conc, beta_2_conc
                                            )
                                        ),
                                        'beta_1': beta_1_conc,
                                        'beta_2': beta_2_conc,
                                        'regressor': reg_conc
                                    }
                            })
                        else:  # ML updating pureML regression model params with newly inputted concatenated train data
                            # Fitting regressor on concatenated substitution data
                            regressor.fit(X_train_conc, y_train_conc)
                            data.update({
                                test_idx + 1:
                                {
                                    'max_train_lvl': train_idx_appended[-1] + 1,
                                    'n_y_train': len(y_train_conc),
                                    'test_lvl': test_idx + 1,
                                    'n_y_test': len(y_test),
                                    'spearman_rho': stats.spearmanr(
                                        y_test,  # Call predict on the BaseSearchCV estimator
                                        regressor.predict(X_test)  # with the best found parameters
                                        )[0],
                                    'regressor': regressor
                                }
                            })
    plot_extrapolation(data, name, conc)

    return data


def plot_extrapolation(
        extrapolation_data: dict,
        name: str = '',
        conc=False
):
    """
    Plot extrapolation results.
    """
    logger.info('Plotting...')
    test_lvls, spearman_rhos, label_infos = [], [], []
    for test_lvl, result_dict in extrapolation_data.items():
        if result_dict['spearman_rho'] is np.nan:
            continue
        test_lvls.append(test_lvl)
        spearman_rhos.append(result_dict['spearman_rho'])
        label_infos.append(
            r'$\leq$' + str(result_dict['max_train_lvl']) + r'$\rightarrow$' + str(result_dict['test_lvl']) +
            '\n' + str(result_dict['n_y_train']) + r'$\rightarrow$' + str(result_dict['n_y_test'])
        )
    label_infos[0] = 'Lvl: ' + label_infos[0].split('\n')[0] + '\n' + r'$N$: ' + label_infos[0].split('\n')[1]
    if not conc:
        label_infos[-1] = label_infos[-1][6] + r'$\rightarrow$' + '>' + \
                          label_infos[-1][6] + '\n' + label_infos[-1].split('\n')[1]
    plt.plot(test_lvls, spearman_rhos, 'x--', markeredgecolor='k', linewidth=0.7, markersize=4)
    plt.fill_between(
        np.array(test_lvls),
        np.repeat(min(spearman_rhos), len(spearman_rhos)),
        np.array(spearman_rhos),
        alpha=0.3
    )
    if conc:
        name += '_train_concat_lvls'
    else:
        name += '_train_lvl_1'
    plt.xticks(test_lvls, label_infos, fontsize=5)
    plt.ylabel(r"Spearman's $\rho$")
    plt.savefig(name + '_extrapolation.png', dpi=500)
    plt.clf()
