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

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from adjustText import adjust_text
import logging
logger = logging.getLogger('pypef.ml.regression')

from pypef.utils.performance import get_performances, get_binarized_classification_performances


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
    _prec, _acc, _bacc, rec, _f1, _mcc, _auroc, _aps = get_binarized_classification_performances(y_true, y_pred)
    if hybrid:
        spearman_rho = stats.spearmanr(y_true, y_pred)[0]
        # Recall: Here, top 10 % fit variants are positive labeled (1), rest are labeled negative (0) by default
        plt.scatter(y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.7, c=y_true, vmin=min(y_true), vmax=max(y_true),
                   label=f'Spearman\'s ' + fr'$\rho$ = {spearman_rho:.3f}' + '\n' 
                   + r'Recall$_\mathrm{top 10 \%}$' + f' = {rec:.3f}\n'
                   + fr'($N$ = {len(y_true)})'
        )
        if name != '':
            file_name = f'DCA_Hybrid_Model_Performance_{name}.png'
        else:
            file_name = 'DCA_Hybrid_Model_Performance.png'
    else:
        r_squared, rmse, nrmse, pearson_r, spearman_rho = get_performances(
            y_true=y_true, y_pred=y_pred
        )
        plt.scatter(
            y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.7, c=y_true, vmin=min(y_true), vmax=max(y_true),
            label=r'$R^2$' + f' = {r_squared:.3f}' + f'\nRMSE = {rmse:.3f}' + f'\nNRMSE = {nrmse:.3f}' 
                  + f'\nPearson\'s ' + r'$r$'+f' = {pearson_r:.3f}' 
                  + f'\nSpearman\'s ' + fr'$\rho$ = {spearman_rho:.3f}' + '\n' 
                  + r'Recall$_\mathrm{top 10 \%}$' + f' = {rec:.3f}\n'
                  + fr'($N$ = {len(y_true)})'
        )
        if name != '':
            file_name = f'ML_Model_Performance_{name}.png'
        else:
            file_name = 'ML_Model_Performance.png'
        # x = np.linspace(min(y_pred), max(y_pred), 100)
        # ax.plot(x, x, color='black', linewidth=0.25)  # plot diagonal line
    plt.legend(prop={'size': 8})
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    logger.info('Plotting...')
    if label:
        logger.info('Adjusting variant labels for plotting can take some '
                    'time (the limit for labeling is 150 data points)...')
        if len(y_true) < 150:
            texts = [plt.text(y_true[i], y_pred[i], txt, fontsize=4)
                     for i, txt in enumerate(variants)]
            adjust_text(
                texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5, time_lim=10)
        else:
            logger.info("Terminating label process. Too many variants "
                        "(> 150) for labeled plotting.")
    # Uncomment for renaming new plots
    # i = 1
    # while os.path.isfile(file_name):
    #     i += 1  # iterate until finding an unused file name
    #     file_name = f'DCA_Hybrid_Model_LS_TS_Performance({i}).png'
    plt.colorbar()
    plt.savefig(file_name, dpi=500)
    plt.close('all')
    logger.info(f'Saved plot as {os.path.abspath(file_name)}...')
