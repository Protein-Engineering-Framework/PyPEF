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

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from adjustText import adjust_text
import logging
logger = logging.getLogger('pypef.ml.regression')

from pypef.utils.performance import get_performances


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
    figure, ax = plt.subplots()
    if hybrid:
        spearman_rho = stats.spearmanr(y_true, y_pred)[0]
        ax.scatter(y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.7,
                   label=f'Spearman\'s ' + fr'$\rho$ = {spearman_rho:.3f}' + '\n' + fr'($N$ = {len(y_true)})')
        file_name = name + 'DCA_Hybrid_Model_Performance.png'
    else:
        r_squared, rmse, nrmse, pearson_r, spearman_rho = get_performances(
            y_true=y_true, y_pred=y_pred
        )
        ax.scatter(
            y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black', alpha=0.7,
            label=r'$R^2$' + f' = {r_squared:.3f}' + f'\nRMSE = {rmse:.3f}' + f'\nNRMSE = {nrmse:.3f}' +
                  f'\nPearson\'s ' + r'$r$'+f' = {pearson_r:.3f}' + f'\nSpearman\'s ' +
                  fr'$\rho$ = {spearman_rho:.3f}' + '\n' + fr'($N$ = {len(y_true)})'
        )
        file_name = name + 'ML_Model_Performance.png'
        # x = np.linspace(min(y_pred), max(y_pred), 100)
        # ax.plot(x, x, color='black', linewidth=0.25)  # plot diagonal line
    ax.legend(prop={'size': 8})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    logger.info('Plotting...')
    if label:
        logger.info('Adjusting variant labels for plotting can take some '
                    'time (the limit for labeling is 150 data points)...')
        if len(y_true) < 150:
            texts = [ax.text(y_true[i], y_pred[i], txt, fontsize=4)
                     for i, txt in enumerate(variants)]
            adjust_text(
                texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5, lim=250)
        else:
            logger.info("Terminating label process. Too many variants "
                        "(> 150) for plotting (labels would overlap).")
    # Uncomment for renaming new plots
    # i = 1
    # while os.path.isfile(file_name):
    #     i += 1  # iterate until finding an unused file name
    #     file_name = f'DCA_Hybrid_Model_LS_TS_Performance({i}).png'
    plt.savefig(file_name, dpi=500)
    plt.close('all')
