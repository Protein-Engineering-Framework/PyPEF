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

import logging
logger = logging.getLogger('pypef.dca.dca_run')
import ray

from pypef.utils.variant_data import read_csv, get_wt_sequence
from pypef.dca.plmc_encoding import save_plmc_dca_encoding_model
from pypef.dca.hybrid_model import get_model_and_type, performance_ls_ts, predict_ps, generate_model_and_save_pkl
from pypef.dca.gremlin_inference import save_gremlin_as_pickle, save_corr_csv, plot_all_corr_mtx
from pypef.utils.low_n_mutation_extrapolation import performance_mutation_extrapolation, low_n


def run_pypef_hybrid_modeling(arguments):
    threads = abs(arguments['--threads']) if arguments['--threads'] is not None else 1
    threads = threads + 1 if threads == 0 else threads
    try:
        _, model_type = get_model_and_type(arguments['--params'], arguments['--mutation_sep'])
    except TypeError:
        model_type = 'undefined'
    except SystemError:
        model_type = 'undefined'
    if model_type in ['GREMLIN', 'DCAHybridModel'] and threads > 1:
        logger.info(f'No (Ray) parallelization for {model_type} model...')
    elif model_type not in ['GREMLIN', 'DCAHybridModel'] and threads > 1:
        ray.init()
        logger.info(f'Using {threads} threads for running...')
    if model_type == 'DCAHybridModel':
        logger.info(
            f"Note that the hybrid model only optimizes model performances in terms of "
            f"Spearman's correlation of measured versus predicted values. Further, the "
            f"hybrid approach uses only Ridge regression for supervised ML-based hybrid "
            f"model contribution. In hybrid modeling, the ranks of predictions are "
            f"important and not the exact predicted value."
        )

    if arguments['--ts']:
        performance_ls_ts(
            ls_fasta=arguments['--ls'],
            ts_fasta=arguments['--ts'],
            threads=threads,
            params_file=arguments['--params'],
            model_pickle_file=arguments['--model'],
            substitution_sep=arguments['--mutation_sep'],
            label=arguments['--label']
        )

    elif arguments['--params'] and arguments['--model']:
        prediction_dict = {}
        prediction_dict.update({
            'drecomb': arguments['--drecomb'],
            'trecomb': arguments['--trecomb'],
            'qarecomb': arguments['--qarecomb'],
            'qirecomb': arguments['--qirecomb'],
            'ddiverse': arguments['--ddiverse'],
            'tdiverse': arguments['--tdiverse'],
            'qdiverse': arguments['--qdiverse']
        })

        predict_ps(
            prediction_dict=prediction_dict,
            params_file=arguments['--params'],
            threads=threads,
            separator=arguments['--mutation_sep'],
            model_pickle_file=arguments['--model'],
            prediction_set=arguments['--ps'],
            negative=arguments['--negative']
        )

    elif arguments['train_and_save']:
        variants, fitnesses, _ = read_csv(arguments['--input'])
        generate_model_and_save_pkl(
            variants=variants,
            ys_true=fitnesses,
            params_file=arguments['--params'],
            wt=arguments['--wt'],
            train_percent_fit=arguments['--fit_size'],
            test_percent=arguments['--test_size'],
            random_state=arguments['--rnd_state'],
            substitution_sep=arguments['--mutation_sep'],
            threads=arguments['--threads']
        )

    elif arguments['low_n'] or arguments['extrapolation']:
        if arguments['low_n']:
            low_n(
                encoded_csv=arguments['--input'],
                hybrid_modeling=arguments['hybrid']
            )
        elif arguments['extrapolation']:
            performance_mutation_extrapolation(
                encoded_csv=arguments['--input'],
                cv_regressor=arguments['--regressor'],
                conc=arguments['--conc'],
                hybrid_modeling=arguments['hybrid']
            )

    elif arguments['param_inference']:
        if arguments['--msa']:
            save_gremlin_as_pickle(
                alignment=arguments['--msa'],
                wt_seq=get_wt_sequence(arguments['--wt']),
                opt_iter=arguments['--opt_iter']
            )
        elif arguments['--params']:
            save_plmc_dca_encoding_model(
                params_file=arguments['--params'],
                substitution_sep=arguments['--mutation_sep']
            )

    elif arguments['save_msa_info']:
        gremlin = save_gremlin_as_pickle(
            alignment=arguments['--msa'],
            wt_seq=get_wt_sequence(arguments['--wt']),
            opt_iter=arguments['--opt_iter']
        )
        save_corr_csv(gremlin)
        plot_all_corr_mtx(gremlin)

    else:
        performance_ls_ts(
            ls_fasta=arguments['--ls'],
            ts_fasta=arguments['--ts'],
            threads=threads,
            params_file=arguments['--params'],
            substitution_sep=arguments['--mutation_sep']
        )
