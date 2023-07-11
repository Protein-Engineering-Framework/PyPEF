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
from os import listdir
from os.path import isfile, join
import ray
import logging
logger = logging.getLogger('pypef.ml.ml_run')

from pypef.ml.parallelization import aaindex_performance_parallel
from pypef.ml.regression import (
    read_models, formatted_output, performance_list, save_model, predict, predict_ts
)
from pypef.utils.to_file import predictions_out
from pypef.utils.low_n_mutation_extrapolation import low_n, performance_mutation_extrapolation


def run_pypef_pure_ml(arguments):
    """
    Running the program, importing all required self-made modules and
    running them dependent on user-passed input arguments using docopt
    for argument parsing.
    """
    threads = abs(arguments['--threads']) if arguments['--threads'] is not None else 1
    threads = threads + 1 if threads == 0 else threads
    if threads > 1:
        ray.init()
    if arguments['--show']:
        if arguments['MODELS'] != str(5):
            try:
                print(read_models(int(arguments['MODELS'])))
            except ValueError:
                print(read_models(5))
            except TypeError:
                print(read_models(5))
        else:
            print(read_models(5))

    else:
        if arguments['--ls'] is not None and arguments['--ts'] is not None:  # LS --> TS
            if arguments['--model'] is None:
                path = os.getcwd()
                try:
                    t_save = int(arguments['--save'])
                except ValueError:
                    t_save = 5
                # Parallelization of AAindex iteration if threads is not None (but int)
                if threads > 1 and arguments['--encoding'] == 'aaidx':
                    logger.info(f'Using {threads} threads for parallel computing. Running...')
                    encoding_performance_list = aaindex_performance_parallel(
                        train_set=arguments['--ls'],
                        test_set=arguments['--ts'],
                        cores=threads,
                        regressor=arguments['--regressor'],
                        no_fft=arguments['--nofft'],
                        sort=arguments['--sort']
                    )

                else:  # run using a single core or use onehot or DCA-based encoding for modeling
                    encoding_performance_list = performance_list(
                        train_set=arguments['--ls'],
                        test_set=arguments['--ts'],
                        encoding=arguments['--encoding'],
                        regressor=arguments['--regressor'],
                        no_fft=arguments['--nofft'],
                        sort=arguments['--sort'],
                        couplings_file=arguments['--params'],  # only for DCA
                        threads=threads  # only for DCA
                    )

                formatted_output(
                    performance_list=encoding_performance_list,
                    no_fft=arguments['--nofft'],
                    minimum_r2=-1E9
                )

                # save_model encodes variants again (possibly change)
                save_model(
                    path=path,
                    performances=encoding_performance_list,
                    training_set=arguments['--ls'],
                    test_set=arguments['--ts'],
                    threshold=t_save,
                    encoding=arguments['--encoding'],
                    regressor=arguments['--regressor'],
                    no_fft=arguments['--nofft'],
                    train_on_all=arguments['--all'],
                    couplings_file=arguments['--params'],  # only for DCA
                    threads=threads,  # only for DCA
                    label=arguments['--label']
                )

        # Prediction of single .fasta file
        elif arguments['--ts'] is not None and arguments['--model'] is not None:
            predict_ts(
                path=os.getcwd(),
                model=arguments['--model'],
                test_set=arguments['--ts'],
                encoding=arguments['--encoding'],
                no_fft=arguments['--nofft'],
                couplings_file=arguments['--params'],  # only for DCA
                label=arguments['--label'],
                threads=threads
            )

        elif arguments['--ps'] is not None and arguments['--model'] is not None:
            predictions = predict(
                path=os.getcwd(),
                prediction_set=arguments['--ps'],
                model=arguments['--model'],
                encoding=arguments['--encoding'],
                mult_path=None,
                no_fft=arguments['--nofft'],
                couplings_file=arguments['--params'],  # only for DCA
                threads=threads  # only for DCA
            )
            if predictions == 'skip' and not arguments['--params']:
                raise SystemError("No couplings file provided. DCA-based sequence encoding "
                                  "requires a (plmc or GREMLIN) parameter file.")
            if arguments['--negative']:
                predictions = sorted(predictions, key=lambda x: x[0], reverse=False)
            predictions_out(
                predictions=predictions,
                model=arguments['--model'],
                prediction_set=arguments['--ps']
            )

        # Prediction on recombinant/diverse variant folder data
        elif arguments['--pmult'] and arguments['--model'] is not None:
            path = os.getcwd()
            recombs_total = []
            recomb_d, recomb_t, recomb_qa, recomb_qi = \
                '/Recomb_Double_Split/', '/Recomb_Triple_Split/', \
                '/Recomb_Quadruple_Split/', '/Recomb_Quintuple_Split/'
            diverse_d, diverse_t, diverse_q = \
                '/Diverse_Double_Split/', '/Diverse_Triple_Split/', '/Diverse_Quadruple_Split/'
            if arguments['--drecomb']:
                recombs_total.append(recomb_d)
            if arguments['--trecomb']:
                recombs_total.append(recomb_t)
            if arguments['--qarecomb']:
                recombs_total.append(recomb_qa)
            if arguments['--qirecomb']:
                recombs_total.append(recomb_qi)
            if arguments['--ddiverse']:
                recombs_total.append(diverse_d)
            if arguments['--tdiverse']:
                recombs_total.append(diverse_t)
            if arguments['--qdiverse']:
                recombs_total.append(diverse_q)
            if arguments['--drecomb'] is False \
                    and arguments['--trecomb'] is False \
                    and arguments['--qarecomb'] is False \
                    and arguments['--qirecomb'] is False \
                    and arguments['--ddiverse'] is False \
                    and arguments['--tdiverse'] is False \
                    and arguments['--qdiverse'] is False:
                raise SystemError('Define prediction target for --pmult, e.g. --pmult --drecomb.')

            for args in recombs_total:
                predictions_total = []
                logger.info(f'Running predictions for variant-sequence files in directory {args[1:-1]}...')
                path_recomb = path + args
                files = [path_recomb + f for f in listdir(path_recomb)
                         if isfile(join(path_recomb, f)) if f.endswith('.fasta')]
                for i, file in enumerate(files):
                    logger.info(f'Encoding files ({i+1}/{len(files)}) for prediction...')
                    predictions = predict(
                        path=path,
                        prediction_set=file,
                        model=arguments['--model'],
                        encoding=arguments['--encoding'],
                        mult_path=path_recomb,
                        no_fft=arguments['--nofft'],
                        couplings_file=arguments['--params'],  # only for DCA
                        threads=threads  # only for DCA
                    )
                    for pred in predictions:
                        predictions_total.append(pred)  # if array gets too large?
                predictions_total = list(dict.fromkeys(predictions_total))  # removing duplicates from list
                if arguments['--negative']:
                    predictions_total = sorted(predictions_total, key=lambda x: x[0], reverse=False)
                else:
                    predictions_total = sorted(predictions_total, key=lambda x: x[0], reverse=True)

                predictions_out(
                    predictions=predictions_total,
                    model=arguments['--model'],
                    prediction_set=f'Top{args[1:-1]}',
                    path=path_recomb
                )

        elif arguments['extrapolation']:
            performance_mutation_extrapolation(
                encoded_csv=arguments['--input'],
                cv_regressor=arguments['--regressor'],
                conc=arguments['--conc']
            )

        elif arguments['low_n']:
            low_n(
                encoded_csv=arguments['--input'],
                cv_regressor=arguments['--regressor']
            )
