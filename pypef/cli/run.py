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

# Docstring essential for docopt arguments
"""
PyPEF - Pythonic Protein Engineering Framework.

Creation of learning and validation sets: to split .CSV data in learning and validation sets run
    pypef mklsvs [...]
Creation of prediction sets: To create prediction sets from CSV data single point mutational variants run
    pypef mkps [...]
Running:
 1. To train and validate models run
        pypef run -l Learning_Set.fasta -v Validation_Set.fasta [-s 5] [--parallel] [-c 4]
    ! Attention using ray for parallel computing ('--parallel') in Windows: Ray is not yet fully supported for Windows !
 2. To plot the validation creating a figure (.png) run
        pypef run -m MODEL12345 -f Validation_Set.fasta
 3. To predict variants run
        pypef run -m MODEL12345 -p Prediction_Set.fasta
    or for predicting variants in created prediction set folders exemplary run
        pypef run -m MODEL12345 --pmult [--drecomb] [...] [--qdiverse]
    or for performing in silico directed evolution run:
        pypef directevo -m MODEL12345 [...]


Usage:
    pypef mklsvs [--wtseq WT_SEQ] [--input CSV_FILE] [--drop THRESHOLD] [--nornd NUMBER]
    pypef mkps [--wtseq WT_SEQ] [--input CSV_FILE] [--drop THRESHOLD]
                                 [--drecomb] [--trecomb] [--qrecomb]
                                 [--ddiverse] [--tdiverse] [--qdiverse]
    pypef run --ls LEARNING_SET --vs VALIDATION_SET [--save NUMBER] [--regressor TYPE] [--nofft] [--all]
                                                       [--sort METRIC] [--parallel] [--cores NUMCORES]
    pypef --show [MODELS]
    pypef run --model MODEL12345 --figure VS_FOR_PLOTTING  [--label] [--color] [--ywt WT_FITNESS] [--nofft]
    pypef run --model MODEL12345 --ps PREDICTION_SET [--nofft] [--negative] [--print]
    pypef run --model MODEL12345 --pmult [--drecomb] [--trecomb] [--qrecomb]
                                          [--ddiverse] [--tdiverse] [--qdiverse] [--nofft] [--negative]
    pypef directevo --model MODEL12345 [--ywt WT_FITNESS] [--wtseq WT_SEQ]
                                        [--numiter NUM_ITER] [--numtraj NUM_TRAJ]
                                        [--temp TEMPERATURE] [--nofft] [--negative] [--print]
                                        [--usecsv] [--csvaa] [--input CSV_FILE] [--drop THRESHOLD]


Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --show                       Show achieved model performances from Model_Results.txt.
  MODELS                       Number of saved models to show [default: 5].
  -w --wtseq WT_SEQ            Input file (in .fa format) for wild-type sequence [default: None].
  -i --input CSV_FILE          Input data file in .csv format [default: None].
  -d --drop THRESHOLD          Below threshold variants will be discarded from the data [default: -9E09].
  -n --nornd NUMBER            Number of randomly created Learning and Validation datasets [default: 0].
  -s --save NUMBER             Number of models to be saved as pickle files [default: 5].
  --parallel                   Parallel computing of training and validation of models [default: False].
  -c --cores NUMCORES          Number of cores used in parallel computing.
  --drecomb                    Create/predict double recombinants [default: False].
  --trecomb                    Create/predict triple recombinants [default: False].
  --qrecomb                    Create/predict quadruple recombinants [default: False].
  --ddiverse                   Create/predict double natural diverse variants [default: False].
  --tdiverse                   Create/predict triple natural diverse variants [default: False].
  --qdiverse                   Create quadruple natural diverse variants [default: False].
  -u --pmult                   Predict for all prediction files in folder for recombinants
                               or for diverse variants [default: False].
  --negative                   Set if more negative values define better variants [default: False].
  -l --ls LEARNING_SET         Input learning set in .fasta format.
  -v --vs VALIDATION_SET       Input validation set in .fasta format.
  --regressor TYPE             Type of regression (R.) to use, options: PLS CV R.: pls_cv, PLS LOOCV R.: pls,
                               Random Forest CV R.: rf, SVM CV R.: svr, MLP CV R.: mlp, Ridge CV R.: ridge,
                               LassoLars CV R.: lassolars [default: pls_cv].
  --nofft                      Raw sequence input, i.e., no FFT for establishing protein spectra
                               as vector inputs [default: False].
  --all                        Finally training on all data [default: False]
  --sort METRIC                Rank models based on metric {1: R^2, 2: RMSE, 3: NRMSE, 4: Pearson's r,
                               5: Spearman's rho} [default: 1].
  -m --model MODEL12345        Model (pickle file) for plotting of validation or for performing predictions.
  -f --figure VS_FOR_PLOTTING  Validation set for plotting using a trained model.
  --label                      Label the plot instances [default: False].
  --color                      Color the plot for "true" and "false" predictions quarters [default: False].
  -p --ps PREDICTION_SET       Prediction set for performing predictions using a trained Model.
  --print                      Print raw encoded and FFT-ed sequence matrices and predictions in Shell [default: False].
  -y --ywt WT_FITNESS          Fitness value (y) of wild-type.
  --numiter NUM_ITER           Number of mutation iterations per evolution trajectory [default: 5].
  --numtraj NUM_TRAJ           Number of trajectories, i.e., evolution pathways [default: 5].
  --temp TEMPERATURE           "Temperature" of Metropolis-Hastings criterion [default: 0.01]
  --usecsv                     Perform directed evolution on single variant csv position data [default: False].
  --csvaa                      Directed evolution csv amino acid substitutions,
                               requires flag "--usecsv" [default: False].
"""

# standard import, for all required modules see requirements.txt file(s)
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
from docopt import docopt
import multiprocessing
# ray imported later locally as is is only used for parallelized running, thus commented out:
# import ray

# importing own modules
from pypef.cli.regression import (
    read_models, formatted_output, r2_list, save_model, predict,
    predictions_out, plot
)
from pypef.cli.learning_validation_sets import (
    get_wt_sequence, csv_input, drop_rows, get_variants, make_sub_ls_vs,
    make_sub_ls_vs_randomly, make_fasta_ls_vs
)
from pypef.cli.prediction import (
    make_combinations_double, make_combinations_triple, make_combinations_quadruple,
    create_split_files, make_combinations_double_all_diverse,
    make_combinations_triple_all_diverse, make_combinations_quadruple_all_diverse
)
from pypef.cli.directed_evolution import run_de_trajectories
# import Modules_Parallelization.r2_list_parallel locally to avoid error
# when not running in parallel, thus commented out:
# from pypef.cli.parallelization import r2_list_parallel


def run_pypef():
    """
    Running the program, importing all required self-made modules and
    running them dependent on user-passed input arguments using docopt
    for argument parsing.
    """
    arguments = docopt(__doc__, version='PyPEF 0.1.7 (November 2021)')
    # print(arguments)  # uncomment for printing parsed docopt arguments
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

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

    if arguments['mklsvs']:
        wt_sequence = get_wt_sequence(arguments['--wtseq'])
        csv_file = csv_input(arguments['--input'])
        t_drop = float(arguments['--drop'])

        print('Length of provided sequence: {} amino acids.'.format(len(wt_sequence)))
        df = drop_rows(csv_file, amino_acids, t_drop)
        no_rnd = arguments['--nornd']

        single_variants, single_values, higher_variants, higher_values = get_variants(df, amino_acids, wt_sequence)
        print('Number of single variants: {}.'.format(len(single_variants)))
        if len(single_variants) == 0:
            print('Found NO single substitution variants for possible recombination!')
        sub_ls, val_ls, sub_vs, val_vs = make_sub_ls_vs(single_variants, single_values, higher_variants, higher_values)
        print('Tip: You can edit your LS and VS datasets just by cutting/pasting between the LS and VS fasta datasets.')

        print('Creating LS dataset...', end='\r')
        make_fasta_ls_vs('LS.fasta', wt_sequence, sub_ls, val_ls)
        print('Creating VS dataset...', end='\r')
        make_fasta_ls_vs('VS.fasta', wt_sequence, sub_vs, val_vs)

        try:
            no_rnd = int(no_rnd)
        except ValueError:
            no_rnd = 0
        if no_rnd != 0:
            random_set_counter = 1
            no_rnd = int(no_rnd)
            while random_set_counter <= no_rnd:
                print('Creating random LV and VS No. {}...'.format(random_set_counter), end='\r')
                sub_ls, val_ls, sub_vs, val_vs = make_sub_ls_vs_randomly(
                    single_variants, single_values,
                    higher_variants, higher_values
                )
                make_fasta_ls_vs('LS_random_' + str(random_set_counter) + '.fasta', wt_sequence, sub_ls, val_ls)
                make_fasta_ls_vs('VS_random_' + str(random_set_counter) + '.fasta', wt_sequence, sub_vs, val_vs)
                random_set_counter += 1
        print('\n\nDone!\n')

    elif arguments['mkps']:
        wt_sequence = get_wt_sequence(arguments['--wtseq'])
        csv_file = csv_input(arguments['--input'])
        t_drop = float(arguments['--drop'])
        df = drop_rows(csv_file, amino_acids, t_drop)
        drop_wt = []
        for i in range(len(df)):
            if df.iloc[i, 0] == 'WT':
                print('Dropping wild-type (WT) from DataFrame as it cannot be used for (re-)combination.')
                drop_wt.append(i)
        df = df.drop(drop_wt).reset_index(drop=True)

        print('Length of provided sequence: {} amino acids.'.format(len(wt_sequence)))
        single_variants, _, higher_variants, _ = get_variants(df, amino_acids, wt_sequence)
        print('Using single substitution variants for (re-)combination. '
              'Number of single variants: {}.'.format(len(single_variants)))
        no_done = False
        if len(single_variants) == 0:
            print('Found NO single substitution variants for possible recombination! '
                  'No prediction files can be created!')
            no_done = True

        if arguments['--drecomb']:
            print('Creating Recomb_Double_Split...')
            for no, files in enumerate(make_combinations_double(single_variants)):
                double_mutants = np.array(files)
                create_split_files(double_mutants, single_variants, wt_sequence, 'Recomb_Double', no)

        if arguments['--trecomb']:
            print('Creating Recomb_Triple_Split...')
            for no, files in enumerate(make_combinations_triple(single_variants)):
                triple_mutants = np.array(files)
                create_split_files(triple_mutants, single_variants, wt_sequence, 'Recomb_Triple', no)

        if arguments['--qrecomb']:
            print('Beware that this step might require much disk space as PyPEF is '
                  'creating prediction files in TXT format. Creating Recomb_Quadruple_Split...')
            for no, files in enumerate(make_combinations_quadruple(single_variants)):
                quadruple_mutants = np.array(files)
                create_split_files(quadruple_mutants, single_variants, wt_sequence, 'Recomb_Quadruple', no)

        if arguments['--ddiverse']:
            print('Creating Diverse_Double_Split...')
            for no, files in enumerate(make_combinations_double_all_diverse(single_variants, amino_acids)):
                doubles = np.array(files)
                create_split_files(doubles, single_variants, wt_sequence, 'Diverse_Double', no + 1)

        if arguments['--tdiverse']:
            print('Beware that this step might require much disk space as PyPEF is '
                  'creating prediction files in TXT format. Creating Diverse_Triple_Split... ')
            for no, files in enumerate(make_combinations_triple_all_diverse(single_variants, amino_acids)):
                triples = np.array(files)
                create_split_files(triples, single_variants, wt_sequence, 'Diverse_Triple', no + 1)

        if arguments['--qdiverse']:
            print('Beware that this step might require much disk space as PyPEF is '
                  'creating prediction files in TXT format. Creating Diverse_Quadruple_Split...')
            for no, files in enumerate(make_combinations_quadruple_all_diverse(single_variants, amino_acids)):
                quadruples = np.array(files)
                create_split_files(quadruples, single_variants, wt_sequence, 'Diverse_Quadruple', no + 1)

        if arguments['--drecomb'] is False and arguments['--trecomb'] is False \
                and arguments['--qrecomb'] is False and arguments['--ddiverse'] is False \
                and arguments['--tdiverse'] is False and arguments['--qdiverse'] is False:
            print('\nInput Error:\nAt least one specification needed: Specify recombinations for mkps ; '
                  'e.g. try: "pypef mkps --drecomb" for performing double recombinant Prediction set.\n')
            no_done = True

        if no_done is False:
            print('\nDone!\n')

    elif arguments['run']:
        if arguments['--ls'] is not None and arguments['--vs'] is not None:
            if arguments['--model'] is None and arguments['--figure'] is None:
                path = os.getcwd()
                try:
                    t_save = int(arguments['--save'])
                except ValueError:
                    t_save = 5
                if arguments['--parallel']:
                    # Parallelization of AAindex iteration
                    # import parallel modules here as ray is yet not supported for Windows
                    import ray
                    ray.init()
                    from pypef.cli.parallelization import r2_list_parallel
                    cores = arguments['--cores']
                    try:
                        cores = int(cores)
                    except (ValueError, TypeError):
                        try:
                            cores = multiprocessing.cpu_count() // 2
                        except NotImplementedError:
                            cores = 4
                    print('Using {} cores for parallel computing. Running...'.format(cores))
                    aaindex_r2_list = r2_list_parallel(
                        arguments['--ls'], arguments['--vs'], cores,
                        arguments['--regressor'], arguments['--nofft'],
                        arguments['--sort']
                    )
                    formatted_output(aaindex_r2_list, arguments['--nofft'])
                    save_model(
                        path, aaindex_r2_list, arguments['--ls'], arguments['--vs'], t_save,
                        arguments['--regressor'], arguments['--nofft'], arguments['--all']
                    )

                else:
                    aaindex_r2_list = r2_list(
                        arguments['--ls'], arguments['--vs'], arguments['--regressor'],
                        arguments['--nofft'], arguments['--sort']
                    )
                    formatted_output(aaindex_r2_list, arguments['--nofft'])
                    save_model(
                        path, aaindex_r2_list, arguments['--ls'], arguments['--vs'], t_save,
                        arguments['--regressor'], arguments['--nofft'], arguments['--all']
                    )
                print('\nDone!\n')

        elif arguments['--figure'] is not None and arguments['--model'] is not None:
            path = os.getcwd()
            plot(
                path, arguments['--figure'], arguments['--model'], arguments['--label'],
                arguments['--color'], arguments['--ywt'], arguments['--nofft']
            )
            print('\nCreated plot!\n')

        # Prediction of single .fasta file
        elif arguments['--ps'] is not None and arguments['--model'] is not None:
            path = os.getcwd()
            predictions = predict(path, arguments['--ps'], arguments['--model'], None, arguments['--nofft'])
            if arguments['--negative']:
                predictions = sorted(predictions, key=lambda x: x[0], reverse=False)
            predictions_out(predictions, arguments['--model'], arguments['--ps'])
            print('\nDone!\n')

        # Prediction on recombinant/diverse variant folder data
        elif arguments['--pmult'] and arguments['--model'] is not None:
            path = os.getcwd()
            recombs_total = []
            recomb_d, recomb_t, recomb_q = '/Recomb_Double_Split/', '/Recomb_Triple_Split/', '/Recomb_Quadruple_Split/'
            diverse_d, diverse_t, diverse_q = '/Diverse_Double_Split/', '/Diverse_Triple_Split/', \
                                              '/Diverse_Quadruple_Split/'
            if arguments['--drecomb']:
                recombs_total.append(recomb_d)
            if arguments['--trecomb']:
                recombs_total.append(recomb_t)
            if arguments['--qrecomb']:
                recombs_total.append(recomb_q)
            if arguments['--ddiverse']:
                recombs_total.append(diverse_d)
            if arguments['--tdiverse']:
                recombs_total.append(diverse_t)
            if arguments['--qdiverse']:
                recombs_total.append(diverse_q)
            if arguments['--drecomb'] is False and arguments['--trecomb'] is False and arguments['--qrecomb'] is False \
                    and arguments['--ddiverse'] is False and arguments['--tdiverse'] is False \
                    and arguments['--qdiverse'] is False:
                print('Define prediction target for --pmult, e.g. --pmult --drecomb.')

            for args in recombs_total:
                predictions_total = []
                print('Running predictions for files in {}...'.format(args[1:-1]))
                path_recomb = path + args
                os.chdir(path)
                files = [f for f in listdir(path_recomb) if isfile(join(path_recomb, f)) if f.endswith('.fasta')]
                for f in tqdm(files):
                    predictions = predict(path, f, arguments['--model'], path_recomb, arguments['--nofft'])
                    for pred in predictions:
                        predictions_total.append(pred)  # perhaps implement numpy.save if array gets too large byte size
                predictions_total = list(dict.fromkeys(predictions_total))  # removing duplicates from list
                if arguments['--negative']:
                    predictions_total = sorted(predictions_total, key=lambda x: x[0], reverse=False)

                else:
                    predictions_total = sorted(predictions_total, key=lambda x: x[0], reverse=True)

                predictions_out(predictions_total, arguments['--model'], 'Top' + args[1:-1])
                os.chdir(path)
            print('\nDone!\n')

    # Metropolis-Hastings-driven directed evolution, similar to Biswas et al.:
    # Low-N protein engineering with data-efficient deep learning,
    # see https://github.com/ivanjayapurna/low-n-protein-engineering/tree/master/directed-evo
    elif arguments['directevo']:
        if arguments['--model'] is not None:
            print('Not counting WT as variant in directed evolution as it cannot be used for (re-)combination.')
            path = os.getcwd()
            try:
                # "temperature" parameter: determines sensitivity of Metropolis-Hastings acceptance criteria
                temp = float(arguments['--temp'])
                # how many subsequent mutation trials per simulated evolution trajectory
                num_iterations = int(arguments['--numiter'])
                # how many separate evolution trajectories to run
                num_trajectories = int(arguments['--numtraj'])
            except ValueError:
                raise ValueError("Define flags 'numiter' and 'numtraj' as integer and 'temp' as float.")

            args_model = arguments['--model']
            s_wt = get_wt_sequence(arguments['--wtseq'])
            y_wt = arguments['--ywt']
            negative = arguments['--negative']

            # Metropolis-Hastings-driven directed evolution on single mutant position csv data
            usecsv = arguments['--usecsv']
            if usecsv is True:
                csv_file = csv_input(arguments['--input'])
                t_drop = float(arguments['--drop'])

                print('Length of provided sequence: {} amino acids.'.format(len(s_wt)))
                df = drop_rows(csv_file, amino_acids, t_drop)
                drop_wt = []
                for i in range(len(df)):
                    if df.iloc[i, 0] == 'WT':
                        print('Using fitness value (y_WT) for wild-type (WT) as specified in CSV.')
                        drop_wt.append(i)
                        y_wt = df.iloc[i, 1]
                df = df.drop(drop_wt).reset_index(drop=True)

                single_variants, single_values, higher_variants, higher_values = get_variants(df, amino_acids, s_wt)
                print('Number of single variants: {}.'.format(len(single_variants)))
                if len(single_variants) == 0:
                    print('Found NO single substitution variants for possible recombination!')
                sub_ls, val_ls, _, _ = make_sub_ls_vs(
                    single_variants, single_values, higher_variants,
                    higher_values, directed_evolution=True
                )
                print('Creating single variant dataset...')

                make_fasta_ls_vs('Single_variants.fasta', s_wt, sub_ls, val_ls)

            else:
                sub_ls = None

            # Metropolis-Hastings-driven directed evolution on single mutant .csv amino acid substitution data
            csvaa = arguments['--csvaa']
            traj_records_folder = 'DE_record'

            print('Running evolution trajectories and plotting...')

            run_de_trajectories(
                s_wt, args_model, y_wt, num_iterations, num_trajectories,
                traj_records_folder, amino_acids, temp, path, sub_ls, arguments['--nofft'],
                negative=negative, save=True, usecsv=usecsv, csvaa=csvaa,
                print_matrix=arguments['--print']
            )
            print('\nDone!')


if __name__ == "__main__":
    run_pypef()
