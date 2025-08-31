# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF


import logging
logger = logging.getLogger('pypef.utils.utils_run')

import os
import numpy as np
import re

from pypef.utils.variant_data import (
    amino_acids, generate_dataframe_and_save_csv,
    get_basename, read_csv_and_shift_pos_ints,
    get_seqs_from_var_name, get_wt_sequence
)
from pypef.utils.learning_test_sets import (
    csv_input, drop_rows, get_variants, make_sub_ls_ts,
    make_sub_ls_ts_randomly, make_fasta_ls_ts
)
from pypef.utils.prediction_sets import (
    make_fasta_ps, make_recombinations_double, make_recombinations_triple,
    make_recombinations_quadruple, make_recombinations_quintuple,
    create_split_files, make_combinations_double_all_diverse,
    make_combinations_triple_all_diverse, make_combinations_quadruple_all_diverse,
    make_ssm_singles
)
from pypef.utils.split import DatasetSplitter
from pypef.utils.directed_evolution import DirectedEvolution
from pypef.utils.sto2a2m import convert_sto2a2m
from pypef.utils.ssm import SSM
from pypef.ml.regression import OneHotEncoding, AAIndexEncoding, full_aaidx_txt_path
from pypef.hybrid.hybrid_model import plmc_or_gremlin_encoding


def run_pypef_utils(arguments):
    if arguments['mklsts']:
        wt_sequence = get_wt_sequence(arguments['--wt'])
        t_drop = float(arguments['--drop'])
        ls_proportion = arguments['--ls_proportion']
        logger.info(f'Length of provided sequence: {len(wt_sequence)} amino acids.')
        if True in [
            arguments['--random'], arguments['--modulo'], 
            arguments['--cont'], arguments['--plot']
        ]:
            logger.info(f'Ignoring set proportion (--ls_proportion).')
        else:
            logger.info(f'Training set proportion (--ls_proportion): {ls_proportion}.')
            
        df = drop_rows(arguments['--input'], amino_acids, t_drop, 
                       arguments['--sep'], arguments['--mutation_sep'])
        no_rnd = arguments['--numrnd']
        single_variants, single_values, higher_variants, higher_values = get_variants(
            df, amino_acids, wt_sequence, arguments['--mutation_sep']
        )
        if len(single_variants) == 0:
            logger.info('Found no single substitution variants for possible recombination!')

        if True in [
            arguments['--random'], arguments['--modulo'], 
            arguments['--cont'], arguments['--plot']
        ]:
            ds = DatasetSplitter(df, mutation_separator=arguments['--mutation_sep'])
            if arguments['--random']:
                train_data, test_data = ds.get_random_df_split_data(include_multis=True)
                cv_technique = 'random'
            elif arguments['--modulo']:
                train_data, test_data = ds.get_modulo_df_split_data()
                cv_technique = 'modulo'
            elif arguments['--cont']:
                train_data, test_data = ds.get_continuous_df_split_data()
                cv_technique = 'continuous'
            elif arguments['--plot']:
                ds.print_shapes()
                ds.plot_distributions() 
            if not arguments['--plot']:
                for i_cv, (train_set, test_set) in enumerate(zip(train_data, test_data)):
                    (
                        single_variants_train, single_values_train, 
                        multi_variants_train, multi_values_train
                    ) = get_variants(
                        train_set, amino_acids, wt_sequence, 
                        arguments['--mutation_sep'], verbose=False
                    )
                    (
                        single_variants_test, single_values_test, 
                        multi_variants_test, multi_values_test
                    ) = get_variants(
                        test_set, amino_acids, wt_sequence, 
                        arguments['--mutation_sep'], verbose=False
                    )
                    make_fasta_ls_ts(
                        f'LS_{cv_technique}_{i_cv + 1 }.fasl', wt_sequence, 
                        single_variants_train + multi_variants_train, 
                        single_values_train + multi_values_train
                    )
                    make_fasta_ls_ts(
                        f'TS_{cv_technique}_{i_cv + 1 }.fasl', wt_sequence, 
                        single_variants_test + multi_variants_test, 
                        single_values_test + multi_values_test
                    )
        else:
            sub_ls, val_ls, sub_ts, val_ts = make_sub_ls_ts(
                single_variants, single_values, 
                higher_variants, higher_values, 
                ls_proportion
            )
            logger.info('Tip: You can edit your LS and TS datasets just by '
                        'cutting/pasting between the LS and TS fasta datasets.')
            make_fasta_ls_ts('LS.fasl', wt_sequence, sub_ls, val_ls)
            make_fasta_ls_ts('TS.fasl', wt_sequence, sub_ts, val_ts)
            try:
                no_rnd = int(no_rnd)
            except ValueError:
                no_rnd = 0
            if no_rnd != 0:
                random_set_counter = 1
                no_rnd = int(no_rnd)
                while random_set_counter <= no_rnd:
                    sub_ls, val_ls, sub_ts, val_ts = make_sub_ls_ts_randomly(
                        single_variants, single_values,
                        higher_variants, higher_values,
                        ls_proportion
                    )
                    make_fasta_ls_ts('LS_default_random_' + str(random_set_counter) + '.fasl', wt_sequence, sub_ls, val_ls)
                    make_fasta_ls_ts('TS_default_random_' + str(random_set_counter) + '.fasl', wt_sequence, sub_ts, val_ts)
                    random_set_counter += 1

    elif arguments['mkps']:
        wt_sequence = get_wt_sequence(arguments['--wt'])
        if not arguments['--ssm']:
            try:
                csv_file = csv_input(arguments['--input'])
            except FileNotFoundError:
                raise SystemError("If creating prediction sets ('mkps') a CSV input is "
                                  "required (if not running 'mkps --ssm').")
            t_drop = float(arguments['--drop'])
            df = drop_rows(csv_file, amino_acids, t_drop)
            drop_wt = []
            for i in range(len(df)):
                if df.iloc[i, 0] == 'WT':
                    logger.info('Dropping wild-type (WT) from DataFrame as it cannot be used for (re-)combination.')
                    drop_wt.append(i)
            df = df.drop(drop_wt).reset_index(drop=True)

            logger.info(f'Length of provided sequence: {len(wt_sequence)} amino acids.')
            single_variants, _, higher_variants, _ = get_variants(df, amino_acids, wt_sequence)
            logger.info(f'Using single substitution variants for (re-)combination. '
                        f'Number of single variants: {len(single_variants)}.')
            if len(single_variants) == 0:
                logger.info('Found NO single substitution variants for possible recombination! '
                            'No prediction files can be created!')

        if arguments['--drecomb']:
            logger.info('Creating Recomb_Double_Split...')
            for no, files in enumerate(make_recombinations_double(single_variants)):
                double_mutants = np.array(files)
                create_split_files(double_mutants, single_variants, wt_sequence, 'Recomb_Double', no)

        if arguments['--trecomb']:
            logger.info('Creating Recomb_Triple_Split...')
            for no, files in enumerate(make_recombinations_triple(single_variants)):
                triple_mutants = np.array(files)
                create_split_files(triple_mutants, single_variants, wt_sequence, 'Recomb_Triple', no)

        if arguments['--qarecomb']:
            logger.info('Beware that this step might require much disk space as PyPEF is '
                        'creating prediction files in TXT format. Creating Recomb_Quadruple_Split...')
            for no, files in enumerate(make_recombinations_quadruple(single_variants)):
                quadruple_mutants = np.array(files)
                create_split_files(quadruple_mutants, single_variants, wt_sequence, 'Recomb_Quadruple', no)

        if arguments['--qirecomb']:
            logger.info('Beware that this step might require much disk space as PyPEF is '
                        'creating prediction files in plain text format. Creating Recomb_Quintuple_Split...')
            for no, files in enumerate(make_recombinations_quintuple(single_variants)):
                quintuple_mutants = np.array(files)
                create_split_files(quintuple_mutants, single_variants, wt_sequence, 'Recomb_Quintuple', no)

        if arguments['--ddiverse']:
            logger.info('Creating Diverse_Double_Split...')
            # if functions required, uncomment the next two lines and comment the other ones
            # for no, files in enumerate(
            #     make_recombinations_double_all_diverse_and_all_positions(wt_sequence, amino_acids)):
            for no, files in enumerate(make_combinations_double_all_diverse(single_variants, amino_acids)):
                doubles = np.array(files)
                create_split_files(doubles, single_variants, wt_sequence, 'Diverse_Double', no + 1)

        if arguments['--tdiverse']:
            logger.info('Beware that this step might require much disk space as PyPEF is '
                        'creating prediction files in plain text format. Creating Diverse_Triple_Split... ')
            for no, files in enumerate(make_combinations_triple_all_diverse(single_variants, amino_acids)):
                triples = np.array(files)
                create_split_files(triples, single_variants, wt_sequence, 'Diverse_Triple', no + 1)

        if arguments['--qdiverse']:
            logger.info('Beware that this step might require much disk space as PyPEF is '
                       'creating prediction files in plain text format. Creating Diverse_Quadruple_Split...')
            for no, files in enumerate(make_combinations_quadruple_all_diverse(single_variants, amino_acids)):
                quadruples = np.array(files)
                create_split_files(quadruples, single_variants, wt_sequence, 'Diverse_Quadruple', no + 1)

        if arguments['--ssm']:
            singles = make_ssm_singles(wt_sequence, amino_acids)
            make_fasta_ps('ssm_singles.fasta', wt_sequence, np.array(singles))

        if True not in [
            arguments['--drecomb'], arguments['--trecomb'], arguments['--qarecomb'],
            arguments['--qirecomb'], arguments['--ddiverse'], arguments['--tdiverse'],
            arguments['--qdiverse'], arguments['--ssm']
        ]:
            logger.info(f'Making prediction set fasta file from {csv_file}...\n')
            make_fasta_ps(
                filename=f'{get_basename(csv_file)}_prediction_set.fasta',
                wt=wt_sequence,
                substitutions=tuple(list(single_variants) + list(higher_variants))
            )

    # Metropolis-Hastings-driven directed evolution, similar to Biswas et al.:
    # Low-N protein engineering with data-efficient deep learning,
    # see https://github.com/ivanjayapurna/low-n-protein-engineering/tree/master/directed-evo
    elif arguments['directevo']:
        if arguments['hybrid'] or arguments['--encoding'] == 'dca':
            dca_encoder = arguments['--params']
            if arguments['ml']:
                ml_or_hybrid = 'ml'
            else:
                ml_or_hybrid = 'hybrid'
        else:
            dca_encoder = None
            ml_or_hybrid = 'ml'
        # Prediction using a saved model Pickle file specific AAindex used for encoding
        # Model saved in Pickle file also for DCA-based encoding, a default file name
        logger.info('Not counting WT as variant in directed evolution '
                    'as it cannot be used for (re-)combination.')
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
        s_wt = get_wt_sequence(arguments['--wt'])
        negative = arguments['--negative']
        # Metropolis-Hastings-driven directed evolution on single mutant position csv data
        usecsv = arguments['--usecsv']
        if usecsv:
            csv_file = csv_input(arguments['--input'])
            t_drop = float(arguments['--drop'])
            logger.info(f'Length of provided sequence: {len(s_wt)} amino acids.')
            df = drop_rows(csv_file, amino_acids, t_drop)
            drop_wt = []
            for i in range(len(df)):
                if df.iloc[i, 0] == 'WT':
                    drop_wt.append(i)
            df = df.drop(drop_wt).reset_index(drop=True)
            single_variants, single_values, higher_variants, higher_values = \
                get_variants(df, amino_acids, s_wt)
            logger.info(f'Number of single variants: {len(single_variants)}.')
            if len(single_variants) == 0:
                logger.info('Found NO single substitution variants for possible recombination!')
            single_vars, single_ys = list(single_variants), list(single_values)  # only tuples to lists

        else:
            single_vars = None  # What happens now? (Full diverse?)
        # Metropolis-Hastings-driven directed evolution on single mutant .csv amino acid substitution data
        csvaa = arguments['--csvaa']  # only use identified substitutions --> taken from CSV file
        logger.info('Running evolution trajectories and plotting...')
        DirectedEvolution(
            ml_or_hybrid=ml_or_hybrid,
            encoding=arguments['--encoding'],
            s_wt=s_wt,
            single_vars=single_vars,
            num_iterations=num_iterations,
            num_trajectories=num_trajectories,
            amino_acids=amino_acids,
            temp=temp,
            path=path,
            model=arguments['--model'],
            no_fft=arguments['--nofft'],
            dca_encoder=dca_encoder,
            usecsv=usecsv,
            csvaa=csvaa,
            negative=negative
        ).plot_trajectories()


    elif arguments['sto2a2m']:
        convert_sto2a2m(
            sto_file=arguments['--sto'],
            inter_gap=arguments['--inter_gap'],
            intra_gap=arguments['--intra_gap']
        )

    elif arguments['reformat_csv']:
        read_csv_and_shift_pos_ints(
            infile=arguments['--input'],
            offset=0,
            col_sep=arguments['--sep'],
            substitution_sep=arguments['--mutation_sep']
        )

    elif arguments['shift_pos']:
        read_csv_and_shift_pos_ints(
            infile=arguments['--input'],
            offset=arguments['--offset'],
            col_sep=arguments['--sep'],
            substitution_sep=arguments['--mutation_sep']
        )

    elif arguments['encode']:  # sole parallelized task for utils for DCA encoding
        df = drop_rows(arguments['--input'], amino_acids, arguments['--drop'])
        wt_sequence = get_wt_sequence(arguments['--wt'])
        logger.info(f'Length of provided sequence: {len(wt_sequence)} amino acids.')
        single_variants, single_values, higher_variants, higher_values = get_variants(
            df, amino_acids, wt_sequence)
        variants = list(single_variants) + list(higher_variants)
        ys_true = list(single_values) + list(higher_values)
        variants, ys_true, sequences = get_seqs_from_var_name(wt_sequence, variants, ys_true)
        assert len(variants) == len(ys_true) == len(sequences)
        logger.info('Encoding variant sequences...')

        if arguments['--encoding'] == 'dca':
            threads = abs(arguments['--threads']) if arguments['--threads'] is not None else 1
            threads = threads + 1 if threads == 0 else threads
            logger.info(f'Using {threads} thread(s) for running...')
            xs, variants, sequences, ys_true, x_wt, model, model_type = plmc_or_gremlin_encoding(
                variants=variants,
                sequences=sequences,
                ys_true=ys_true,
                params_file=arguments['--params'],
                substitution_sep=arguments['--mutation_sep'],
                threads=threads,
                verbose=True
            )
            assert len(xs) == len(variants) == len(ys_true)

            if variants[0][0] != variants[0][-1]:  # WT is required for DCA-based hybrid modeling
                y_wt = 1
                # better using re then: wt = variants[0][0] + str(variants[0][1:-1] + variants[0][0])
                wt = variants[0][0] + re.findall(r"\d+", variants[0])[0] + variants[0][0]
                variants = list(variants)
                variants.insert(0, wt)  # inserting WT at pos. 0
                xs = list(xs)
                xs.insert(0, list(x_wt.flatten()))
                ys_true = list(ys_true)
                ys_true.insert(0, y_wt)  # set WT fitness to 1 or use arguments y_wt?

        elif arguments['--encoding'] == 'onehot':
            onehot_encoder = OneHotEncoding(sequences)
            xs = onehot_encoder.collect_encoded_sequences()

        elif arguments['--encoding'] == 'aaidx':
            if arguments['--model'] is None:
                raise SystemError(
                    "Define the AAindex to use for encoding with the "
                    "flag --model AAINDEX, e.g.: --model CORJ870104."
                )
            aa_index_encoder = AAIndexEncoding(
                full_aaidx_txt_path(arguments['--model'] + '.txt'), sequences
            )
            x_fft, x_raw = aa_index_encoder.collect_encoded_sequences()
            if arguments['--nofft']:
                xs = x_raw
            else:
                xs = x_fft

        else:
            raise SystemError("Unknown encoding option.")
        logger.info(f'{len(variants)} variants (plus inserted WT) remained after encoding. '
                    f'Saving to encoding CSV file...')
        generate_dataframe_and_save_csv(  # put WT at pos. 0 for hybrid low_N or extrapolation
            variants=variants,
            sequence_encodings=xs,
            fitnesses=ys_true,
            csv_file=arguments['--input'],
            encoding_type=arguments['--encoding']
        )
    elif arguments['predict_ssm']:
        SSM(
            wt_seq=get_wt_sequence(arguments['--wt']),
            model=arguments['--llm'],
            pdb=arguments['--pdb'],
            params_file=arguments['--params']
        )
    else:
        raise RuntimeError("Unknown run option!")
