
import os
import sys
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from adjustText import adjust_text

# Add local PyPEF path if not using pip-installed PyPEF version
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pypef.dca.gremlin_inference import GREMLIN
from pypef.dca.hybrid_model import DCAHybridModel, get_delta_e_statistical_model, remove_gap_pos
from pypef.utils.variant_data import get_seqs_from_var_name

if not tf.config.list_physical_devices('GPU'):
    print('Using CPU for computations...')
else:
    print('Using GPU for computations... if facing an (out-of-memory) error, '
          'try reducing variable MAX_WT_SEQUENCE_LENGTH to e.g. 400...')

MAX_WT_SEQUENCE_LENGTH = 1000

single_point_mut_data = os.path.abspath(os.path.join(
    os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
higher_mut_data = os.path.abspath(os.path.join(
    os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))


def plot_performance(mut_data, plot_name, mut_sep=':'):
    tested_dsets = []
    dset_dca_perfs = []
    dset_hybrid_perfs = []
    n_tested_datasets = 0
    plt.figure(figsize=(40, 12))
    train_test_size_texts = []
    for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
        print(f'\n{i+1}/{len(mut_data.items())}\n'
              f'===============================================================')
        csv_path = dset_paths['CSV_path']
        msa_path = dset_paths['MSA_path']
        wt_seq = dset_paths['WT_sequence']
        msa_start = dset_paths['MSA_start']
        msa_end = dset_paths['MSA_end']
        wt_seq = wt_seq[msa_start - 1:msa_end]
        print('CSV path:', csv_path)
        print('MSA path:', msa_path)
        print('MSA start:', msa_start, '- MSA end:', msa_end)
        print('WT sequence (trimmed from MSA start to MSA end):\n' + wt_seq)
        # Getting % usage of virtual_memory (3rd field)
        #import psutil;print(f'RAM used: {round(psutil.virtual_memory()[3]/1E9, 3)} '
        #      f'GB ({psutil.virtual_memory()[2]} %)')
        variant_fitness_data = pd.read_csv(csv_path, sep=',')
        print('N_variant-fitness-tuples:', np.shape(variant_fitness_data)[0])
        #if np.shape(variant_fitness_data)[0] > 400000:
        #    print('More than 400000 variant-fitness pairs which represents a '
        #          'potential out-of-memory risk, skipping dataset...')
        #    continue
        variants = variant_fitness_data['mutant']
        fitnesses = variant_fitness_data['DMS_score']
        variants_split = []
        for variant in variants:
            # Split double and higher substituted variants to multiple single substitutions; 
            # e.g. separated by ':' or '/'
            variants_split.append(variant.split(mut_sep))
        variants, fitnesses, sequences = get_seqs_from_var_name(
            wt_seq, variants_split, fitnesses, shift_pos=msa_start - 1)
        # Only model sequences with length of max. 800 amino acids to avoid out of memory errors 
        print('Sequence length:', len(wt_seq))
        if len(wt_seq) > MAX_WT_SEQUENCE_LENGTH:
            print(f'Sequence length over {MAX_WT_SEQUENCE_LENGTH}, which represents a potential out-of-memory risk '
                  f'(when running on GPU, set threshold to length ~400 dependent on available VRAM), '
                  f'skipping dataset...')
            continue
        gremlin = GREMLIN(alignment=msa_path, wt_seq=wt_seq, opt_iter=100, max_msa_seqs=10000)
        gaps_1_indexed = gremlin.gaps_1_indexed
        count_gap_variants = 0
        n_muts = []
        for variant in variants_split:
            n_muts.append(len(variant))
            for substitution in variant:
                if int(substitution[1:-1]) in gaps_1_indexed:
                    count_gap_variants += 1
                    break
        max_muts = max(n_muts)
        print(f'N max. (multiple) amino acid substitutions: {max_muts}')
        ratio_input_vars_at_gaps = count_gap_variants / len(variants)
        if count_gap_variants > 0:
            print(f'{int(count_gap_variants)} of {len(variants)} ({ratio_input_vars_at_gaps * 100:.2f} %) input '
                  f'variants to be predicted are variants with amino acid substitutions at gap ' 
                  f'positions (these variants will be predicted/labeled with a fitness of 0.0).\n'
                  f'Gap positions (1-indexed): {gaps_1_indexed}')
        if ratio_input_vars_at_gaps >= 1.0:
            print(f'Gap positions (1-indexed): {gaps_1_indexed}\n'
                  f'100% substitutions at gap positions, skipping dataset...')
            continue
        # gaps = gremlin.gaps
        #variants, sequences, fitnesses = remove_gap_pos(gaps, variants, sequences, fitnesses)
        x_dca = gremlin.collect_encoded_sequences(sequences)
        x_wt = gremlin.x_wt
        # Statistical model performance
        y_pred = get_delta_e_statistical_model(x_dca, x_wt)
        print(f'Statistical DCA model performance on all {len(fitnesses)} datapoints; Spearman\'s rho: '
              f'{abs(spearmanr(fitnesses, y_pred)[0]):.3f}')
        train_test_size_texts.append(plt.text(
            n_tested_datasets, 
            abs(spearmanr(fitnesses, y_pred)[0]), 
            f'0' + r'$\rightarrow$' + f'{len(fitnesses)}', 
            color='tab:blue', size=4, ha='right'
        ))
        assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)
        hybrid_perfs = []
        for i_t, train_size in enumerate([25, 50, 75, 100, 200]):
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_dca, fitnesses, train_size=train_size, random_state=42)
                hybrid_model = DCAHybridModel(x_train=x_train, y_train=y_train, x_wt=x_wt)
                beta_1, beta_2, regressor = hybrid_model.settings(x_train=x_train, y_train=y_train)
                y_test_pred = hybrid_model.hybrid_prediction(
                    x=x_test, reg=regressor, beta_1=beta_1, beta_2=beta_2)
                print(f'Hybrid DCA model performance on {len(y_test)} datapoints (Train size: {train_size}: '
                      f'N Train: {len(y_train)}, N Test: {len(y_test)}). Spearman\'s rho: '
                      f'{abs(spearmanr(y_test, y_test_pred)[0]):.3f}')
                hybrid_perfs.append(abs(spearmanr(y_test, y_test_pred)[0]))
                train_test_size_texts.append(
                    plt.text(n_tested_datasets, 
                             abs(spearmanr(y_test, y_test_pred)[0]), 
                             f'{len(y_train)}'  + r'$\rightarrow$' + f'{len(y_test)}', 
                             color=['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'][i_t],
                             size=4, ha='right'
                    )
                )
            except ValueError:
                hybrid_perfs.append(np.nan)
        n_tested_datasets += 1
        tested_dsets.append(f'({n_tested_datasets}) {dset_key} '
                            f'({len(variants)}, {100.0 - (ratio_input_vars_at_gaps * 100):.2f}%, {max_muts})')
        dset_dca_perfs.append(abs(spearmanr(fitnesses, y_pred)[0]))
        dset_hybrid_perfs.append(hybrid_perfs)
        #import gc;gc.collect()  # Potentially GC is needed to free some RAM (deallocated VRAM -> partly stored in RAM?) after each run
    plt.plot(range(len(tested_dsets)), dset_dca_perfs, 'o--', markersize=8, color='tab:blue', label='DCA')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_dca_perfs)), color='tab:blue', linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_dca_perfs), f'{np.nanmean(dset_dca_perfs):.2f}', color='tab:blue'))

    plt.plot(range(len(tested_dsets)), np.array(dset_hybrid_perfs)[:, 0], 'o--', markersize=8, color='tab:orange', label='Hybrid (25)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(np.array(dset_hybrid_perfs)[:, 0])), color='tab:orange', linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(np.array(dset_hybrid_perfs)[:, 0]), f'{np.nanmean(np.array(dset_hybrid_perfs)[:, 0]):.2f}', color='tab:orange'))

    plt.plot(range(len(tested_dsets)), np.array(dset_hybrid_perfs)[:, 1], 'o--', markersize=8, color='tab:green', label='Hybrid (50)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(np.array(dset_hybrid_perfs)[:, 1])), color='tab:green', linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(np.array(dset_hybrid_perfs)[:, 1]), f'{np.nanmean(np.array(dset_hybrid_perfs)[:, 1]):.2f}', color='tab:green'))

    plt.plot(range(len(tested_dsets)), np.array(dset_hybrid_perfs)[:, 2], 'o--', markersize=8, color='tab:red',  label='Hybrid (75)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(np.array(dset_hybrid_perfs)[:, 2])), color='tab:red', linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(np.array(dset_hybrid_perfs)[:, 2]), f'{np.nanmean(np.array(dset_hybrid_perfs)[:, 2]):.2f}', color='tab:red'))

    plt.plot(range(len(tested_dsets)), np.array(dset_hybrid_perfs)[:, 3], 'o--', markersize=8, color='tab:purple', label='Hybrid (100)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(np.array(dset_hybrid_perfs)[:, 3])), color='tab:purple', linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(np.array(dset_hybrid_perfs)[:, 3]), f'{np.nanmean(np.array(dset_hybrid_perfs)[:, 3]):.2f}', color='tab:purple'))

    plt.plot(range(len(tested_dsets)), np.array(dset_hybrid_perfs)[:, 4], 'o--', markersize=8, color='tab:brown', label='Hybrid (200)') 
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(np.array(dset_hybrid_perfs)[:, 4])), color='tab:brown', linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(np.array(dset_hybrid_perfs)[:, 4]), f'{np.nanmean(np.array(dset_hybrid_perfs)[:, 4]):.2f}', color='tab:brown'))

    plt.xticks(range(len(tested_dsets)), tested_dsets, rotation=45, ha='right')
    plt.margins(0.01)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.ylabel(r'|Spearmanr $\rho$|')
    adjust_text(train_test_size_texts, expand=(1.2, 2))
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__),  f'{plot_name}.png') + '.')


with open(single_point_mut_data, 'r') as fh:
    s_mut_data = json.loads(fh.read())
with open(higher_mut_data, 'r') as fh:
    h_mut_data = json.loads(fh.read())
plot_performance(mut_data=s_mut_data, plot_name='single_point_mut_performance')
plot_performance(mut_data=h_mut_data, plot_name='multi_point_mut_performance')
