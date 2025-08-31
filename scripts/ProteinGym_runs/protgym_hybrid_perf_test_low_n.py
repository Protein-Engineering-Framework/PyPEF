
import os
import copy
import gc
import time
import warnings
import psutil
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from Bio import SeqIO, BiopythonParserWarning
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)

import sys  # Use local directory PyPEF files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pypef.dca.gremlin_inference import GREMLIN
from pypef.llm.utils import get_batches
from pypef.llm.esm_lora_tune import (
    get_esm_models, esm_tokenize_sequences, 
    esm_train, esm_infer, corr_loss
)
from pypef.llm.prosst_lora_tune import (
    get_logits_from_full_seqs, get_prosst_models, get_structure_quantizied, 
    prosst_tokenize_sequences, prosst_train
)
from pypef.utils.variant_data import get_seqs_from_var_name
from pypef.utils.helpers import get_vram, get_device
from pypef.hybrid.hybrid_model import (
    DCALLMHybridModel, reduce_by_batch_modulo, get_delta_e_statistical_model
)


JUST_PLOT_RESULTS = False


def compute_performances(mut_data, mut_sep=':', start_i: int = 0, already_tested_is: list = []):
    # Get cpu, gpu or mps device for training.
    device = get_device()
    print(f"Using {device.upper()} device")
    get_vram()
    MAX_WT_SEQUENCE_LENGTH = 1000
    print(f"Maximum sequence length: {MAX_WT_SEQUENCE_LENGTH}")
    print(f"Loading LLM models into {device} device...")
    prosst_base_model, prosst_lora_model, prosst_tokenizer, prosst_optimizer = get_prosst_models()
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to(device)
    esm_base_model, esm_lora_model, esm_tokenizer, esm_optimizer = get_esm_models()
    esm_base_model = esm_base_model.to(device)
    get_vram()
    plt.figure(figsize=(40, 12))
    numbers_of_datasets = [i + 1 for i in range(len(mut_data.keys()))]
    for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
        if i >= start_i and i not in already_tested_is:  # i > 3 and i <21:  #i == 18 - 1:
            start_time = time.time()
            print(f'\n{i+1}/{len(mut_data.items())}\n'
                  f'===============================================================')
            hybrid_perfs = []
            csv_path = dset_paths['CSV_path']
            msa_path = dset_paths['MSA_path']
            wt_seq = dset_paths['WT_sequence']
            msa_start = dset_paths['MSA_start']
            msa_end = dset_paths['MSA_end']
            pdb = dset_paths['PDB_path']
            wt_seq = wt_seq[msa_start - 1:msa_end]
            print('CSV path:', csv_path)
            print('MSA path:', msa_path)
            print('MSA start:', msa_start, '- MSA end:', msa_end)
            print('WT sequence (trimmed from MSA start to MSA end):\n' + wt_seq)
            #if msa_start != 1:
            #    print('Continuing (TODO: requires cut of PDB input struture residues)...')
            #    continue
            # Getting % usage of virtual_memory (3rd field)
            print(f'RAM used: {round(psutil.virtual_memory()[3]/1E9, 3)} '
                  f'GB ({psutil.virtual_memory()[2]} %)')
            variant_fitness_data = pd.read_csv(csv_path, sep=',')
            print('N_variant-fitness-tuples:', np.shape(variant_fitness_data)[0])
            #if np.shape(variant_fitness_data)[0] > 400000:
            #    print('More than 400000 variant-fitness pairs which represents a '
            #          'potential out-of-memory risk, skipping dataset...')
            #    continue
            variants = variant_fitness_data['mutant'].to_numpy()
            variants_orig = variants
            fitnesses = variant_fitness_data['DMS_score'].to_numpy()
            if len(fitnesses) <= 50:
                print('Number of available variants <= 50, skipping dataset...')
                continue
            variants_split = []
            for variant in variants:
                # Split double and higher substituted variants to multiple single substitutions
                # e.g. separated by ':' or '/'
                variants_split.append(variant.split(mut_sep))
            variants, fitnesses, sequences = get_seqs_from_var_name(
                wt_seq, variants_split, fitnesses, shift_pos=msa_start - 1)
            # Only model sequences with length of max. 800 amino acids to avoid out of memory errors 
            print('Sequence length:', len(wt_seq))
            count_gap_variants = 0
            n_muts = []
            for variant in variants_split:
                n_muts.append(len(variant))
            max_muts = max(n_muts)
            print(f'N max. (multiple) amino acid substitutions: {max_muts}')
            if len(wt_seq) > MAX_WT_SEQUENCE_LENGTH:
                print(f'Sequence length over {MAX_WT_SEQUENCE_LENGTH}, which represents '
                      f'a potential out-of-memory risk (when running on GPU, set '
                      f'threshold to length ~400 dependent on available VRAM); '
                      f'skipping dataset...')
                with open(out_results_csv, 'a') as fh:
                    fh.write(
                        f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},'
                        f'{max_muts},Sequence too long ({len(wt_seq)} > {MAX_WT_SEQUENCE_LENGTH})\n'
                    )
                continue
            _ratio_input_vars_at_gaps = count_gap_variants / len(variants)
            pdb_seq = str(list(SeqIO.parse(pdb, "pdb-atom"))[0].seq)
            try:
                assert wt_seq == pdb_seq  # pdb_seq.startswith(wt_seq)
            except AssertionError:
                print(
                    f"Wild-type sequence is not matching PDB-extracted sequence:"
                    f"\nWT sequence:\n{wt_seq}\nPDB sequence:\n{pdb_seq}\nSkipping dataset..."
                )
                with open(out_results_csv, 'a') as fh:
                    fh.write(
                        f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},'
                        f'{max_muts},PDBseq neq WTseq\n'
                    )
                    continue
            
            print('GREMLIN-DCA: optimization...')
            gremlin = GREMLIN(alignment=msa_path, opt_iter=100, optimize=True)
            x_dca = gremlin.collect_encoded_sequences(sequences)
            x_wt = gremlin.x_wt
            y_pred_dca = get_delta_e_statistical_model(x_dca, x_wt)
            print(f'DCA (unsupervised performance): {spearmanr(fitnesses, y_pred_dca)[0]:.3f}') 
            dca_unopt_perf = spearmanr(fitnesses, y_pred_dca)[0]

            try:
                x_esm, esm_attention_mask = esm_tokenize_sequences(
                    sequences, esm_tokenizer, max_length=len(wt_seq))
                y_esm = esm_infer(
                    get_batches(x_esm, dtype=float, batch_size=1), 
                    esm_attention_mask, 
                    esm_base_model
                )
                print(f'ESM1v (unsupervised performance): '
                      f'{spearmanr(fitnesses, y_esm.cpu())[0]:.3f}')
                esm_unopt_perf = spearmanr(fitnesses, y_esm.cpu())[0]
            except RuntimeError:
                esm_unopt_perf = np.nan

            try:
                input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
                    pdb, prosst_tokenizer, wt_seq)
                x_prosst = prosst_tokenize_sequences(sequences=sequences, vocab=prosst_vocab)
                y_prosst = get_logits_from_full_seqs(
                        x_prosst, prosst_base_model, input_ids, prosst_attention_mask, 
                        structure_input_ids, train=False
                )
                print(f'ProSST (unsupervised performance): '
                      f'{spearmanr(fitnesses, y_prosst.cpu())[0]:.3f}')
                prosst_unopt_perf = spearmanr(fitnesses, y_prosst.cpu())[0]
            except RuntimeError:
                prosst_unopt_perf = np.nan
            
            if np.isnan(esm_unopt_perf) and np.isnan(prosst_unopt_perf):
                print('Both LLM\'s had RunTimeErrors, skipping dataset...')
                continue 

            ns_y_test = [len(variants)]
            for i_t, train_size in enumerate([100, 200, 1000]):
                prosst_lora_model_2 = copy.deepcopy(prosst_lora_model)
                prosst_optimizer = torch.optim.Adam(prosst_lora_model_2.parameters(), lr=0.0001)
                esm_lora_model_2 = copy.deepcopy(esm_lora_model)
                esm_optimizer = torch.optim.Adam(esm_lora_model_2.parameters(), lr=0.0001)
                print('\nTRAIN SIZE:', train_size, '\n-------------------------------------------\n')
                get_vram()
                try:
                    (
                        x_dca_train, x_dca_test, 
                        x_llm_train_prosst, x_llm_test_prosst,
                        x_llm_train_esm, x_llm_test_esm, 
                        y_train, y_test
                    ) = train_test_split(
                        x_dca,
                        x_prosst,
                        x_esm, 
                        fitnesses, 
                        train_size=train_size, 
                        random_state=42
                    )
                except ValueError as e:
                    print(f"Only {len(fitnesses)} variant-fitness pairs in total, "
                          f"cannot split the data in N_Train = {train_size} and N_Test "
                          f"(N_Total - N_Train) [Excepted error: {e}].")
                    for k in [np.nan, np.nan, np.nan]:
                        hybrid_perfs.append(k)
                    ns_y_test.append(np.nan)
                    continue
                (
                    x_dca_train, 
                    x_llm_train_prosst,
                    x_llm_train_esm, 
                    y_train,
                ) = (
                    reduce_by_batch_modulo(x_dca_train),  
                    reduce_by_batch_modulo(x_llm_train_prosst),
                    reduce_by_batch_modulo(x_llm_train_esm), 
                    reduce_by_batch_modulo(y_train),
                )
                llm_dict_prosst = {
                    'prosst': {
                        'llm_base_model': prosst_base_model,
                        'llm_model': prosst_lora_model_2,
                        'llm_optimizer': prosst_optimizer,
                        'llm_train_function': prosst_train,
                        'llm_inference_function': get_logits_from_full_seqs,
                        'llm_loss_function': corr_loss,
                        'x_llm' : x_llm_train_prosst,
                        'llm_attention_mask':  prosst_attention_mask,
                        'input_ids': input_ids,
                        'structure_input_ids': structure_input_ids
                    }
                }
                llm_dict_esm = {
                    'esm1v': {
                        'llm_base_model': esm_base_model,
                        'llm_model': esm_lora_model_2,
                        'llm_optimizer': esm_optimizer,
                        'llm_train_function': esm_train,
                        'llm_inference_function': esm_infer,
                        'llm_loss_function': corr_loss,
                        'x_llm' : x_llm_train_esm,
                        'llm_attention_mask':  esm_attention_mask
                    }
                }
                print(f'Train: {len(np.array(y_train))} --> Test: {len(np.array(y_test))}')
                if len(y_test) <= 50:
                    print(f"Only {len(fitnesses)} in total, splitting the data "
                          f"in N_Train = {len(y_train)} and N_Test = {len(y_test)} "
                          f"results in N_Test <= 50 variants - not getting "
                          f"performance for N_Train = {len(y_train)}...")
                    for k in [np.nan, np.nan, np.nan]:
                        hybrid_perfs.append(k)
                    ns_y_test.append(np.nan)
                    continue
                get_vram()
                for i_m, method in enumerate([None, llm_dict_esm, llm_dict_prosst]):
                    print('\n~~~ ' + ['DCA hybrid', 'DCA+ESM1v hybrid', 'DCA+ProSST hybrid'][i_m] + ' ~~~')
                    try:
                        hm = DCALLMHybridModel(
                            x_train_dca=np.array(x_dca_train), 
                            y_train=y_train,
                            llm_model_input=method,
                            x_wt=x_wt
                        )
                        y_test_pred = hm.hybrid_prediction(
                            x_dca=np.array(x_dca_test), 
                            x_llm=[
                                None, 
                                np.asarray(x_llm_test_esm), 
                                np.asarray(x_llm_test_prosst)
                            ][i_m]
                        )
                        print(f'Hybrid performance: {spearmanr(y_test, y_test_pred)[0]:.3f}')
                        hybrid_perfs.append(spearmanr(y_test, y_test_pred)[0])
                    except RuntimeError as e:  # modeling_prosst.py, line 920, in forward 
                        # or UnboundLocalError in prosst_lora_tune.py, line 167
                        hybrid_perfs.append(np.nan)
                ns_y_test.append(len(y_test_pred))
                del prosst_lora_model_2
                del esm_lora_model_2
                torch.cuda.empty_cache()
                gc.collect()

            dt = time.time() - start_time
            dset_hybrid_perfs_i = ''
            for hp in hybrid_perfs:
                dset_hybrid_perfs_i += f'{hp},'
            dset_ns_y_test_i = ''
            for ns_y_t in ns_y_test:
                dset_ns_y_test_i += f'{ns_y_t},'
            with open(out_results_csv, 'a') as fh:
                fh.write(
                    f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},{max_muts},{dca_unopt_perf},'
                    f'{esm_unopt_perf},{prosst_unopt_perf},{dset_hybrid_perfs_i}{dset_ns_y_test_i}{int(dt)}\n')
                

def plot_csv_data(csv, plot_name):
    train_test_size_texts = []
    df = pd.read_csv(csv, sep=',')  
    tested_dsets = df['No.']
    dset_dca_perfs = df['Untrained_Performance_DCA']
    dset_esm_perfs = df['Untrained_Performance_ESM1v']
    dset_prosst_perfs = df['Untrained_Performance_ProSST']
    dset_hybrid_perfs_dca_100 = df['Hybrid_DCA_Trained_Performance_100']
    dset_hybrid_perfs_dca_200 = df['Hybrid_DCA_Trained_Performance_200']
    dset_hybrid_perfs_dca_1000 = df['Hybrid_DCA_Trained_Performance_1000']
    dset_hybrid_perfs_dca_esm_100 = df['Hybrid_DCA_ESM1v_Trained_Performance_100']
    dset_hybrid_perfs_dca_esm_200 = df['Hybrid_DCA_ESM1v_Trained_Performance_200']
    dset_hybrid_perfs_dca_esm_1000 = df['Hybrid_DCA_ESM1v_Trained_Performance_1000']
    dset_hybrid_perfs_dca_prosst_100 = df['Hybrid_DCA_ProSST_Trained_Performance_100']
    dset_hybrid_perfs_dca_prosst_200 = df['Hybrid_DCA_ProSST_Trained_Performance_200']
    dset_hybrid_perfs_dca_prosst_1000 = df['Hybrid_DCA_ProSST_Trained_Performance_1000']

    blue_colors = mpl.colormaps['Blues'](np.linspace(0.3, 0.9, 4))
    red_colors = mpl.colormaps['Reds'](np.linspace(0.3, 0.9, 4))
    green_colors = mpl.colormaps['Greens'](np.linspace(0.3, 0.9, 4))

    plt.figure(figsize=(80, 12))
    plt.plot(range(len(tested_dsets)), dset_dca_perfs, 'o--', markersize=8, 
             color=blue_colors[0], label='DCA (0)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_dca_perfs)), 
             color=blue_colors[0], linestyle='--')
    for i, (p, n_test) in enumerate(zip(
        dset_dca_perfs, df['N_Y_test'].astype('Int64').to_list())):
        plt.text(i, 0.975, i, color='black', size=2)
        plt.text(i, 0.980, f'0'  + r'$\rightarrow$' + f'{n_test}', color='black', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_dca_perfs), 
                                          f'{np.nanmean(dset_dca_perfs):.2f}', color=blue_colors[0]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_100, 
             'o--', markersize=8, color=blue_colors[1], label='Hybrid (100)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_100)), 
             color=blue_colors[1], linestyle='--')
    for i, (p, n_test) in enumerate(zip(dset_hybrid_perfs_dca_100, df['N_Y_test_100'].astype('Int64').to_list())):
        plt.text(i, 0.985, f'100'  + r'$\rightarrow$' + f'{n_test}', color='black', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_100), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_100):.2f}', color=blue_colors[1]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_200, 
             'o--', markersize=8, color=blue_colors[2], label='Hybrid (200)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_200)), 
             color=blue_colors[2], linestyle='--')
    for i, (p, n_test) in enumerate(zip(
        dset_hybrid_perfs_dca_200, df['N_Y_test_200'].astype('Int64').to_list())):
       plt.text(i, 0.990, f'200'  + r'$\rightarrow$' + f'{n_test}', color='black', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_200), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_200):.2f}', color=blue_colors[2]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_1000, 
             'o--', markersize=8, color=blue_colors[3],  label='Hybrid (1000)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_1000)), 
             color=blue_colors[3], linestyle='--')
    for i, (p, n_test) in enumerate(zip(
        dset_hybrid_perfs_dca_1000, df['N_Y_test_1000'].astype('Int64').to_list())):
        plt.text(i, 0.995, f'1000'  + r'$\rightarrow$' + f'{n_test}', color='black', size=2)
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_1000), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_1000):.2f}', color=blue_colors[3]))


    plt.plot(range(len(tested_dsets)), dset_esm_perfs, 
             'o--', markersize=8, color=green_colors[0], label='ESM (0)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_esm_perfs)), 
             color=green_colors[0], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_esm_perfs), 
                                          f'{np.nanmean(dset_esm_perfs):.2f}', color=green_colors[0]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_esm_100, 
             'o--', markersize=8, color=green_colors[1], label='Hybrid (100)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_esm_100)), 
             color=green_colors[1], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_esm_100), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_esm_100):.2f}', color=green_colors[1]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_esm_200, 
             'o--', markersize=8, color=green_colors[2], label='Hybrid (200)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_esm_200)), 
             color=green_colors[2], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_esm_200), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_esm_200):.2f}', color=green_colors[2]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_esm_1000, 
             'o--', markersize=8, color=green_colors[3],  label='Hybrid (1000)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_esm_1000)), 
             color=green_colors[3], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_esm_1000), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_esm_1000):.2f}', color=green_colors[3]))


    plt.plot(range(len(tested_dsets)), dset_prosst_perfs, 
             'o--', markersize=8, color=red_colors[0], label='ProSST (0)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_prosst_perfs)), 
             color=red_colors[0], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_prosst_perfs), 
                                          f'{np.nanmean(dset_prosst_perfs):.2f}', color=red_colors[0]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_prosst_100, 
             'o--', markersize=8, color=red_colors[1], label='Hybrid (100)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_prosst_100)), 
             color=red_colors[1], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_prosst_100), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_prosst_100):.2f}', color=red_colors[1]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_prosst_200, 
             'o--', markersize=8, color=red_colors[2], label='Hybrid (200)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_prosst_200)), 
             color=red_colors[2], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_prosst_200), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_prosst_200):.2f}', color=red_colors[2]))

    plt.plot(range(len(tested_dsets)), dset_hybrid_perfs_dca_prosst_1000, 
             'o--', markersize=8, color=red_colors[3],  label='Hybrid (1000)')
    plt.plot(range(len(tested_dsets) + 1), np.full(len(tested_dsets) + 1, np.nanmean(dset_hybrid_perfs_dca_prosst_1000)), 
             color=red_colors[3], linestyle='--')
    train_test_size_texts.append(plt.text(len(tested_dsets), np.nanmean(dset_hybrid_perfs_dca_prosst_1000), 
                                          f'{np.nanmean(dset_hybrid_perfs_dca_prosst_1000):.2f}', color=red_colors[3]))


    plt.grid(zorder=-1)
    plt.xticks(
        range(len(tested_dsets)), 
        ['(' + str(n) + ') ' + name  for (n, name) in zip(tested_dsets, df['Dataset'].to_list())], 
        rotation=45, ha='right'
    )
    plt.margins(0.01)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.xlabel('Tested dataset')
    plt.ylabel(r'Spearman $\rho$')
    adjust_text(train_test_size_texts, expand=(1.2, 2))
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__),  f'{plot_name}.png') + '.')

    plt.clf()
    plt.figure(figsize=(24, 12))
    sns.set_style("whitegrid")
    df_ = df[[
        'Untrained_Performance_DCA',
        'Hybrid_DCA_Trained_Performance_100',
        'Hybrid_DCA_Trained_Performance_200',
        'Hybrid_DCA_Trained_Performance_1000',
        'Untrained_Performance_ESM1v',
        'Hybrid_DCA_ESM1v_Trained_Performance_100',
        'Hybrid_DCA_ESM1v_Trained_Performance_200',
        'Hybrid_DCA_ESM1v_Trained_Performance_1000',
        'Untrained_Performance_ProSST',
        'Hybrid_DCA_ProSST_Trained_Performance_100',
        'Hybrid_DCA_ProSST_Trained_Performance_200',
        'Hybrid_DCA_ProSST_Trained_Performance_1000',
        ]]
    print(df_)
    plot = sns.violinplot(
        df_, saturation=0.4, 
        palette=[blue_colors[0], blue_colors[1], blue_colors[2], blue_colors[3],
                 green_colors[0],green_colors[1], green_colors[2], green_colors[3],
                 red_colors[0], red_colors[1], red_colors[2], red_colors[3]]
    )
    plt.ylabel(r'Spearmanr $\rho$')
    sns.swarmplot(df_, color='black')
    dset_perfs = [
        dset_dca_perfs,
        dset_hybrid_perfs_dca_100,
        dset_hybrid_perfs_dca_200,
        dset_hybrid_perfs_dca_1000,
        dset_esm_perfs,
        dset_hybrid_perfs_dca_esm_100,
        dset_hybrid_perfs_dca_esm_200,
        dset_hybrid_perfs_dca_esm_1000,
        dset_prosst_perfs,
        dset_hybrid_perfs_dca_prosst_100,
        dset_hybrid_perfs_dca_prosst_200,
        dset_hybrid_perfs_dca_prosst_1000
    ]
    for n in range(0, len(dset_perfs)):
        plt.text(
            n + 0.15, -0.075, 
            r'$\overline{\rho}=$' + f'{np.nanmean(dset_perfs[n]):.3f}\n' 
            + r'$N_\mathrm{Datasets}=$' + f'{np.count_nonzero(~np.isnan(np.array(dset_perfs)[n]))}'
        )
    plot.set_xticks(range(len(plot.get_xticklabels())))
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.ylim(-0.09, 1.09)
    plt.margins(0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}_violin.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__), f'{plot_name}_violin.png') + '.')


if __name__ == '__main__':
    single_point_mut_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
    higher_mut_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))
    with open(single_point_mut_data, 'r') as fh:
        s_mut_data = json.loads(fh.read())
    with open(higher_mut_data, 'r') as fh:
        h_mut_data = json.loads(fh.read())
    combined_mut_data = s_mut_data.copy()
    combined_mut_data.update(h_mut_data)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    out_results_csv = os.path.join(os.path.dirname(__file__), 'results/dca_esm_and_hybrid_opt_results.csv')
    if os.path.exists(out_results_csv):
        print(f'\nReading existing file {out_results_csv}...')
        df = pd.read_csv(out_results_csv, sep=',')
        print(df)
        try:
            start_i = df['No.'].to_list()[-1]
            already_tested_is = [i - 1 for i in df['No.'].to_list()]
        except IndexError:
            start_i = 0
            already_tested_is = []
        print(list(combined_mut_data.keys())[start_i-1])
        print(f'Already tested datasets:')
        for i in already_tested_is:
            print(f'{i + 1} {list(combined_mut_data.keys())[i]}')
        try:
            print(f'\nContinuing getting model performances at {start_i + 1} '
                  f'{list(combined_mut_data.keys())[start_i]} '
                  f'(last tested dataset: {start_i}, {list(combined_mut_data.keys())[start_i - 1]})')
        except IndexError:
            print('\nComputed all results already?!')
    else:
        with open(out_results_csv, 'w') as fh:
            print(f'\nCreating new file {out_results_csv}...')
            fh.write(
                f'No.,Dataset,N_Variants,N_Max_Muts,Untrained_Performance_DCA,Untrained_Performance_ESM1v,'
                f'Untrained_Performance_ProSST,Hybrid_DCA_Trained_Performance_100,'
                f'Hybrid_DCA_ESM1v_Trained_Performance_100,Hybrid_DCA_ProSST_Trained_Performance_100,'
                f'Hybrid_DCA_Trained_Performance_200,Hybrid_DCA_ESM1v_Trained_Performance_200,'
                f'Hybrid_DCA_ProSST_Trained_Performance_200,Hybrid_DCA_Trained_Performance_1000,'
                f'Hybrid_DCA_ESM1v_Trained_Performance_1000,Hybrid_DCA_ProSST_Trained_Performance_1000,'
                f'N_Y_test,N_Y_test_100,N_Y_test_200,N_Y_test_1000,Time_in_s\n'
            )
            start_i = 0
            already_tested_is = []

    if not JUST_PLOT_RESULTS:
        compute_performances(
            mut_data=combined_mut_data, 
            start_i=start_i, 
            already_tested_is=already_tested_is
        )

    with open(out_results_csv, 'r') as fh:
        lines = fh.readlines()
    clean_out_results_csv = os.path.join(
        os.path.dirname(__file__), 
        'results/dca_esm_and_hybrid_opt_results_clean.csv'
    )
    with open(clean_out_results_csv, 'w') as fh2:
        header = lines[0]
        content = lines[1:]
        sort_keys = []
        for line in content:
                sort_keys.append(int(line.split(',')[0]))
        content_sorted, sort_keys_sorted = [l for l in zip(*sorted(
            zip(content, sort_keys), key=lambda x: x[1]))]
        fh2.write(header)
        for line in content_sorted:
            if (
                not line.split(',')[1].startswith('OOM') 
                and not line.split(',')[1].startswith('X') 
                and not line.split(',')[4].startswith('PDBseq neq WTseq')
                and not line.split(',')[4].startswith('Sequence too long')
            ):
                fh2.write(line)
    
    plot_csv_data(csv=clean_out_results_csv, plot_name='mut_performance')
