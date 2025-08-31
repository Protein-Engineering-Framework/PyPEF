
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, BiopythonParserWarning
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)

import sys  # Use local directory PyPEF files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pypef.dca.gremlin_inference import GREMLIN
from pypef.llm.utils import get_batches, corr_loss
from pypef.llm.esm_lora_tune import (
    get_esm_models, esm_tokenize_sequences, 
    esm_train, esm_infer
)
from pypef.llm.prosst_lora_tune import (
    get_logits_from_full_seqs, get_prosst_models, get_structure_quantizied, 
    prosst_tokenize_sequences, prosst_train
)
from pypef.llm.inference import inference
from pypef.utils.variant_data import get_seqs_from_var_name
from pypef.utils.helpers import get_vram, get_device
from pypef.hybrid.hybrid_model import (
    DCALLMHybridModel, reduce_by_batch_modulo, get_delta_e_statistical_model
)
from pypef.utils.split import DatasetSplitter


JUST_PLOT_RESULTS = False


def compute_performances(mut_data, mut_sep=':', start_i: int = 0, already_tested_is: list = []):
    # Get cpu, gpu or mps device for training.
    device = get_device()
    print(f"Using {device.upper()} device")
    get_vram()
    MAX_WT_SEQUENCE_LENGTH = 600
    MAX_VARIANT_FITNESS_PAIRS = 5000
    N_CV = 5
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
            if len(fitnesses) <= 50 or len(fitnesses) > MAX_VARIANT_FITNESS_PAIRS:
                print(f'Number of available variants <= 50 or > {MAX_VARIANT_FITNESS_PAIRS}'
                      f', skipping dataset...')
                with open(out_results_csv, 'a') as fh:
                    fh.write(
                        f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},'
                        f'{max_muts},{len(fitnesses)} variant fitness pairs (below 50 '
                        f'or more than {MAX_WT_SEQUENCE_LENGTH})\n'
                    )
                continue
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
            # ESM unsupervised
            try:
                x_esm, esm_attention_mask = esm_tokenize_sequences(
                    sequences, esm_tokenizer, max_length=len(wt_seq), verbose=False
                )
                y_esm = inference(sequences, 'esm', model=esm_base_model, verbose=False)
                print(f'ESM1v (unsupervised performance): '
                      f'{spearmanr(fitnesses, y_esm.cpu())[0]:.3f}')
                esm_unopt_perf = spearmanr(fitnesses, y_esm.cpu())[0]
            except RuntimeError:
                esm_unopt_perf = np.nan
            # ProSST unsupervised
            try:
                input_ids, prosst_attention_mask, structure_input_ids = get_structure_quantizied(
                    pdb, prosst_tokenizer, wt_seq, verbose=False
                    )
                x_prosst = prosst_tokenize_sequences(sequences=sequences, vocab=prosst_vocab, verbose=False)
                y_prosst = inference(sequences, 'prosst', pdb_file=pdb, wt_seq=wt_seq, model=prosst_base_model, verbose=False)
                print(f'ProSST (unsupervised performance): '
                      f'{spearmanr(fitnesses, y_prosst.cpu())[0]:.3f}')
                prosst_unopt_perf = spearmanr(fitnesses, y_prosst.cpu())[0]
            except RuntimeError:
                prosst_unopt_perf = np.nan
            
            if np.isnan(esm_unopt_perf) and np.isnan(prosst_unopt_perf):
                print('Both LLM\'s had RunTimeErrors, skipping dataset...')
                continue 

            ds = DatasetSplitter(df_or_csv_file=csv_path, n_cv=N_CV, mutation_separator=mut_sep)
            ds.plot_distributions()
            if max_muts >= 2:  # Only using random cross-validation splits
                print("Only performing random splits as data contains multi-substituted variants...")
                target_split_indices = [ds.get_random_single_multi_split_indices()]
            else:              # Using random, modulo, continuous CV splits
                print("Only single substituted variants found, performing random, modulo, and continuous data splits...")
                target_split_indices = ds.get_all_split_indices()
            temp_results = {}
            for c in ["Random", "Modulo", "Continuous"]:
                temp_results.update({c: {}})
                for s in range(N_CV):
                    temp_results[c].update({f'Split {s}': {}})
                    for m in ['DCA', 'ESM1v', 'ProSST', 'DCA hybrid', 'DCA+ESM1v hybrid', 'DCA+ProSST hybrid']:
                        # Prefill with NaN's
                        temp_results[c][f'Split {s}'].update({m: np.nan})
            for i_category, (train_indices, test_indices) in enumerate(target_split_indices):
                category = ["Random", "Modulo", "Continuous"][i_category]
                print(f'Category: {category}')
                for i_split, (train_i, test_i) in enumerate(zip(
                    train_indices, test_indices
                )):
                    print(f'    Split: {i_split + 1}')
                    try:
                        _train_sequences, test_sequences = np.asarray(sequences)[train_i], np.asarray(sequences)[test_i]
                        x_dca_train, x_dca_test = np.asarray(x_dca)[train_i], np.asarray(x_dca)[test_i]
                        x_llm_train_prosst, x_llm_test_prosst = np.asarray(x_prosst)[train_i], np.asarray(x_prosst)[test_i]
                        x_llm_train_esm, x_llm_test_esm = np.asarray(x_esm)[train_i], np.asarray(x_esm)[test_i]
                        y_train, y_test = np.asarray(fitnesses)[train_i], np.asarray(fitnesses)[test_i]
                        prosst_lora_model_2 = copy.deepcopy(prosst_lora_model)
                        prosst_optimizer = torch.optim.Adam(prosst_lora_model_2.parameters(), lr=0.0001)
                        esm_lora_model_2 = copy.deepcopy(esm_lora_model)
                        esm_optimizer = torch.optim.Adam(esm_lora_model_2.parameters(), lr=0.0001)
                        train_size, test_size = len(train_i), len(test_i)
                    except ValueError as e:
                        print(f"Only {len(fitnesses)} variant-fitness pairs in total, "
                              f"cannot split the data in N_Train = {train_size} and N_Test "
                              f"(N_Total - N_Train) [Excepted error: {e}].")
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
                    print(f'        Train: {len(np.array(y_train))} --> Test: {len(np.array(y_test))}')
                    if len(y_test) <= 50:
                        print(f"        Only {len(fitnesses)} in total, splitting the data "
                              f"in N_Train = {len(y_train)} and N_Test = {len(y_test)} "
                              f"results in N_Test <= 50 variants - not getting "
                              f"performance for N_Train = {len(y_train)}...")
                        continue

                    y_test_pred_dca = get_delta_e_statistical_model(x_dca_test, x_wt)
                    temp_results[category][f'Split {i_split}'].update({'DCA': spearmanr(y_test, y_test_pred_dca)[0]})
                    print(f'        DCA ZeroShot (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred_dca)[0]:.3f}')
                    y_test_pred_esm = inference(test_sequences, 'esm', model=esm_base_model, verbose=False)
                    temp_results[category][f'Split {i_split}'].update({'ESM1v': spearmanr(y_test, y_test_pred_esm)[0]})
                    print(f'        ESM1v ZeroShot (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred_esm)[0]:.3f}')
                    y_test_pred_prosst = inference(test_sequences, 'prosst', model=prosst_base_model, pdb_file=pdb, wt_seq=wt_seq, verbose=False)
                    temp_results[category][f'Split {i_split}'].update({'ProSST': spearmanr(y_test, y_test_pred_prosst)[0]})
                    print(f'        ProSST ZeroShot (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred_prosst)[0]:.3f}')

                    for i_m, method in enumerate([None, llm_dict_esm, llm_dict_prosst]):
                        m_str = ['DCA hybrid', 'DCA+ESM1v hybrid', 'DCA+ProSST hybrid'][i_m]
                        try:
                            hm = DCALLMHybridModel(
                                x_train_dca=np.array(x_dca_train), 
                                y_train=y_train,
                                llm_model_input=method,
                                x_wt=x_wt,
                                verbose=False
                            )
                            y_test_pred = hm.hybrid_prediction(
                                x_dca=np.array(x_dca_test), 
                                x_llm=[
                                    None, 
                                    np.asarray(x_llm_test_esm), 
                                    np.asarray(x_llm_test_prosst)
                                ][i_m],
                                verbose=False
                            )
                            print(f'        {m_str} (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred)[0]:.3f} '
                                  f'(train size={train_size}, test_size={test_size})')
                            temp_results[category][f'Split {i_split}'].update({m_str: spearmanr(y_test, y_test_pred)[0]})
                        except RuntimeError as e:  # modeling_prosst.py in forward
                            continue
                    del prosst_lora_model_2
                    del esm_lora_model_2
                    torch.cuda.empty_cache()
                    gc.collect()

            dt = time.time() - start_time

            with open(out_results_csv, 'a') as fh:
                fh.write(
                    f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},{max_muts},'
                    f'{dca_unopt_perf},{esm_unopt_perf},{prosst_unopt_perf},'
                    f'{temp_results['Random']['Split 0']['DCA hybrid']},{temp_results['Random']['Split 1']['DCA hybrid']},'
                    f'{temp_results['Random']['Split 2']['DCA hybrid']},{temp_results['Random']['Split 3']['DCA hybrid']},'
                    f'{temp_results['Random']['Split 4']['DCA hybrid']},'
                    f'{temp_results['Random']['Split 0']['DCA+ESM1v hybrid']},{temp_results['Random']['Split 1']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Random']['Split 2']['DCA+ESM1v hybrid']},{temp_results['Random']['Split 3']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Random']['Split 4']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Random']['Split 0']['DCA+ProSST hybrid']},{temp_results['Random']['Split 1']['DCA+ProSST hybrid']},'
                    f'{temp_results['Random']['Split 2']['DCA+ProSST hybrid']},{temp_results['Random']['Split 3']['DCA+ProSST hybrid']},'
                    f'{temp_results['Random']['Split 4']['DCA+ProSST hybrid']},'
                    f'{temp_results['Modulo']['Split 0']['DCA hybrid']},{temp_results['Modulo']['Split 1']['DCA hybrid']},'
                    f'{temp_results['Modulo']['Split 2']['DCA hybrid']},{temp_results['Modulo']['Split 3']['DCA hybrid']},'
                    f'{temp_results['Modulo']['Split 4']['DCA hybrid']},'
                    f'{temp_results['Modulo']['Split 0']['DCA+ESM1v hybrid']},{temp_results['Modulo']['Split 1']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Modulo']['Split 2']['DCA+ESM1v hybrid']},{temp_results['Modulo']['Split 3']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Modulo']['Split 4']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Modulo']['Split 0']['DCA+ProSST hybrid']},{temp_results['Modulo']['Split 1']['DCA+ProSST hybrid']},'
                    f'{temp_results['Modulo']['Split 2']['DCA+ProSST hybrid']},{temp_results['Modulo']['Split 3']['DCA+ProSST hybrid']},'
                    f'{temp_results['Modulo']['Split 4']['DCA+ProSST hybrid']},'
                    f'{temp_results['Continuous']['Split 0']['DCA hybrid']},{temp_results['Continuous']['Split 1']['DCA hybrid']},'
                    f'{temp_results['Continuous']['Split 2']['DCA hybrid']},{temp_results['Continuous']['Split 3']['DCA hybrid']},'
                    f'{temp_results['Continuous']['Split 4']['DCA hybrid']},'
                    f'{temp_results['Continuous']['Split 0']['DCA+ESM1v hybrid']},{temp_results['Continuous']['Split 1']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Continuous']['Split 2']['DCA+ESM1v hybrid']},{temp_results['Continuous']['Split 3']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Continuous']['Split 4']['DCA+ESM1v hybrid']},'
                    f'{temp_results['Continuous']['Split 0']['DCA+ProSST hybrid']},{temp_results['Continuous']['Split 1']['DCA+ProSST hybrid']},'
                    f'{temp_results['Continuous']['Split 2']['DCA+ProSST hybrid']},{temp_results['Continuous']['Split 3']['DCA+ProSST hybrid']},'
                    f'{temp_results['Continuous']['Split 4']['DCA+ProSST hybrid']},'
                    f'{int(dt)}\n')
                

def plot_csv_data(csv):
    blue_colors = mpl.colormaps['Blues'](np.linspace(0.3, 0.9, 4))
    red_colors = mpl.colormaps['Reds'](np.linspace(0.3, 0.9, 4))
    green_colors = mpl.colormaps['Greens'](np.linspace(0.3, 0.9, 4))
    plt.figure(figsize=(24, 12))
    sns.set_style("whitegrid")
    df = pd.read_csv(csv, sep=',')  
    df_mean = pd.DataFrame()
    print(df)
    for method in ['DCA_hybrid', 'DCA+ESM1v_hybrid', 'DCA+ProSST_hybrid']:
        for split_technique in ['Random', 'Modulo', 'Continuous']:
            performances = []
            for split in range(1, 6):
                performances.append(df[f'{split_technique}_Split_{split}_{method}'].to_list())
            df_mean[f'{method}_{split_technique}_mean'] = np.mean(performances, axis=0)
    plot = sns.violinplot(
        df_mean, saturation=0.4,
        palette=[
            blue_colors[0], blue_colors[1], blue_colors[2],
            green_colors[0],green_colors[1], green_colors[2],
            red_colors[0], red_colors[1], red_colors[2]
        ]
    )     
    sns.swarmplot(df_mean, color='black')
    for n in range(0, df_mean.shape[1]):
        plt.text(
            n + 0.15, -0.075, 
            r'$\overline{\rho}=$' + f'{np.nanmean(df_mean.iloc[:, n]):.3f}\n' 
            + r'$N_\mathrm{Datasets}=$' + f'{np.count_nonzero(~np.isnan(np.array(df_mean.iloc[:, n])))}'
        )
    plot.set_xticks(range(len(plot.get_xticklabels())))
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.ylabel(r'Spearman $\rho$')
    plt.xlabel('Splitting technique')
    plt.ylim(-0.09, 1.09)
    plt.margins(0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'crossval_pgym_violin.png'), dpi=300)
    print('Saved file as ' + os.path.join(os.path.dirname(__file__), 'crossval_pgym_violin.png') + '.')
    #plt.show()


if __name__ == '__main__':
    single_point_mut_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"single_point_dms_mut_data.json"
    ))
    higher_mut_data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"higher_point_dms_mut_data.json"
    ))
    with open(single_point_mut_data, 'r') as fh:
        s_mut_data = json.loads(fh.read())
    with open(higher_mut_data, 'r') as fh:
        h_mut_data = json.loads(fh.read())
    combined_mut_data = s_mut_data.copy()
    combined_mut_data.update(h_mut_data)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    out_results_csv = os.path.join(
        os.path.dirname(__file__), 'results/dca_esm_and_hybrid_5cv-split_results.csv'
    )
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
                'No.,Dataset,N_Variants,N_Max_Muts,'
                'Untrained_Performance_DCA,Untrained_Performance_ESM1v,Untrained_Performance_ProSST,'
                'Random_Split_1_DCA_hybrid,Random_Split_2_DCA_hybrid,Random_Split_3_DCA_hybrid,'
                'Random_Split_4_DCA_hybrid,Random_Split_5_DCA_hybrid,'
                'Random_Split_1_DCA+ESM1v_hybrid,Random_Split_2_DCA+ESM1v_hybrid,Random_Split_3_DCA+ESM1v_hybrid,'
                'Random_Split_4_DCA+ESM1v_hybrid,Random_Split_5_DCA+ESM1v_hybrid,'
                'Random_Split_1_DCA+ProSST_hybrid,Random_Split_2_DCA+ProSST_hybrid,Random_Split_3_DCA+ProSST_hybrid,'
                'Random_Split_4_DCA+ProSST_hybrid,Random_Split_5_DCA+ProSST_hybrid,'
                'Modulo_Split_1_DCA_hybrid,Modulo_Split_2_DCA_hybrid,Modulo_Split_3_DCA_hybrid,'
                'Modulo_Split_4_DCA_hybrid,Modulo_Split_5_DCA_hybrid,'
                'Modulo_Split_1_DCA+ESM1v_hybrid,Modulo_Split_2_DCA+ESM1v_hybrid,Modulo_Split_3_DCA+ESM1v_hybrid,'
                'Modulo_Split_4_DCA+ESM1v_hybrid,Modulo_Split_5_DCA+ESM1v_hybrid,'
                'Modulo_Split_1_DCA+ProSST_hybrid,Modulo_Split_2_DCA+ProSST_hybrid,Modulo_Split_3_DCA+ProSST_hybrid,'
                'Modulo_Split_4_DCA+ProSST_hybrid,Modulo_Split_5_DCA+ProSST_hybrid,'
                'Continuous_Split_1_DCA_hybrid,Continuous_Split_2_DCA_hybrid,Continuous_Split_3_DCA_hybrid,'
                'Continuous_Split_4_DCA_hybrid,Continuous_Split_5_DCA_hybrid,'
                'Continuous_Split_1_DCA+ESM1v_hybrid,Continuous_Split_2_DCA+ESM1v_hybrid,Continuous_Split_3_DCA+ESM1v_hybrid,'
                'Continuous_Split_4_DCA+ESM1v_hybrid,Continuous_Split_5_DCA+ESM1v_hybrid,'
                'Continuous_Split_1_DCA+ProSST_hybrid,Continuous_Split_2_DCA+ProSST_hybrid,Continuous_Split_3_DCA+ProSST_hybrid,'
                'Continuous_Split_4_DCA+ProSST_hybrid,Continuous_Split_5_DCA+ProSST_hybrid,'
                'Time_in_s\n'
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
        'results/dca_esm_and_hybrid_5cv-split_results_clean.csv'
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
    
    plot_csv_data(csv=clean_out_results_csv)
