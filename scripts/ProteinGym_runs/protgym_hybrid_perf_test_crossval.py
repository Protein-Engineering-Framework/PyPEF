
import os
import gc
import time
import warnings
import psutil
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, BiopythonParserWarning

from pypef.plm.inference import esm_setup, plm_inference, prosst_setup, tokenize_sequences
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)

import sys  # Use local directory PyPEF files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pypef.dca.gremlin_inference import GREMLIN
from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.prosst_lora_tune import get_prosst_models, get_structure_quantizied
from pypef.utils.variant_data import get_seqs_from_var_name, check_alignment, shift_and_trim_vars_seqs
from pypef.utils.helpers import get_vram, get_device
from pypef.hybrid.hybrid_model import (
    DCALLMHybridModel, get_delta_e_statistical_model
)
from pypef.utils.split import DatasetSplitter
from pypef import __version__
version = __version__.split('-')[0]
# e.g., version = '0.4.3'

import logging
package_logger = logging.getLogger('pypef')
package_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
package_logger.addHandler(handler)

JUST_PLOT_RESULTS = False

ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
ESM_REVISION = None
PROSST_REVISION = "e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8"
HYBRID_MODELS = ['DCA hybrid', 'DCA+ESM hybrid', 'DCA+ProSST hybrid', 'DCA+ESM+ProSST hybrid']
ZERO_SHOT_MODELS = ['DCA', 'ESM', 'ProSST']
CATEGORIES = ['Random', 'Modulo', 'Continuous']
N_CV = 5


def compute_performances(mut_data, mut_sep=':', start_i: int = 0, already_tested_is: list = []):
    # Get cpu, gpu or mps device for training.
    device = get_device()
    print(f"Using {device.upper()} device")
    get_vram()
    MAX_WT_SEQUENCE_LENGTH = 600
    MAX_VARIANT_FITNESS_PAIRS = 5000
    print(f"Maximum sequence length: {MAX_WT_SEQUENCE_LENGTH}")
    print(f"Loading LLM models into {device} device...")
    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models(
        seed=42, revision=PROSST_REVISION)
    prosst_base_model = prosst_base_model.to(device).float()
    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
        model=ESM_MODEL, seed=42, revision=ESM_REVISION)
    esm_base_model = esm_base_model.to(device)
    get_vram()
    plt.figure(figsize=(40, 12))
    numbers_of_datasets = [i + 1 for i in range(len(mut_data.keys()))]
    for i, (dset_key, dset_paths) in enumerate(mut_data.items()):
        if i >= start_i and i not in already_tested_is:
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
            # Getting % usage of virtual_memory (3rd field)
            print(f'RAM used: {round(psutil.virtual_memory()[3]/1E9, 3)} '
                  f'GB ({psutil.virtual_memory()[2]} %)')
            variant_fitness_data = pd.read_csv(csv_path, sep=',')
            print('N_variant-fitness-tuples:', np.shape(variant_fitness_data)[0])
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
                        f'or more than {MAX_VARIANT_FITNESS_PAIRS})\n'
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

            # PDB alignment handling
            pdb_seq = str(list(SeqIO.parse(pdb, "pdb-atom"))[0].seq)
            if pdb_seq != wt_seq:
                mapping = check_alignment(wt_seq, pdb_seq)
                pdb_trimmed_wt = mapping['common_seq']
                print(f"WT/PDB mismatch. Trimmed WT to PDB length: {len(pdb_trimmed_wt)}")
                if mapping and mapping['identity'] > 0.8:
                    pdb_vars, _orig_vars, gremlin_seqs, pdb_trimmed_seqs = shift_and_trim_vars_seqs(
                        vars_list=variants,
                        seqs_list=sequences,
                        alignment_mapping=mapping,
                        msa_start=msa_start
                    )
                    variants = pdb_vars
                    print(f"Shifted variants. PDB-trimmed length: {len(pdb_trimmed_seqs[0])}, "
                          f"GREMLIN length: {len(gremlin_seqs[0])}")
                else:
                    print(
                        f"Wild-type sequence is not matching PDB-extracted sequence"
                        f"\nWT sequence:\n{wt_seq}\nPDB sequence:\n{pdb_seq}\nSkipping dataset..."
                    )
                    with open(out_results_csv, 'a') as fh:
                        fh.write(
                            f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},'
                            f'{max_muts},PDBseq neq WTseq\n'
                        )
                    continue
            else:
                pdb_trimmed_wt = wt_seq
                gremlin_seqs = sequences
                pdb_trimmed_seqs = sequences

            # GREMLIN DCA encoding (uses MSA-length sequences)
            print('GREMLIN-DCA: optimization...')
            gremlin = GREMLIN(alignment=msa_path, opt_iter=100, optimize=True)
            x_dca = gremlin.collect_encoded_sequences(gremlin_seqs)
            x_wt = gremlin.x_wt
            y_pred_dca = get_delta_e_statistical_model(x_dca, x_wt)
            print(f'DCA (unsupervised performance): {spearmanr(fitnesses, y_pred_dca)[0]:.3f}')
            dca_unopt_perf = spearmanr(fitnesses, y_pred_dca)[0]

            # ESM unsupervised (uses PDB-trimmed sequences)
            try:
                x_esm, esm_attention_mask = tokenize_sequences(
                    pdb_trimmed_seqs, esm_tokenizer, max_length=len(pdb_trimmed_wt) + 2)
                wt_tokens, _ = tokenize_sequences(
                    [pdb_trimmed_wt],
                    esm_tokenizer,
                    max_length=len(pdb_trimmed_wt) + 2
                )
                wt_tokens = torch.tensor(wt_tokens[0], dtype=torch.long)  # shape (L,)
                y_esm = plm_inference(
                    tokenized_sequences=x_esm,
                    wt_input_ids=wt_tokens,
                    attention_mask=esm_attention_mask,
                    model=esm_base_model,
                    batch_size=5,
                    train=False,
                    device=device,
                    verbose=True
                ).cpu()
                print(f'ESM (unsupervised performance): '
                      f'{spearmanr(fitnesses, y_esm.cpu())[0]:.3f}')
                esm_unopt_perf = spearmanr(fitnesses, y_esm.cpu())[0]
            except RuntimeError:
                esm_unopt_perf = np.nan
            # ProSST unsupervised (uses PDB-trimmed sequences)
            try:
                wt_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
                    pdb, prosst_tokenizer, pdb_trimmed_wt
                )
                x_prosst, _prosst_attention_mask = tokenize_sequences(
                    sequences=pdb_trimmed_seqs,
                    tokenizer=prosst_tokenizer,
                    max_length=len(pdb_trimmed_wt) + 2
                )
                y_prosst = plm_inference(
                    tokenized_sequences=x_prosst,
                    wt_input_ids=wt_input_ids,
                    attention_mask=prosst_attention_mask,
                    model=prosst_base_model,
                    wt_structure_input_ids=wt_structure_input_ids,
                    batch_size=5,
                    train=False,
                    device=device,
                    verbose=True
                ).cpu()
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
            for c in CATEGORIES:
                temp_results[c] = {}
                for s in range(N_CV):
                    temp_results[c][f'Split {s}'] = {}
                    for m in ZERO_SHOT_MODELS + HYBRID_MODELS:
                        temp_results[c][f'Split {s}'][m] = np.nan
            llm_dict_esm = llm_dict_prosst = llm_dict_ensemble = None
            for i_category, (train_indices, test_indices) in enumerate(target_split_indices):
                category = CATEGORIES[i_category]
                print(f'~~~ Category: {category} ~~~')
                for i_split, (train_i, test_i) in enumerate(zip(
                    train_indices, test_indices
                )):
                    print(f'    ----- Split: {i_split + 1} -----')
                    try:
                        pdb_train_seqs = np.asarray(pdb_trimmed_seqs)[train_i]
                        pdb_test_seqs = np.asarray(pdb_trimmed_seqs)[test_i]
                        x_dca_train, x_dca_test = np.asarray(x_dca)[train_i], np.asarray(x_dca)[test_i]
                        x_llm_train_prosst, x_llm_test_prosst = np.asarray(x_prosst)[train_i], np.asarray(x_prosst)[test_i]
                        x_llm_train_esm, x_llm_test_esm = np.asarray(x_esm)[train_i], np.asarray(x_esm)[test_i]
                        y_train, y_test = np.asarray(fitnesses)[train_i], np.asarray(fitnesses)[test_i]
                        train_size, test_size = len(train_i), len(test_i)
                    except ValueError as e:
                        print(f"Only {len(fitnesses)} variant-fitness pairs in total, "
                              f"cannot split the data in N_Train = {train_size} and N_Test "
                              f"(N_Total - N_Train) [Excepted error: {e}].")
                        continue
                    if llm_dict_esm is not None:
                        llm_dict_esm = llm_dict_prosst = llm_dict_ensemble = None
                        gc.collect()
                        torch.cuda.empty_cache()

                    print(f'        Train: {len(np.array(y_train))} --> Test: {len(np.array(y_test))}')
                    if len(y_test) <= 50:
                        print(f"        Only {len(fitnesses)} in total, splitting the data "
                              f"in N_Train = {len(y_train)} and N_Test = {len(y_test)} "
                              f"results in N_Test <= 50 variants - not getting "
                              f"performance for N_Train = {len(y_train)}...")
                        continue

                    llm_dict_esm = esm_setup(
                            wt_seq=pdb_trimmed_wt, sequences=list(pdb_train_seqs),
                            model=ESM_MODEL,
                            seed=42, revision=ESM_REVISION, device=device, verbose=True
                    )
                    llm_dict_prosst = prosst_setup(
                            wt_seq=pdb_trimmed_wt, pdb_file=pdb, sequences=list(pdb_train_seqs),
                            seed=42, revision=PROSST_REVISION, device=device, verbose=True
                    )
                    llm_dict_ensemble = {**llm_dict_esm, **llm_dict_prosst}

                    y_test_pred_dca = get_delta_e_statistical_model(x_dca_test, x_wt)
                    temp_results[category][f'Split {i_split}'].update({'DCA': spearmanr(y_test, y_test_pred_dca)[0]})
                    print(f'        DCA ZeroShot (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred_dca)[0]:.3f}')
                    y_test_pred_esm = plm_inference(
                        tokenized_sequences=x_llm_test_esm,
                        wt_input_ids=wt_tokens,
                        attention_mask=esm_attention_mask,
                        model=esm_base_model,
                        batch_size=5,
                        train=False,
                        device=device,
                        verbose=True
                    ).cpu()
                    temp_results[category][f'Split {i_split}'].update({'ESM': spearmanr(y_test, y_test_pred_esm)[0]})
                    print(f'        ESM ZeroShot (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred_esm)[0]:.3f}')
                    y_test_pred_prosst = plm_inference(
                        tokenized_sequences=x_llm_test_prosst,
                        wt_input_ids=wt_input_ids,
                        attention_mask=prosst_attention_mask,
                        model=prosst_base_model,
                        wt_structure_input_ids=wt_structure_input_ids,
                        batch_size=5,
                        train=False,
                        device=device,
                        verbose=True
                    ).cpu()
                    temp_results[category][f'Split {i_split}'].update({'ProSST': spearmanr(y_test, y_test_pred_prosst)[0]})
                    print(f'        ProSST ZeroShot (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred_prosst)[0]:.3f}')

                    for i_m, method in enumerate([None, llm_dict_esm, llm_dict_prosst, llm_dict_ensemble]):
                        m_str = HYBRID_MODELS[i_m]
                        try:
                            gauss_opt = (method is not None)
                            hm = DCALLMHybridModel(
                                x_train_dca=np.array(x_dca_train),
                                y_train=y_train,
                                llm_model_input=method,
                                x_dca_wt=x_wt,
                                sequences=list(pdb_train_seqs),
                                wt_sequence=pdb_trimmed_wt,
                                gauss_opt=gauss_opt,
                                lora_train=False,
                                pdb_struct=pdb,
                                splitting_scheme='block-random',
                                batch_size=5,
                                seed=42,
                                verbose=False
                            )
                            x_llm_test_dict = [
                                None,
                                {'esm': np.asarray(x_llm_test_esm)},
                                {'prosst': np.asarray(x_llm_test_prosst)},
                                {'esm': np.asarray(x_llm_test_esm), 'prosst': np.asarray(x_llm_test_prosst)}
                            ][i_m]
                            y_test_pred, indiv_preds = hm.hybrid_prediction(
                                x_dca=np.array(x_dca_test),
                                x_llm_dict=x_llm_test_dict,
                                sequences=list(pdb_test_seqs)
                            )
                            print(f'        Individual model performances:')
                            for model_name, model_preds in indiv_preds.items():
                                print(f'          {model_name}: {spearmanr(y_test, model_preds)[0]:.3f}')
                            print(f'        {m_str} (split {i_split + 1}) performance: {spearmanr(y_test, y_test_pred)[0]:.3f} '
                                  f'(train size={train_size}, test_size={test_size})')
                            temp_results[category][f'Split {i_split}'].update({m_str: spearmanr(y_test, y_test_pred)[0]})
                        except RuntimeError as e:  # modeling_prosst.py in forward
                            print(f'        {m_str} RuntimeError: {e}')
                            continue
                    gc.collect()
                    torch.cuda.empty_cache()

            dt = time.time() - start_time

            with open(out_results_csv, 'a') as fh:
                line = (f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},{max_muts},'
                        f'{dca_unopt_perf},{esm_unopt_perf},{prosst_unopt_perf}')
                for cat in CATEGORIES:
                    for model in HYBRID_MODELS:
                        for s in range(N_CV):
                            line += f',{temp_results[cat][f"Split {s}"][model]}'
                line += f',{int(dt)}\n'
                fh.write(line)


def _build_csv_header():
    """Build the CSV header string for the results file."""
    header = ('No.,Dataset,N_Variants,N_Max_Muts,'
              'Untrained_Performance_DCA,Untrained_Performance_ESM,Untrained_Performance_ProSST')
    for cat in CATEGORIES:
        for model in HYBRID_MODELS:
            model_col = model.replace(' ', '_')
            for s in range(1, N_CV + 1):
                header += f',{cat}_Split_{s}_{model_col}'
    header += ',Time_in_s\n'
    return header


def plot_csv_data(csv, plot_name):
    blue_colors = mpl.colormaps['Blues'](np.linspace(0.3, 0.9, 4))
    red_colors = mpl.colormaps['Reds'](np.linspace(0.3, 0.9, 4))
    green_colors = mpl.colormaps['Greens'](np.linspace(0.3, 0.9, 4))
    purple_colors = mpl.colormaps['Purples'](np.linspace(0.3, 0.9, 4))
    plt.figure(figsize=(24, 12))
    sns.set_style("whitegrid")
    df = pd.read_csv(csv, sep=',')
    df_mean = pd.DataFrame()
    print(df)
    color_map = {
        'DCA_hybrid': blue_colors,
        'DCA+ESM_hybrid': green_colors,
        'DCA+ProSST_hybrid': red_colors,
        'DCA+ESM+ProSST_hybrid': purple_colors,
    }
    palette = []
    for method, colors in color_map.items():
        for i_cat, split_technique in enumerate(CATEGORIES):
            performances = []
            for split in range(1, N_CV + 1):
                col = f'{split_technique}_Split_{split}_{method}'
                if col in df.columns:
                    performances.append(df[col].to_list())
            if performances:
                df_mean[f'{method}_{split_technique}_mean'] = np.mean(performances, axis=0)
                palette.append(colors[i_cat])
    plot = sns.violinplot(
        df_mean, saturation=0.4, palette=palette
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
    plt.savefig(plot_name, dpi=300)
    print(f'Saved file as {os.path.abspath(plot_name)}')
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
        os.path.dirname(__file__), f'results/dca_esm_and_hybrid_5cv-split_results_v{version}.csv'
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
            fh.write(_build_csv_header())
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
    clean_out_results_csv = os.path.splitext(out_results_csv)[0] + '_clean.csv'
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
                and not line.split(',')[4].startswith('More than')
            ):
                fh2.write(line)

    plot_csv_data(
        csv=clean_out_results_csv, 
        plot_name=os.path.join(os.path.dirname(__file__), f'crossval_mut_performance_violin_v{version}.png')
    )
