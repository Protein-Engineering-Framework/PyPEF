
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
from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.inference import esm_setup, plm_inference, prosst_setup, tokenize_sequences
from pypef.plm.prosst_lora_tune import get_prosst_models, get_structure_quantizied
from pypef.utils.variant_data import get_seqs_from_var_name, check_alignment, shift_and_trim_vars_seqs
from pypef.utils.helpers import get_vram, get_device
from pypef.hybrid.hybrid_model import DCALLMHybridModel, get_delta_e_statistical_model
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

ESM_MODEL = 'facebook/esm2_t33_650M_UR50D'
ESM_REVISION = None
PROSST_REVISION = "e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8"

JUST_PLOT_RESULTS = False



def compute_performances(mut_data, mut_sep=':', start_i: int = 0, already_tested_is: list = []):
    # TODO: Add (R)MSE next to Spearman
    # Get cpu, gpu or mps device for training.
    # LoRA-training takes a lot of time (e.g for 50 epochs at each step) while not giving better
    # results compared to Gaussian optimization or if additive predictor model next to Gaussian optimized
    # PLM embeddings.
    LORA_TRAIN = False
    MAX_WT_SEQUENCE_LENGTH = 1000
    MAX_N_VARIANTS = 100000
    seed = 42
    device = get_device()

    print(f"Using {device.upper()} device")

    get_vram()
    print(f"Maximum sequence length: {MAX_WT_SEQUENCE_LENGTH}")
    print(f"Loading LLM models into {device} device...")
    print('Getting ProSST models...')
    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models(
        seed=42, revision=PROSST_REVISION)
    prosst_base_model = prosst_base_model.to(device).float()
    print('Getting ESM models...')
    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
        model=ESM_MODEL, seed=42, revision=ESM_REVISION)
    esm_base_model = esm_base_model.to(device)
    get_vram()
    prosst_unopt_perfs, esm_unopt_perfs = [], []
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
            variants = variant_fitness_data['mutant'].to_numpy()
            variants_orig = variants
            fitnesses = variant_fitness_data['DMS_score'].to_numpy()
            variants_split = []
            for variant in variants:
                # Split double and higher substituted variants to multiple single substitutions
                # e.g. separated by ':' or '/'
                variants_split.append(variant.split(mut_sep))
            n_muts = []
            for variant in variants_split:
                n_muts.append(len(variant))
            max_muts = max(n_muts)
            print('N_variant-fitness-tuples:', np.shape(variant_fitness_data)[0])
            if np.shape(variant_fitness_data)[0] > MAX_N_VARIANTS:
                print(f'More than {MAX_N_VARIANTS} variant-fitness pairs which takes too '
                      f'long to compute, skipping dataset...')
                with open(out_results_csv, 'a') as fh:
                    fh.write(
                        f'{numbers_of_datasets[i]},{dset_key},{len(variants_orig)},'
                        f'{max_muts},More than {MAX_N_VARIANTS} variant-fitness pairs\n'
                    )
                continue
            if len(fitnesses) <= 50:
                print('Number of available variants <= 50, skipping dataset...')
                continue

            variants, fitnesses, sequences = get_seqs_from_var_name(
                wt_seq, variants_split, fitnesses, shift_pos=msa_start - 1)
            # Only model sequences with length of max. 800 amino acids to avoid out of memory errors
            print('Sequence length:', len(wt_seq))
            for s in sequences:
                assert len(s) == len(wt_seq)
            count_gap_variants = 0
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
                        f"\nWT sequence:\n{wt_seq}\nPDB sequence:\n{pdb_seq}. TODO: Shifting "
                        f"variants and trimming sequences. Skipping dataset..."
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
                      f'{spearmanr(fitnesses, y_esm)[0]:.3f}')
                esm_unopt_perf = spearmanr(fitnesses, y_esm)[0]
            except RuntimeError:
                esm_unopt_perf = np.nan
            gc.collect()
            torch.cuda.empty_cache()

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

            prosst_unopt_perfs.append(prosst_unopt_perf)
            esm_unopt_perfs.append(esm_unopt_perf)
            print(f'Current mean ProSST unsupervised: N={len(prosst_unopt_perfs)} SpearCorr={np.nanmean(prosst_unopt_perfs):.3f}')
            print(f'Current mean ESM unsupervised: N={len(esm_unopt_perfs)} SpearCorr={np.nanmean(esm_unopt_perfs):.3f}')
            ns_y_test = [len(variants)]
            for i_t, train_size in enumerate([100, 200, 1000]):
                print('\nTRAIN SIZE:', train_size, '\n-------------------------------------------\n')
                get_vram()
                try:
                    (
                        s_train, s_test,
                        x_dca_train, x_dca_test,
                        x_llm_train_prosst, x_llm_test_prosst,
                        x_llm_train_esm, x_llm_test_esm,
                        y_train, y_test
                    ) = train_test_split(
                        pdb_trimmed_seqs,
                        x_dca,
                        x_prosst,
                        x_esm,
                        fitnesses,
                        train_size=train_size,
                        random_state=seed
                    )
                except ValueError as e:
                    print(f"Only {len(fitnesses)} variant-fitness pairs in total, "
                          f"cannot split the data in N_Train = {train_size} and N_Test "
                          f"(N_Total - N_Train) [Excepted error: {e}].")

                    hybrid_perfs.extend([np.nan, np.nan, np.nan, np.nan])
                    ns_y_test.append(np.nan)
                    continue

                llm_dict_esm = esm_setup(
                        wt_seq=pdb_trimmed_wt, sequences=list(s_train),
                        model=ESM_MODEL,
                        seed=seed, revision=ESM_REVISION, device=device, verbose=True
                )
                llm_dict_prosst = prosst_setup(
                        wt_seq=pdb_trimmed_wt, pdb_file=pdb, sequences=list(s_train),
                        seed=seed, revision=PROSST_REVISION, device=device, verbose=True
                )
                llm_dict_ensemble = {**llm_dict_esm, **llm_dict_prosst}
                print(f'Train: {len(np.array(y_train))} --> Test: {len(np.array(y_test))}')
                if len(y_test) <= 50:
                    print(f"Only {len(fitnesses)} in total, splitting the data "
                          f"in N_Train = {len(y_train)} and N_Test = {len(y_test)} "
                          f"results in N_Test <= 50 variants - not getting "
                          f"performance for N_Train = {len(y_train)}...")
                    hybrid_perfs.extend([np.nan, np.nan, np.nan, np.nan])
                    ns_y_test.append(np.nan)
                    continue
                get_vram()
                for i_m, llm_dict in enumerate([None, llm_dict_esm, llm_dict_prosst, llm_dict_ensemble]):
                    print('\n~~~ ' + ['DCA hybrid', 'DCA+ESM hybrid', 'DCA+ProSST hybrid', 'DCA+ESM+ProSST hybrid'][i_m] + ' ~~~')
                    try:
                        if i_m == 0:
                            lora_train=False
                            gauss_opt=False
                        else:
                            lora_train=LORA_TRAIN
                            gauss_opt=True
                        hm = DCALLMHybridModel(
                            x_train_dca=np.array(x_dca_train),
                            y_train=y_train,
                            llm_model_input=llm_dict,
                            x_dca_wt=x_wt,
                            sequences=list(s_train),
                            wt_sequence=pdb_trimmed_wt,
                            lora_train=lora_train,
                            gauss_opt=gauss_opt,
                            pdb_struct=pdb,
                            splitting_scheme='random',
                            batch_size=5,
                            seed=seed,
                            n_epochs=50,  # Only if lora_train==True
                            verbose=False
                        )
                        y_test_pred, indiv_preds = hm.hybrid_prediction(
                            x_dca=np.array(x_dca_test),
                            x_llm_dict=[
                                None,
                                {'esm': np.asarray(x_llm_test_esm)},
                                {'prosst': np.asarray(x_llm_test_prosst)},
                                {'esm': np.asarray(x_llm_test_esm), 'prosst': np.asarray(x_llm_test_prosst)}
                            ][i_m],
                            sequences=list(s_test)
                        )
                        print(f'Individual model performances:')
                        for model_name, model_preds in indiv_preds.items():
                            print(f'  {model_name}: {spearmanr(y_test, model_preds)[0]:.3f}')
                        print(f'Hybrid performance: {spearmanr(y_test, y_test_pred)[0]:.3f}')
                        hybrid_perfs.append(spearmanr(y_test, y_test_pred)[0])
                    except RuntimeError as e:  # modeling_prosst.py, line 920, in forward
                        # or UnboundLocalError in prosst_lora_tune.py, line 167
                        print(e, '\nAppending performance NaN...')
                        hybrid_perfs.append(np.nan)
                ns_y_test.append(len(y_test_pred))
                gc.collect()
                torch.cuda.empty_cache()

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
                    f'{esm_unopt_perf},{prosst_unopt_perf},{dset_hybrid_perfs_i}{dset_ns_y_test_i}{int(dt)}\n'
                )


def plot_csv_data(csv, plot_name):
    df = pd.read_csv(csv, sep=',')
    num_dsets = len(df)
    train_test_size_texts = []

    configs = [
        ('DCA', 'Hybrid_DCA', 'Blues'),
        ('ESM', 'Hybrid_DCA_ESM', 'Greens'),
        ('ProSST', 'Hybrid_DCA_ProSST', 'Reds'),
        ('Triple', 'Hybrid_DCA_ESM_ProSST', 'Purples')
    ]

    training_sizes = [0, 100, 200, 1000]
    all_perfs = []
    all_column_names = []
    all_colors = []

    plt.figure(figsize=(80, 12))

    for model_label, prefix, cmap_name in configs:
        colors = mpl.colormaps[cmap_name](np.linspace(0.3, 0.9, len(training_sizes)))

        for idx, size in enumerate(training_sizes):
            # Construct column name dynamically
            col = (f'Untrained_Performance_{model_label}' if size == 0 and model_label != 'Triple'
                   else f'{prefix}_Trained_Performance_{size}')
            if model_label == 'Triple' and size == 0: continue # Skip if no untrained baseline for triple

            if col not in df.columns:
                continue # Safety check

            data = df[col]
            mean_val = np.nanmean(data)
            color = colors[idx]

            plt.plot(range(num_dsets), data, 'o--', markersize=8, color=color, label=f'{model_label} ({size})')
            plt.plot(range(num_dsets + 1), np.full(num_dsets + 1, mean_val), color=color, linestyle='--')

            train_test_size_texts.append(plt.text(num_dsets, mean_val, f'{mean_val:.2f}', color=color))

            # Add training metadata (Only for DCA/First group to avoid clutter, or as needed)
            if model_label == 'DCA':
                n_test_col = 'N_Y_test' if size == 0 else f'N_Y_test_{size}'
                y_pos = 0.975 + (idx * 0.005)
                for i, n_test in enumerate(df[n_test_col].astype('Int64')):
                    plt.text(i, y_pos, f'{size}' + r'$\rightarrow$' + f'{n_test}', color='black', size=2)

            all_perfs.append(data)
            all_column_names.append(col)
            all_colors.append(color)

    plt.grid(zorder=-1)
    plt.xticks(
        range(num_dsets), ['(' + str(n) + ') ' + name for (n, name) in zip(df['No.'], df['Dataset'])], 
        rotation=45, ha='right'
    )
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.ylabel(r'Spearman $\rho$')
    adjust_text(train_test_size_texts, expand=(1.2, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{plot_name}.png'), dpi=300)

    # Violin Plot Section
    plt.clf()
    plt.figure(figsize=(28, 12))
    sns.set_style("whitegrid")

    df_violin = df[all_column_names]
    plot = sns.violinplot(data=df_violin, saturation=0.4, palette=all_colors)
    sns.swarmplot(data=df_violin, color='black', size=3)

    tick_locations = range(len(all_column_names))
    plot.set_xticks(tick_locations)

    for n, perf_data in enumerate(all_perfs):
        plt.text(n + 0.15, -0.075,
                 r'$\overline{\rho}=$' + f'{np.nanmean(perf_data):.3f}\n' +
                 r'$N_\mathrm{D}=$' + f'{np.count_nonzero(~np.isnan(perf_data))}')

    plot.set_xticklabels(all_column_names, rotation=45, ha='right')
    plt.ylim(-0.09, 1.09)
    plt.tight_layout()
    plt_location = os.path.join(os.path.dirname(__file__), f'{plot_name}_violin.png')
    plt.savefig(plt_location, dpi=300)
    plt.close()
    print(f'Saved plot at {plt_location}.')


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
    out_results_csv = os.path.join(os.path.dirname(__file__), f'results/dca_esm_and_hybrid_opt_results_v{version}.csv')
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
            if JUST_PLOT_RESULTS:
                print(f'Plotting model performance results up to {start_i + 1} '
                      f'{list(combined_mut_data.keys())[start_i]} '
                      f'(last tested dataset: {start_i}, {list(combined_mut_data.keys())[start_i - 1]}).')
            else:
                print(f'\nContinuing getting model performances at {start_i + 1} '
                      f'{list(combined_mut_data.keys())[start_i]} '
                      f'(last tested dataset: {start_i}, {list(combined_mut_data.keys())[start_i - 1]})...')
        except IndexError:
            print('\nComputed all results already?!')
    else:
        with open(out_results_csv, 'w') as fh:
            print(f'\nCreating new file {out_results_csv}...')
            fh.write(
                f'No.,Dataset,N_Variants,N_Max_Muts,Untrained_Performance_DCA,Untrained_Performance_ESM,'
                f'Untrained_Performance_ProSST,Hybrid_DCA_Trained_Performance_100,'
                f'Hybrid_DCA_ESM_Trained_Performance_100,Hybrid_DCA_ProSST_Trained_Performance_100,'
                f'Hybrid_DCA_ESM_ProSST_Trained_Performance_100,'
                f'Hybrid_DCA_Trained_Performance_200,Hybrid_DCA_ESM_Trained_Performance_200,'
                f'Hybrid_DCA_ProSST_Trained_Performance_200,'
                f'Hybrid_DCA_ESM_ProSST_Trained_Performance_200,Hybrid_DCA_Trained_Performance_1000,'
                f'Hybrid_DCA_ESM_Trained_Performance_1000,Hybrid_DCA_ProSST_Trained_Performance_1000,'
                f'Hybrid_DCA_ESM_ProSST_Trained_Performance_1000,'
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

    plot_csv_data(csv=clean_out_results_csv, plot_name=f'low_n_mut_performance_v{version}')
