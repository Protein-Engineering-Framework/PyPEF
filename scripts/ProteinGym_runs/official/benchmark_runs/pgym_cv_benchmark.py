"""
Main benchmarking script to evaluate PyPEF hybrid LLM-DCA model on ProteinGym DMS assays.
Structured based on Kermut run script.
"""

from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from scipy.stats import spearmanr
from Bio import SeqIO, BiopythonParserWarning
import warnings
warnings.filterwarnings(action='ignore', category=BiopythonParserWarning)


from pypef.utils.variant_data import get_mismatches
from pypef.llm.prosst_lora_tune import prosst_setup, prosst_tokenize_sequences
from pypef.llm.esm_lora_tune import esm_setup, esm_tokenize_sequences
from pypef.dca.gremlin_inference import GREMLIN, get_delta_e_statistical_model
from pypef.hybrid.hybrid_model import DCALLMHybridModel


@hydra.main(version_base=None, config_path="../configs", config_name="proteingym_data_setup")
def main(cfg: DictConfig) -> None:
    # Experiment settings
    split_method = cfg.split_method
    progress_bar = cfg.progress_bar
    llm = cfg.llm
    sequence_col, target_col = "mutated_sequence", "DMS_score"
    assert cfg.split_method in ["fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"]
    use_multiples = True if cfg.split_method == "fold_rand_multiples" else False

    # Verify input paths
    if use_multiples:
        DMS_data_folder = Path(cfg.DMS_data_folder_multiples)
    else:
        DMS_data_folder = Path(cfg.DMS_data_folder_singles)

    DMS_reference_file_path = Path(cfg.DMS_reference_file_path)
    DMS_MSA_folder = Path(cfg.DMS_MSA_data_path)
    DMS_PDB_folder = Path(cfg.DMS_PDB_data_path)
    output_scores_folder = Path(cfg.output_scores_folder)

    # Load dataset
    DMS_idx = cfg.DMS_idx
    df_ref = pd.read_csv(DMS_reference_file_path)
    if use_multiples:
        df_ref = df_ref[df_ref['includes_multiple_mutants'] == True]
        df_ref['single_index'] = df_ref.index
        df_ref = df_ref.reset_index(drop=True)
    DMS_id = df_ref.loc[DMS_idx, "DMS_id"]
    DMS_msa = df_ref.loc[DMS_idx, "MSA_filename"]
    msa_start = df_ref.loc[DMS_idx, "MSA_start"]
    msa_end = df_ref.loc[DMS_idx, "MSA_end"]
    msa_start_shift = 0
    msa_end_shift = 0
    csv_substitutions_file = (DMS_data_folder / f"{DMS_id}.csv").resolve()
    output_path = output_scores_folder / f"{split_method}/pypef_hybrid/{llm}/{DMS_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.resolve().exists():
        if not cfg.overwrite:
            print(f"Output file already exists: {output_path.resolve()}")
            return
        else:
            print(f"Overwriting existing output file: {output_path.resolve()}")
    else:
        print("Output does not yet exist")

    if llm == "prosst":
        wt_sequence = df_ref.loc[DMS_idx, "target_seq"]
        DMS_pdb = df_ref.loc[DMS_idx, "pdb_file"]
        pdb_range = df_ref.loc[DMS_idx, "pdb_range"]
        pdb_start = int(pdb_range.split('-')[0])
        pdb_end = int(pdb_range.split('-')[1])
        print(f"Substitution effect CSV: {csv_substitutions_file}")
        print(f"Length of wild type sequence (untrimmed): {len(wt_sequence)}")
        print("PDB", (DMS_PDB_folder / DMS_pdb).resolve())
        print(f"PDB range: {pdb_range} [{pdb_start - 1}:{pdb_end}] len: {pdb_end - (pdb_start - 1)}")
        print("MSA:", (DMS_MSA_folder / DMS_msa).resolve())
        print(f"MSA range: {msa_start}-{msa_end} [{msa_start - 1}:{msa_end}] len: {msa_end - (msa_start - 1)}")
        if pdb_start != msa_start or pdb_end != msa_end:
            print("PDB and MSA start and/or end are not matching, trying to trim the MSA "
                  "in the following to match PDB length...")
            msa_start_shift = pdb_start - msa_start    # Assuming that PDB starts later, e.g.:
            msa_end_shift = pdb_end - msa_end          # PDB range: 291-794 [290:794] len: 503, MSA range: 281-804 [280:804] len: 523
            print("msa_start + msa_start_shift:", msa_start + msa_start_shift, msa_start_shift)
            print("msa_end + msa_end_shift:", msa_end + msa_end_shift, msa_end_shift)
        pdb_wt_seq = wt_sequence[pdb_start - 1:pdb_end]
        pdb_seq = str(list(SeqIO.parse(DMS_PDB_folder / DMS_pdb, "pdb-atom"))[0].seq)
        wt_msa_seq = wt_sequence[msa_start + msa_start_shift - 1:msa_end + msa_end_shift]
        assert pdb_wt_seq == pdb_seq, (
            f"PDB sequences do not match: PDB subsequence from CSV data and "
            f"extracted sequence from input PDB file:\n{pdb_wt_seq}\n{pdb_seq}"
        )
        assert wt_msa_seq == pdb_seq, (
            f"PDB sequence and WT sequence from MSA (trimmed from MSA start to end) do not match:\n{wt_msa_seq} "
            f"(len={len(wt_msa_seq)})\n{pdb_seq} (len={len(pdb_seq)})"
        )
    elif llm == "esm1v":
        pass
    else:
        raise RuntimeError("Unknown LLM option.")
    #if DMS_id == "BRCA2_HUMAN_Erwood_2022_HEK293T":
    #    # Disable distance kernel due to sequenc length
    #    cfg.gp.mutation_kernel.use_distances = False
    #    cfg.gp.mutation_kernel.model._target_ = "kermut.model.kernel.Kermut_no_d"

    print(f"Output path: {output_path.resolve()}")

    print(
        f"Using {split_method} split on DMS idx {DMS_idx}: {DMS_id}",
        flush=True,
    )

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = pd.read_csv(csv_substitutions_file)

    # Prepare output
    df_predictions = pd.DataFrame(columns=["fold", "mutant", "y", "y_pred", "y_var"])

    df = df.reset_index(drop=True)
    print('GREMLIN DCA (MSA optimization)...')
    gremlin = GREMLIN(
        alignment=DMS_MSA_folder / DMS_msa, opt_iter=100, optimize=True, 
        msa_start=msa_start_shift, msa_end=msa_end_shift
    )  # For ProSST: Trim MSA according to PDB sequence length

    if llm == "prosst":
        assert gremlin.first_msa_seq.upper() == wt_msa_seq, f"{gremlin.first_msa_seq.upper()}\n   !=\n{wt_msa_seq}"
        n_mismatches, mismatches = get_mismatches(wt_msa_seq, gremlin.first_msa_seq.upper())
        print(f'Ratio of mismatches: {n_mismatches / len(wt_msa_seq)}, N={n_mismatches}, Mismatches="{mismatches}"')
        assert (n_mismatches / len(wt_msa_seq)) <= 0.05
    y_full = df[target_col].values
    seq_full = df[sequence_col].values
    trimmed_seqs = []
    for i, seq in enumerate(seq_full):
        # For ProSST: Trim MSA according to PDB sequence length
        seq = seq[msa_start - 1 + msa_start_shift: msa_end + msa_end_shift]
        trimmed_seqs.append(seq)
    trimmed_seqs = np.asarray(trimmed_seqs)

    x_dca_full = gremlin.collect_encoded_sequences(trimmed_seqs)
    x_dca_full = np.array(x_dca_full)
    y_pred_dca = get_delta_e_statistical_model(x_dca_full, gremlin.x_wt)
    print(f'DCA (unsupervised performance, Spear. corr.): {spearmanr(y_full, y_pred_dca)[0]:.3f}')  
    try:
        unique_folds = df[split_method].unique()
    except KeyError as e:
        raise RuntimeError(f"KeyError: {e}. Available columns: {df.columns.to_list()}")

    for i, test_fold in enumerate(tqdm(unique_folds, disable=not progress_bar)):
        print(f'Split {i+1}/{len(unique_folds)}...')
        # Assign splits
        train_idx = (df[split_method] != test_fold).tolist()
        test_idx = (df[split_method] == test_fold).tolist()
        s_train = trimmed_seqs[train_idx]
        s_test = trimmed_seqs[test_idx]
        x_dca_train = x_dca_full[train_idx]
        x_dca_test = x_dca_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]
        
        print(f"    Test fold: {test_fold}: N_Train={len(y_train)}, N_Test={len(y_test)} "
              f"Test proportion: {len(y_test) / (len(y_train) + len(y_test)):.3f}")
        if llm == "prosst":
            llm_kwargs = prosst_setup(
                wt_seq=pdb_wt_seq, 
                pdb_file=DMS_PDB_folder / DMS_pdb, 
                sequences=s_train, 
                device='cuda'
            )
            vocab = llm_kwargs['prosst']['llm_vocab']
            x_llm_test = np.asarray(prosst_tokenize_sequences(
                sequences=s_test, vocab=vocab, verbose=False))
        elif llm == "esm1v":
            llm_kwargs = esm_setup(sequences=s_train)
            tokenizer = llm_kwargs['esm1v']['llm_tokenizer']
            x_llm_test, _attn_masks = esm_tokenize_sequences(
                sequences=s_test, tokenizer=tokenizer, max_length=len(s_test[0])
            )
        
        if df.shape[0] >= 100000:  # Not CV-training the P-LM on much data but just relying on DCA
            llm_kwargs = None      # Datasets: HIS7_YEAST_Pokusaeva_2019.csv
            x_llm_test = None
            print(f'\nSkipping LLM CV training for dataset {csv_substitutions_file} as it '
                  f'would take up (too) much time...')
        hm = DCALLMHybridModel(
            x_train_dca=np.array(x_dca_train), 
            y_train=y_train,
            llm_model_input=llm_kwargs,
            x_wt=gremlin.x_wt,
            verbose=True
        )
        y_test_pred = hm.hybrid_prediction(
            x_dca=np.array(x_dca_test), 
            x_llm=x_llm_test,
            verbose=True
        )
        print(f"    Performance (Spearman corr.): {spearmanr(y_test, y_test_pred)[0]:.3f}")

        df_pred_fold = pd.DataFrame(
            {
                "fold": test_fold,
                "mutant": df.loc[test_idx, "mutant"],
                "y": y_test,
                "y_pred": y_test_pred
            }
        )
        if df_predictions.empty:
            df_predictions = df_pred_fold
        else:
            df_predictions = pd.concat([df_predictions, df_pred_fold])

    df_predictions.to_csv(output_path, index=False)


if __name__ == "__main__":
    # e.g. run with 
    #   python pgym_cv_benchmark.py split_method=fold_random_5 DMS_idx=3 llm=prosst overwrite=true
    main()
