
# Run me from parent dir:
#   Linux
#       export PYTHONPATH="${PYTHONPATH}:${PWD}" && python -m pytest ./tests/   # -v -m "not (pip_specific or requires_gpu)" --log-cli-level=INFO
#   Windows
#       $env:PYTHONPATH = "${PWD};${env:PYTHONPATH}";python -m pytest .\tests\  # -v -m "not (pip_specific or requires_gpu)" --log-cli-level=INFO
# python -m pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu

# 

import os
seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Export the PYTHONHASHSEED env variable before running this script! 
os.environ['PYTHONHASHSEED'] = str(seed)
import sys
import time
import torch
import numpy as np
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# Very slow
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)
# Even if only using CPU, these settings don't hurt
torch.cuda.manual_seed_all(seed)
# Force deterministic algorithms
torch.use_deterministic_algorithms(True)
# benchmark=False prevents torch from searching for the 
# "fastest" (and often random) convolution algorithm
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import gpytorch
import pytest
import hashlib

from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path, get_regressor_performances
from pypef.dca.gremlin_inference import GREMLIN
from pypef.utils.variant_data import (
    extract_pdb_coords, get_seqs_from_var_name, get_sequences_from_file, 
    get_wt_sequence, split_variants
)
from pypef.plm.inference import (
    plm_inference, esm_setup, prosst_setup, tokenize_sequences, get_plm_embeddings
)
from pypef.hybrid.hybrid_model import DCALLMHybridModel
from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.prosst_lora_tune import (
    get_prosst_models, get_structure_quantizied, 
    prosst_simple_vocab_aa_tokenizer
)
from pypef.plm.utils import hybrid_corr_mse_loss
from pypef.gaussian_process.gauss_opt import get_gp_kernel_model
from pypef.utils.helpers import get_device
from pypef.gaussian_process.kermut.utils import prepare_kermut_inputs
from pypef.gaussian_process.kermut.gp.instantiate_gp import instantiate_gp
from pypef.gaussian_process.kermut.gp.optimize_gp import optimize_gp
from pypef.gaussian_process.kermut.gp.predict import predict

os.environ["TQDM_DISABLE"] = "0"  # "0": show tqdm progress bars, "1": disable progress bars
LOGGING = False
if LOGGING:
    import logging
    package_logger = logging.getLogger('pypef')
    package_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s")
    handler.setFormatter(formatter)
    package_logger.addHandler(handler)


device = ["cpu", get_device()][1]  # Set device to be used

py_ver = sys.version_info
print(f"Python version: {py_ver[0:3]}")
print(f"Torch version: {torch.__version__}")
torch_version = [int(i) for i in torch.__version__.split('+')[0].split('.')]
torch_cpu_or_cuda_version = torch.__version__.split('+')[1]
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA runtime (used by PyTorch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# This checks which CUDA version PyTorch was COMPILED against
print(f"PyTorch compiled with CUDA: {torch.version.cuda}")

csv_blat_ecolx_avgfp = os.path.abspath(
    os.path.join(__file__, '../../datasets/AVGFP/avGFP.csv'
))
msa_file_avgfp = os.path.abspath(os.path.join(
    __file__, '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
))
pdb_file_avgfp = os.path.abspath(os.path.join(
    __file__, '../../datasets/AVGFP/GFP_AEQVI.pdb'
))
wt_seq_file_avgfp = os.path.abspath(os.path.join(
    __file__, '../../datasets/AVGFP/P42212_F64L.fasta'
))
df = pd.read_csv(csv_blat_ecolx_avgfp, sep=';')
mutants = df['variant'].to_list()
y = df['fitness'].to_list()
mutants, y, sequences = get_seqs_from_var_name(
    get_wt_sequence(wt_seq_file_avgfp), split_variants(mutants), y
)
_m_train_avgfp, _m_test_avgfp, s_train_avgfp, s_test_avgfp, y_train_avgfp, y_test_avgfp = train_test_split(
    mutants, sequences, y, train_size=400, test_size=400, random_state=42
)

msa_file_aneh = os.path.abspath(
    os.path.join(__file__, '../../datasets/ANEH/ANEH_jhmmer.a2m'
))
pdb_file_aneh = os.path.abspath(os.path.join(
    __file__, '../../datasets/ANEH/AF-Q9UR30-F1-model_v4.pdb'
))
wt_seq_file_aneh = os.path.abspath(os.path.join(
    __file__, '../../datasets/ANEH/Sequence_WT_ANEH.fasta'
))
ls_b = os.path.abspath(os.path.join(
    __file__, '../../datasets/ANEH/LS_B.fasl'
))
ts_b = os.path.abspath(
    os.path.join(__file__, '../../datasets/ANEH/TS_B.fasl'
))

csv_blat_ecolx_stiffler2015 = os.path.abspath(
    os.path.join(__file__, '../../datasets/BLAT_ECOLX/BLAT_ECOLX_Stiffler_2015.csv'
))
pdb_blat_ecolx = os.path.abspath(
    os.path.join(__file__, '../../datasets/BLAT_ECOLX/BLAT_ECOLX.pdb'
))
wt_seq_file_blat_ecolx = os.path.abspath(
    os.path.join(__file__, '../../datasets/BLAT_ECOLX/blat_ecolx_wt.fasta'
))
df = pd.read_csv(csv_blat_ecolx_stiffler2015)
mutants = df['mutant'].to_list()
sequences = df['mutated_sequence'].to_list()
y = df['DMS_score'].to_list()
m_train_blat, m_test_blat, s_train_blat, s_test_blat, y_train_blat, y_test_blat = train_test_split(
    mutants, sequences, y, train_size=400, test_size=400, random_state=42
)

train_seqs_aneh, _train_vars_aneh, train_ys_aneh = get_sequences_from_file(ls_b)
test_seqs_aneh, _test_vars_aneh, test_ys_aneh = get_sequences_from_file(ts_b)


def get_model_hash(model):
    hash_gen = hashlib.sha256()
    for param in model.parameters():
        hash_gen.update(param.detach().cpu().numpy().tobytes())
    return hash_gen.hexdigest()


def assert_portable_corr(rho, name=""):
    """Sanity-guard for quantities that depend on LoRA fine-tuning.

    LoRA-trained results are NOT portable across GPU architecture / torch /
    cuDNN versions.

    Run-twice reproducibility of the LoRA path on a FIXED machine is covered
    by the `--seed`-based checks in scripts/CLI/run_cli_tests_{linux.sh,win.ps1}.
    """
    assert np.isfinite(rho) and -1.0 <= rho <= 1.0, (
        f"{name} correlation not a finite value in [-1, 1]: {rho}"
    )


def test_gremlin_avgfp():
    print("\n\ntest_gremlin_avgfp()..." + "\n" + "=" * 80 + "\n")
    g = GREMLIN(
        alignment=msa_file_avgfp,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100,
        device=device
    )
    t1 = time.time()
    wt_score = g.get_wt_score()  
    np.testing.assert_almost_equal(wt_score, 952.1102220697624, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)
    t2 = time.time()
    print(f"Function run time: {t2 - t1:.2f} s\n")


def test_hybrid_model_dca_llm_aneh(
        msa=msa_file_aneh, 
        train_seqs=train_seqs_aneh, 
        test_seqs=test_seqs_aneh,
        y_train=train_ys_aneh,
        y_test=test_ys_aneh,
        wt_seq=get_wt_sequence(wt_seq_file_aneh),
        pdb_file=pdb_file_aneh
):
    t1 = time.time()
    print("\n\ntest_hybrid_model_dca_llm_aneh()..." + "\n" + "=" * 80 + "\n")
    g = GREMLIN(
        alignment=msa,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100,
        device=device
    )
    wt_score = g.get_wt_score()
    np.testing.assert_almost_equal(wt_score, 1743.2087199198131, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)
    y_pred = g.get_scores(np.append(train_seqs, test_seqs))
    np.testing.assert_almost_equal(
        spearmanr(np.append(y_train, y_test), y_pred)[0], 
        -0.5528510930046211, 
        decimal=7
    )
    x_dca_train = g.get_scores(train_seqs, encode=True)
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, np.sum(x_dca_train, axis=1))[0],
        -0.5556053466180598,
        decimal=7
    )
    assert len(train_seqs[0]) == len(g.wt_seq)
    print('len(aneh_wt_seq)', len(wt_seq))

    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
        model="facebook/esm1v_t33_650M_UR90S_3", seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70")
    esm_base_model.eval()
    esm_base_model = esm_base_model.to(device)
    # ESM lm_head decoder must stay tied to the input word embeddings. 
    # A globally forced tie_word_embeddings=False leaves the ESM decoder 
    # randomly initialized -> near-random zero-shot log-likelihoods.
    assert torch.equal(
        esm_base_model.esm.embeddings.word_embeddings.weight,
        esm_base_model.lm_head.decoder.weight,
    ), "ESM lm_head decoder is not tied to word embeddings — zero-shot scoring would be random!"
    total_sum = sum(p.sum().item() for p in esm_base_model.parameters())
    print('Total model parameter sum (ESM):', total_sum)
    np.testing.assert_allclose(total_sum, -15522.971457719803, atol=1e-1)
    model_hash = get_model_hash(esm_base_model)
    print(f"Model Hash: {model_hash}")
    x_esm, esm_attention_mask = tokenize_sequences(
        train_seqs, esm_tokenizer, max_length=len(wt_seq) + 2
    )
    # Tokenize WT sequence once
    wt_tokens_esm, _ = tokenize_sequences(
            [wt_seq],
            esm_tokenizer,
            max_length=len(wt_seq) + 2
    )
    wt_tokens_esm = torch.tensor(wt_tokens_esm[0], dtype=torch.long)  # shape (L,)
    y_pred_esm = plm_inference(
        tokenized_sequences=x_esm, wt_input_ids=wt_tokens_esm, 
        attention_mask=esm_attention_mask, model=esm_base_model,
        device=device
    ).cpu()

    np.testing.assert_almost_equal(
        spearmanr(y_train, y_pred_esm)[0], 
         -0.713214007088901, 
        decimal=7
    )

    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models(
        seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8")
    prosst_base_model.eval()
    prosst_base_model = prosst_base_model.to(device)
    # Guard intent: ProSST is intentionally untied but its decoder head is
    # weight-copied from the embeddings (parity), so decoder values must equal
    # the embeddings — otherwise zero-shot scoring would be random.
    assert torch.equal(
        prosst_base_model.prosst.embeddings.word_embeddings.weight,
        prosst_base_model.cls.predictions.decoder.weight,
    ), "ProSST decoder head is not weight-matched to embeddings — zero-shot scoring would be random!"
    total_sum = sum(p.sum().item() for p in prosst_base_model.parameters())
    #for name, param in prosst_base_model.named_parameters():
    #    s = param.sum().item()
    #    print(f"{name}: {s:.10f}")
    print('Total model parameter sum (ProSST):', total_sum)
    np.testing.assert_allclose(total_sum, -36152.75924, atol=1e-1)
    model_hash = get_model_hash(prosst_base_model)
    print(f"Model Hash: {model_hash}")
    assert model_hash == "f4ad803baa080bf43414365b653ebbf3798826a91c30b268fe97dc810ee2ef45"

    wt_tokens_prosst, prosst_attention_mask, wt_structure_tokens_prosst = get_structure_quantizied(
        pdb_file, prosst_tokenizer, wt_seq, device=device)
    
    assert wt_structure_tokens_prosst.shape[1] == wt_tokens_prosst.shape[1]
    
    # [ 1, 13, 18,  3, 15,  7,  3, 11,  7, 15, 18, 18,  3, 18, 10, 18, 15, 14,
    #  ...
    #  21, 16, 11,  2]
    #print(wt_input_ids.cpu().numpy())
    seq_tok_sum  = wt_tokens_prosst.cpu().numpy().sum()
    seq_tok_sha = hashlib.sha256(wt_tokens_prosst.cpu().numpy().tobytes()).hexdigest()
    assert seq_tok_sum == 4781 and seq_tok_sha == "a18fbcf68f4909d9e25f130a254706f0e8234f171458630e68431d1137e43280"

    # [   1 1940 1537 1776  530  853  497 1227  200 1605 1160  878  473 1902
    #  ...
    #  1247  750 1174  531  135 1393  471    2]]
    #print(wt_structure_input_ids.cpu().numpy())
    struct_tok_sum = wt_structure_tokens_prosst.cpu().numpy().sum()
    struct_tok_sha = hashlib.sha256(wt_structure_tokens_prosst.cpu().numpy().tobytes()).hexdigest()
    assert struct_tok_sum == 417050 and struct_tok_sha == "df077674dd7c9054328537c1f7bd9c8e9bf80d59f287216ed3c2eeeeb7b8a39b"

    x_prosst, _prosst_attention_mask = tokenize_sequences(
        sequences=train_seqs, 
        tokenizer=prosst_tokenizer, 
        max_length=len(wt_seq) + 2
    )

    y_pred_prosst = plm_inference(
        tokenized_sequences=x_prosst, wt_input_ids=wt_tokens_prosst, 
        attention_mask=prosst_attention_mask, model=prosst_base_model, 
        wt_structure_input_ids=wt_structure_tokens_prosst, device=device
    ).cpu()

    assert spearmanr(y_train, y_pred_prosst)[0] in [-0.7425657069861902, -0.5022957688493356]

    x_dca_test = g.get_scores(test_seqs, encode=True)

    for i, setup in enumerate(['ESM', 'ProSST']):
        print(f'~~~ {setup} ~~~')
        if setup == 'ESM':
            llm_dict = esm_setup(
                wt_seq=wt_seq, sequences=train_seqs, model="facebook/esm1v_t33_650M_UR90S_3",
                seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70", device=device, verbose=True
            )
        else:  # elif setup == 'ProSST':
            llm_dict = prosst_setup(
                wt_seq=wt_seq, pdb_file=pdb_file, sequences=train_seqs, 
                seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8", device=device, verbose=True
            )
        x_llm_test, _ = tokenize_sequences(test_seqs, llm_dict[['esm', 'prosst'][i]]['llm_tokenizer'])

        if i == 0:
            y_test_pred_1 = plm_inference(
                tokenized_sequences=x_llm_test, wt_input_ids=wt_tokens_esm, 
                attention_mask=esm_attention_mask, model=esm_base_model,
                device=device
            ).cpu()
        if i == 1:
            y_test_pred_1 = plm_inference(
                tokenized_sequences=x_llm_test, wt_input_ids=wt_tokens_prosst, 
                attention_mask=prosst_attention_mask, model=prosst_base_model,
                wt_structure_input_ids=wt_structure_tokens_prosst,
                device=device
            ).cpu()

        # y_test_pred_1: SignificanceResult(statistic=np.float64(-0.7050183991342079), pvalue=np.float64(1.3613669161432091e-05)) 30
        print('y_test_pred_1:', spearmanr(y_test, y_test_pred_1)[0], len(test_ys_aneh))

        hm = DCALLMHybridModel(
            x_train_dca=np.array(x_dca_train), 
            y_train=y_train,
            llm_model_input=llm_dict,
            x_dca_wt=g.x_wt,
            seed=42,
            lora_train=True,
            n_epochs=5,  # Training (only) the LoRA model
            device=device,
        )

        y_test_pred_2 = plm_inference(
            tokenized_sequences=x_llm_test,
            wt_input_ids=llm_dict[['esm', 'prosst'][i]]['wt_input_ids'],
            attention_mask=llm_dict[['esm', 'prosst'][i]]['llm_attention_mask'],
            model=llm_dict[['esm', 'prosst'][i]]['llm_base_model'],
            wt_structure_input_ids=llm_dict.get(['esm', 'prosst'][i], {}).get('wt_structure_input_ids'),
            device=device
        ).cpu()

        # y_llm_ttest (y_test_pred_2): SignificanceResult(statistic=np.float64(0.40948038333552483), pvalue=np.float64(0.024635229903827348)) 30
        print('y_llm_ttest (y_test_pred_2):', spearmanr(y_test, y_test_pred_2)[0], len(test_ys_aneh))

        y_test_pred_lora = plm_inference(
            tokenized_sequences=x_llm_test,
            wt_input_ids=llm_dict[['esm', 'prosst'][i]]['wt_input_ids'],
            attention_mask=llm_dict[['esm', 'prosst'][i]]['llm_attention_mask'],
            model=llm_dict[['esm', 'prosst'][i]]['llm_model'],
            wt_structure_input_ids=llm_dict.get(['esm', 'prosst'][i], {}).get('wt_structure_input_ids'),
            device=device
        ).cpu()

        # y_llm_ttest_lora (y_test_pred_3): SignificanceResult(statistic=np.float64(0.40948038333552483), pvalue=np.float64(0.024635229903827348)) 30
        print('y_llm_ttest_lora (y_test_pred_3):', spearmanr(y_test, y_test_pred_lora)[0], len(test_ys_aneh))
        print(f'Train-on-train: {spearmanr(hm.y_ttrain, hm.y_llm_ttrain)[0]:.3f} (unsupervised)'
              f'--> {spearmanr(hm.y_ttrain, hm.y_llm_lora_ttrain)[0]:.3f} (supervised)  | len = {len(hm.y_ttrain)}')
        print(f'Train-on-test: {spearmanr(hm.y_ttest, hm.y_llm_ttest)[0]:.3f} (unsupervised)'
              f'--> {spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0]:.3f} (supervised) | len = {len(hm.y_ttest)}')
        
        torch.testing.assert_close(y_test_pred_1, y_test_pred_2)
        assert not np.allclose(y_test_pred_lora, y_test_pred_1, rtol=1e-5, atol=1e-8)
        assert not np.allclose(hm.y_llm_ttrain, hm.y_llm_lora_ttrain, rtol=1e-5, atol=1e-8)
        assert not np.allclose(hm.y_llm_ttest, hm.y_llm_lora_ttest, rtol=1e-5, atol=1e-8)

        x_llm_input = {
                ['esm', 'prosst'][i]: x_llm_test,
        }

        y_pred_test, _individual_predictions = hm.hybrid_prediction(x_dca=x_dca_test, x_llm_dict=x_llm_input)
        print(hm.betas, hm.ridge_opt)
        print('hm.y_dca_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ttest)[0], len(hm.y_ttest))
        print('hm.y_dca_ridge_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest)[0], len(hm.y_ttest))
        print('hm.y_llm_ttest:', spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], len(hm.y_ttest))
        print('hm.y_llm_lora_ttest:', spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], len(hm.y_ttest))
        print('Hybrid prediction:', spearmanr(y_test, y_pred_test)[0], len(y_test))
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ttest)[0], -0.5342743713116743, 
            decimal=7
        )
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest)[0], 0.717333573331078, 
            decimal=7
        )

        # Portable golden values: base-PLM zero-shot (y_llm_ttrain/y_llm_ttest)
        # reproduces across devices. LoRA-derived quantities (y_llm_lora_*, and
        # the hybrid y_pred_test that weights the LoRA predictor in) do NOT —
        # guard them with assert_portable_corr() instead of a golden value.
        if i == 0:
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttrain, hm.y_llm_ttrain)[0], -0.6825218561297186
            )
            assert_portable_corr(
                spearmanr(hm.y_ttrain, hm.y_llm_lora_ttrain)[0], "ESM y_llm_lora_ttrain"
            )
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], -0.7704181041760417
            )
            assert_portable_corr(
                spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], "ESM y_llm_lora_ttest"
            )
            assert_portable_corr(
                spearmanr(y_test, y_pred_test)[0], "ESM hybrid y_pred_test"
            )

        elif i == 1:
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttrain, hm.y_llm_ttrain)[0], -0.6814974117251794
            )
            assert_portable_corr(
                spearmanr(hm.y_ttrain, hm.y_llm_lora_ttrain)[0], "ProSST y_llm_lora_ttrain"
            )
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], -0.8330644449247571
            )
            assert_portable_corr(
                spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], "ProSST y_llm_lora_ttest"
            )
            assert_portable_corr(
                spearmanr(y_test, y_pred_test)[0], "ProSST hybrid y_pred_test"
            )
    t2 = time.time()
    print(f"Function run time: {t2 - t1:.2f} s\n")


def test_hybrid_model_dca_llm_avgfp(
        msa=msa_file_avgfp, 
        train_seqs=s_train_avgfp, 
        test_seqs=s_test_avgfp,
        y_train=y_train_avgfp,
        y_test=y_test_avgfp,
        wt_seq=get_wt_sequence(wt_seq_file_avgfp),
        pdb_file=pdb_file_avgfp,
        nolora_test=False
):
    t1 = time.time()
    print("\n\ntest_hybrid_model_dca_llm_avgfp()..." + "\n" + "=" * 80 + "\n")
    g = GREMLIN(
        alignment=msa,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100,
        device=device
    )
    wt_score = g.get_wt_score()
    np.testing.assert_almost_equal(wt_score, 952.1101980988492, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)
    y_pred = g.get_scores(np.append(train_seqs, test_seqs))
    np.testing.assert_almost_equal(
        spearmanr(np.append(y_train, y_test), y_pred)[0], 
        0.6689211155149526, 
        decimal=7
    )
    x_dca_train = g.get_scores(train_seqs, encode=True)
    np.testing.assert_almost_equal(
        spearmanr(y_train, np.sum(x_dca_train, axis=1))[0],
        0.6612121950762191,
        decimal=7
    )
    assert len(train_seqs[0]) == len(g.wt_seq)

    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
        model="facebook/esm1v_t33_650M_UR90S_3", seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70")
    esm_base_model.eval()
    esm_base_model = esm_base_model.to(device)
    assert torch.equal(
        esm_base_model.esm.embeddings.word_embeddings.weight,
        esm_base_model.lm_head.decoder.weight,
    ), "ESM lm_head decoder is not tied to word embeddings — zero-shot scoring would be random!"
    x_esm, esm_attention_mask = tokenize_sequences(
        train_seqs, esm_tokenizer, max_length=len(wt_seq) + 2
    )
    # Tokenize WT sequence once
    wt_tokens, _ = tokenize_sequences(
            [wt_seq],
            esm_tokenizer,
            max_length=len(wt_seq) + 2
    )
    wt_tokens = torch.tensor(wt_tokens[0], dtype=torch.long)  # shape (L,)
    y_pred_esm = plm_inference(
        tokenized_sequences=x_esm, wt_input_ids=wt_tokens, 
        attention_mask=esm_attention_mask, model=esm_base_model,
        device=device
    ).cpu()
    np.testing.assert_almost_equal(
        spearmanr(y_train, y_pred_esm)[0], 
        0.5294118088238051, 
        decimal=7
    )

    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models(
        seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8"
    )
    prosst_base_model.eval()
    prosst_base_model = prosst_base_model.to(device)
    assert torch.equal(
        prosst_base_model.prosst.embeddings.word_embeddings.weight,
        prosst_base_model.cls.predictions.decoder.weight,
    ), "ProSST decoder head is not weight-matched to embeddings — zero-shot scoring would be random!"
    wt_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
        pdb_file, prosst_tokenizer, wt_seq, device=device
    )
    
    assert wt_structure_input_ids.shape[1] == wt_input_ids.shape[1], (
        f"{wt_structure_input_ids.shape[1]} != {wt_input_ids.shape[1]}"
    )
    
    # [ 1, 13, 18,  3, 15,  7,  3, 11,  7, 15, 18, 18,  3, 18, 10, 18, 15, 14,
    #  ...
    #  21, 16, 11,  2]
    #print(wt_input_ids.cpu().numpy())
    seq_tok_sum  = wt_input_ids.cpu().numpy().sum()
    seq_tok_sha = hashlib.sha256(wt_input_ids.cpu().numpy().tobytes()).hexdigest()
    assert seq_tok_sum == 2877 and seq_tok_sha == "b58b72f97ad60af69ff4f69d790322219d07ab49a0a120e5c289d9d040f3e73c"

    # [   1 1940 1537 1776  530  853  497 1227  200 1605 1160  878  473 1902
    #  ...
    #  1247  750 1174  531  135 1393  471    2]]
    #print(wt_structure_input_ids.cpu().numpy())
    struct_tok_sum = wt_structure_input_ids.cpu().numpy().sum()
    struct_tok_sha = hashlib.sha256(wt_structure_input_ids.cpu().numpy().tobytes()).hexdigest()
    assert struct_tok_sum == 268652 and struct_tok_sha == "ebeb2c92a03155aaba35460b39f05c9b2e9cb0d98a79d5c9795c70d8f98f9e3e"

    x_prosst, _prosst_attention_mask = tokenize_sequences(
        sequences=train_seqs, 
        tokenizer=prosst_tokenizer, 
        max_length=len(wt_seq) + 2
    )

    y_pred_prosst = plm_inference(
        tokenized_sequences=x_prosst, wt_input_ids=wt_input_ids, 
        attention_mask=prosst_attention_mask, model=prosst_base_model, 
        wt_structure_input_ids=wt_structure_input_ids, device=device
    ).cpu()

    print(spearmanr(y_train, y_pred_prosst)[0])
    #assert spearmanr(y_train, y_pred_prosst)[0] in [-0.7425657069861902, -0.5022957688493356]
    

    x_dca_test = g.get_scores(test_seqs, encode=True)
    for i, setup in enumerate(['ESM', 'ProSST', 'Ensemble']):
        print(f'~~~ {setup} ~~~')
        if setup == 'ESM':
            llm_dict_esm = esm_setup(
                wt_seq=wt_seq, sequences=train_seqs, model="facebook/esm1v_t33_650M_UR90S_3",
                seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70", device=device, verbose=True
            )
            llm_dict = llm_dict_esm
        elif setup == 'ProSST':  # elif setup == 'ProSST':
            llm_dict_prosst = prosst_setup(
                wt_seq=wt_seq, pdb_file=pdb_file, sequences=train_seqs, 
                seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8", device=device, verbose=True
            )
            llm_dict = llm_dict_prosst
        else:
            llm_dict_ensemble = {**llm_dict_esm, **llm_dict_prosst}
            llm_dict = llm_dict_ensemble

        hm = DCALLMHybridModel(
            x_train_dca=np.array(x_dca_train), 
            y_train=y_train,
            llm_model_input=llm_dict,
            x_dca_wt=g.x_wt,
            sequences=train_seqs,
            wt_sequence=wt_seq,
            seed=42,
            lora_train=True,
            gauss_opt=True,
            pdb_struct=pdb_file,
            n_epochs=5,
            device=device
        )

        if i == 0:
            x_llm_test_esm, _ = tokenize_sequences(test_seqs, llm_dict[['esm', 'prosst'][i]]['llm_tokenizer'])
            x_llm_test = x_llm_test_esm
            x_llm_input = {'esm': x_llm_test_esm}
        elif i == 1:
            x_llm_test_prosst, _ = tokenize_sequences(test_seqs, llm_dict[['esm', 'prosst'][i]]['llm_tokenizer'])
            x_llm_test = x_llm_test_prosst
            x_llm_input = {'prosst': x_llm_test_prosst}

        if i in [0, 1]:
            y_test_pred = plm_inference(
                tokenized_sequences=x_llm_test,
                wt_input_ids=llm_dict[['esm', 'prosst'][i]]['wt_input_ids'],
                attention_mask=llm_dict[['esm', 'prosst'][i]]['llm_attention_mask'],
                model=llm_dict[['esm', 'prosst'][i]]['llm_base_model'],
                wt_structure_input_ids=llm_dict.get(['esm', 'prosst'][i], {}).get('wt_structure_input_ids'),
                device=device
            ).cpu()

        if i == 2:
            x_llm_input = {
                'esm': x_llm_test_esm,
                'prosst': x_llm_test_prosst
            }

        print('y_llm_ttest:', spearmanr(y_test, y_test_pred)[0], len(test_ys_aneh))
        print(f'Train-on-train: {spearmanr(hm.y_ttrain, hm.y_llm_ttrain)[0]:.3f} (unsupervised)'
              f'--> {spearmanr(hm.y_ttrain, hm.y_llm_lora_ttrain)[0]:.3f} (supervised)  | len = {len(hm.y_ttrain)}')
        print(f'Train-on-test: {spearmanr(hm.y_ttest, hm.y_llm_ttest)[0]:.3f} (unsupervised)'
              f'--> {spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0]:.3f} (supervised) | len = {len(hm.y_ttest)}')

        y_pred_test, _ = hm.hybrid_prediction(x_dca=x_dca_test, x_llm_dict=x_llm_input, sequences=test_seqs)
        print('Weights (beta\'s):', hm.betas, 'Regressor:', hm.ridge_opt)
        print('hm.y_dca_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ttest)[0], len(hm.y_ttest))
        print('hm.y_dca_ridge_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest)[0], len(hm.y_ttest))
        print('hm.y_llm_ttest:', spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], len(hm.y_ttest))
        print('hm.y_llm_lora_ttest:', spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], len(hm.y_ttest))
        print('hm.y_gp_opt_ttest:', spearmanr(hm.y_ttest, hm.y_gp_opt_ttest)[0], len(hm.y_ttest))
        print('hm.all_betas:', str(hm.all_betas).replace('\n', ''), len(hm.all_betas))
        print('Hybrid prediction:', spearmanr(y_test, y_pred_test)[0], len(y_test))
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ttest)[0], 0.5948787474579608, 
            decimal=7
        )
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest)[0], 0.6459513240471454, 
            decimal=7
        )

        # Portable golden values: DCA, base-PLM zero-shot, and GP reproduce
        # bit-for-bit across GPU/torch/cuDNN versions. LoRA-derived quantities
        # (y_llm_lora_ttest, and the hybrid y_pred_test that weights the LoRA
        # predictor in) do NOT — see assert_portable_corr().
        if i == 0:
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], 0.4626402221687696
            )
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_gp_opt_ttest)[0], 0.6719074588353634
            )
            assert_portable_corr(
                spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], "ESM y_llm_lora_ttest"
            )
            assert_portable_corr(
                spearmanr(y_test, y_pred_test)[0], "ESM hybrid y_pred_test"
            )

        elif i == 1:
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], 0.6459731910520218
            )
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_gp_opt_ttest)[0], 0.7626992630819357,
                decimal=2
            )
            assert_portable_corr(
                spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], "ProSST y_llm_lora_ttest"
            )
            assert_portable_corr(
                spearmanr(y_test, y_pred_test)[0], "ProSST hybrid y_pred_test"
            )

        elif i == 2:
            np.testing.assert_almost_equal(
                spearmanr(hm.y_ttest, hm.y_gp_opt_ttest)[0], 0.7626992630819357,
                decimal=2
            )
            assert_portable_corr(
                spearmanr(y_test, y_pred_test)[0], "Ensemble hybrid y_pred_test"
            )

        if nolora_test:  # Over 300 min on GitHub Actions tests, skipping for now by default
            # LoRA-free hybrid (DCA + ridge + base-PLM + GP)
            hm_nolora = DCALLMHybridModel(
                x_train_dca=np.array(x_dca_train),
                y_train=y_train,
                llm_model_input=llm_dict,
                x_dca_wt=g.x_wt,
                sequences=train_seqs,
                wt_sequence=wt_seq,
                seed=42,
                lora_train=False,
                gauss_opt=True,
                pdb_struct=pdb_file,
                n_epochs=5,
                device=device
            )
            y_pred_test_nolora, _ = hm_nolora.hybrid_prediction(
                x_dca=x_dca_test, x_llm_dict=x_llm_input, sequences=test_seqs
            )
            nolora_rho = spearmanr(y_test, y_pred_test_nolora)[0]
            print(f'LoRA-free ({setup}) hybrid feature_names:', hm_nolora.feature_names)
            print(f'LoRA-free ({setup}) hybrid prediction:', nolora_rho, len(y_test))
            assert not any("lora" in fn.lower() for fn in hm_nolora.feature_names), (
                f"LoRA predictor leaked into a lora_train=False hybrid: {hm_nolora.feature_names}"
            )
            # Portable golden values: DCA, ridge, base-PLM zero-shot, and GP (torch.optim.Adam 
            # on CPU) all reproduce (but not bit-for-bit?) across torch versions (verified 
            # 2.12.1 through 2.13.0, cpu and cu13x builds).
            nolora_golden = [
                0.6887306170663565,   # ESM
                0.7676185476159226,   # ProSST, often 0.767617797611235
                0.7693334333339582,   # Ensemble (ESM + ProSST)
            ][i]
            np.testing.assert_almost_equal(nolora_rho, nolora_golden, decimal=3)
    t2 = time.time()
    print(f"Function run time: {t2 - t1:.2f} s\n")


def test_dataset_b_results():
    t1 = time.time()
    print("\n\ntest_dataset_b_results()..." + "\n" + "=" * 80 + "\n")
    aaindex = "WOLR810101.txt"
    x_fft_train, _ = AAIndexEncoding(
        full_aaidx_txt_path(aaindex), train_seqs_aneh
    ).collect_encoded_sequences()
    x_fft_test, _ = AAIndexEncoding(
        full_aaidx_txt_path(aaindex), test_seqs_aneh
    ).collect_encoded_sequences()
    performances = get_regressor_performances(
        x_learn=x_fft_train,
        x_test=x_fft_test,
        y_learn=train_ys_aneh,
        y_test=test_ys_aneh,
        regressor='pls_loocv'
    )  
    # Dataset B PLS_LOOCV results: R², RMSE, NRMSE, Pearson's r, Spearman's rho 
    # RMSE, in Python 3.10 14.669 and from Python 3.11 on 14.17:
    np.testing.assert_almost_equal(performances[1], 14.48, decimal=0)
    # R²
    np.testing.assert_almost_equal(performances[0], 0.72, decimal=2)
    #  NRMSE, Pearson's r, Spearman's rho
    np.testing.assert_almost_equal(performances[2:5], [0.52, 0.86, 0.89], decimal=2)
    t2 = time.time()
    print(f"Function run time: {t2 - t1:.2f} s\n")


@pytest.mark.requires_gpu
def test_plm_corr_blat_ecolx():
    t1 = time.time()
    print("\n\ntest_plm_corr_blat_ecolx() [CUDA]..." + "\n" + "=" * 80 + "\n")
    blat_ecolx_wt_seq = get_wt_sequence(wt_seq_file_blat_ecolx)
    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models(
        seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8")
    prosst_vocab = prosst_tokenizer.get_vocab()
    prosst_base_model = prosst_base_model.to("cuda")
    assert torch.equal(
        prosst_base_model.prosst.embeddings.word_embeddings.weight,
        prosst_base_model.cls.predictions.decoder.weight,
    ), "ProSST decoder head is not weight-matched to embeddings — zero-shot scoring would be random!"
    df = pd.read_csv(csv_blat_ecolx_stiffler2015)
    sequences = df['mutated_sequence'].to_list()
    y_true = df['DMS_score'].to_list()
    for x in ['facebook/esm1v_t33_650M_UR90S_3']:
        esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
            model=x, seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70")
        esm_base_model = esm_base_model.to("cuda")
        assert torch.equal(
            esm_base_model.esm.embeddings.word_embeddings.weight,
            esm_base_model.lm_head.decoder.weight,
        ), "ESM lm_head decoder is not tied to word embeddings — zero-shot scoring would be random!"
        x_esm, esm_attention_mask = tokenize_sequences(
            sequences, esm_tokenizer, max_length=len(blat_ecolx_wt_seq) + 2)
        # Tokenize WT sequence once
        wt_tokens, _ = tokenize_sequences(
            [blat_ecolx_wt_seq],
            esm_tokenizer,
            max_length=len(blat_ecolx_wt_seq) + 2
        )
        wt_tokens = torch.tensor(wt_tokens[0], dtype=torch.long)  # shape (L,)
        
        y_esm = plm_inference(
            tokenized_sequences=x_esm,
            wt_input_ids=wt_tokens,
            attention_mask=esm_attention_mask,
            model=esm_base_model,
            mask_token_id=esm_tokenizer.mask_token_id,
            inference_type='mutation-masking',
            batch_size=5,
            train=False,
            device="cuda",
            verbose=True
        ).cpu()
        print(f'{x}: ESM (unsupervised performance mutation-masking): '  
              f'{spearmanr(y_true, y_esm.cpu())[0]}')
        np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.6360220086687192, decimal=6)
        
        y_esm = plm_inference(
            tokenized_sequences=x_esm,
            wt_input_ids=wt_tokens,
            attention_mask=esm_attention_mask,
            model=esm_base_model,
            mask_token_id=None, # do not define for unmasked
            inference_type='wt-marginal',
            batch_size=5,
            train=False,
            device="cuda",
            verbose=True
        ).cpu()
        print(f'{x}: ESM (unsupervised performance wt-marginal): '  
              f'{spearmanr(y_true, y_esm.cpu())[0]}')
        np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.6494103102128445, decimal=6)

        y_esm = plm_inference(
            tokenized_sequences=x_esm,
            wt_input_ids=wt_tokens,
            attention_mask=esm_attention_mask,
            model=esm_base_model,
            mask_token_id=None, # do not define for unmasked
            inference_type='full-sequence',
            batch_size=5,
            train=False,
            device="cuda",
            verbose=True
        ).cpu()
        print(f'{x}: ESM (unsupervised performance full-sequence): '  
              f'{spearmanr(y_true, y_esm.cpu())[0]}')
        np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.6395329171141655, decimal=6)
        
        #y_esm = plm_inference(
        #    tokenized_sequences=x_esm,
        #    wt_input_ids=wt_tokens,
        #    attention_mask=esm_attention_mask,
        #    model=esm_base_model,
        #    mask_token_id=esm_tokenizer.mask_token_id,
        #    inference_type='full-masking',
        #    batch_size=5,
        #    train=False,
        #    device="cuda",
        #    verbose=True
        #).cpu()
        #print(f'{x}: ESM (unsupervised performance): '  
        #      f'{spearmanr(y_true, y_esm.cpu())[0]}')
        #np.testing.assert_almost_equal(spearmanr(y_true, y_esm.cpu())[0], 0.666666666666666, decimal=6)

    wt_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
        pdb_blat_ecolx, prosst_tokenizer, blat_ecolx_wt_seq, device="cuda"
    )
    x_prosst2 = prosst_simple_vocab_aa_tokenizer(sequences, prosst_vocab)

    assert wt_structure_input_ids.shape[1] == wt_input_ids.shape[1]

    x_prosst, prosst_attention_mask_ = tokenize_sequences(
        sequences=sequences, 
        tokenizer=prosst_tokenizer, 
        max_length=len(blat_ecolx_wt_seq) + 2
    )
    assert x_prosst[0][1:-1] == x_prosst2.tolist()[0][1:-1], (
        f"{x_prosst[0][1:-1]} != {x_prosst2.tolist()[0][1:-1]}")
    assert prosst_attention_mask.tolist()[0] == prosst_attention_mask_, (
        f"{prosst_attention_mask.tolist()[0]} != {prosst_attention_mask_}")

    y_prosst = plm_inference(
            tokenized_sequences=x_prosst,
            wt_input_ids=wt_input_ids,
            attention_mask=prosst_attention_mask,
            model=prosst_base_model,
            mask_token_id=prosst_tokenizer.mask_token_id,
            inference_type='mutation-masking',
            wt_structure_input_ids=wt_structure_input_ids,
            batch_size=5,
            train=False,
            device="cuda",
            verbose=True   
    ).cpu()
    print(f'ProSST (unsupervised performance mutation-masking): '  # ProSST not made/trained for Masked-LM
          f'{spearmanr(y_true, y_prosst.cpu())[0]}')
    #np.testing.assert_almost_equal(spearmanr(y_true, y_prosst.cpu())[0], 0.607137337377509, decimal=6)  # 0.020519849040693375

    y_prosst = plm_inference(
            tokenized_sequences=x_prosst,
            wt_input_ids=wt_input_ids,
            attention_mask=prosst_attention_mask,
            model=prosst_base_model,
            mask_token_id=None,  # do not define for unmasked
            inference_type='wt-marginal',
            wt_structure_input_ids=wt_structure_input_ids,
            batch_size=5,
            train=False,
            device="cuda",
            verbose=True        
    ).cpu()
    print(f'ProSST (unsupervised performance wt-marginal): '  # ProteinGym: ProSST: 0.760
          f'{spearmanr(y_true, y_prosst.cpu())[0]}')
    np.testing.assert_almost_equal(spearmanr(y_true, y_prosst.cpu())[0], 0.7438456514605788, decimal=3)

    y_prosst = plm_inference(
            tokenized_sequences=x_prosst,
            wt_input_ids=wt_input_ids,
            attention_mask=prosst_attention_mask,
            model=prosst_base_model,
            mask_token_id=None,  # do not define for unmasked
            inference_type='full-sequence',
            wt_structure_input_ids=wt_structure_input_ids,
            batch_size=5,
            train=False,
            device="cuda",
            verbose=True        
    ).cpu()
    print(f'ProSST (unsupervised performance full-sequence): '
          f'{spearmanr(y_true, y_prosst.cpu())[0]}')
    np.testing.assert_almost_equal(spearmanr(y_true, y_prosst.cpu())[0], 0.5651544861475942, decimal=3)

    #y_prosst = plm_inference(
    #        tokenized_sequences=x_prosst,
    #        wt_input_ids=wt_input_ids,
    #        attention_mask=prosst_attention_mask,
    #        model=prosst_base_model,
    #        mask_token_id=prosst_tokenizer.mask_token_id,
    #        inference_type='full-masking',
    #        wt_structure_input_ids=wt_structure_input_ids,
    #        batch_size=5,
    #        train=False,
    #        device="cuda",
    #        verbose=True        
    #).cpu()
    #print(f'ProSST (unsupervised performance): '  # ProteinGym: ProSST: 0.760
    #      f'{spearmanr(y_true, y_prosst.cpu())[0]}')

    t2 = time.time()
    print(f"Function run time: {t2 - t1:.2f} s\n")


def test_gaussian_process_opt():
    t1 = time.time()
    print("\n\ntest_gaussian_process_opt()..." + "\n" + "=" * 80 + "\n")
    print("Getting ProSST models")
    wt_seq = get_wt_sequence(wt_seq_file_blat_ecolx)
    prosst_base_model, _prosst_lora_model, prosst_tokenizer, _prosst_optimizer = get_prosst_models(
        seed=seed, revision="e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8")
    prosst_base_model = prosst_base_model.to(device)
    assert torch.equal(
        prosst_base_model.prosst.embeddings.word_embeddings.weight,
        prosst_base_model.cls.predictions.decoder.weight,
    ), "ProSST decoder head is not weight-matched to embeddings — zero-shot scoring would be random!"

    esm_base_model, _esm_lora_model, esm_tokenizer, _esm_optimizer = get_esm_models(
         model="facebook/esm1v_t33_650M_UR90S_3", seed=seed, revision="0b00fd112e63f6b5e70a9cd8484d4e660312ce70")
    assert torch.equal(
        esm_base_model.esm.embeddings.word_embeddings.weight,
        esm_base_model.lm_head.decoder.weight,
    ), "ESM lm_head decoder is not tied to word embeddings — zero-shot scoring would be random!"

    wt_prosst_input_ids, prosst_attention_mask, wt_structure_input_ids = get_structure_quantizied(
        pdb_blat_ecolx, prosst_tokenizer, wt_seq, device=device, verbose=True
    )

    wt_esm_input_ids, _esm_attention_mask = tokenize_sequences([wt_seq], esm_tokenizer)
    wt_esm_input_ids = torch.tensor(wt_esm_input_ids[0], dtype=torch.long)  # shape (L,)

    x_prosst_tok_train, _prosst_attention_mask = tokenize_sequences(s_train_blat, prosst_tokenizer)
    print("Getting ProSST embeddings...")
    x_prosst_emb_train = get_plm_embeddings(
        x_prosst_tok_train, prosst_base_model, wt_prosst_input_ids, 
        prosst_attention_mask, mode="mean", wt_structure_input_ids=wt_structure_input_ids, 
        device=device, verbose=True
    )

    x_esm_tok_train, esm_attention_mask = tokenize_sequences(s_train_blat, esm_tokenizer)
    print("Getting ESM embeddings...")
    x_esm_emb_train = get_plm_embeddings(
        x_esm_tok_train, esm_base_model, wt_esm_input_ids, esm_attention_mask, 
        mode="mean", device=device, verbose=True
    )

    y_train = torch.tensor(y_train_blat).float().to(device)
    y_test = torch.tensor(y_test_blat).float().to(device)

    x_prosst_tok_test, _prosst_attention_mask = tokenize_sequences(s_test_blat, prosst_tokenizer)
    print("Getting ProSST test sequence embeddings...")
    x_prosst_emb_test = get_plm_embeddings(
        x_prosst_tok_test, prosst_base_model, wt_prosst_input_ids, prosst_attention_mask,  
        mode="mean", wt_structure_input_ids=wt_structure_input_ids, device=device, verbose=True
    )

    x_esm_tok_test, esm_attention_mask = tokenize_sequences(s_test_blat, esm_tokenizer)
    print("Getting ESM test sequence embeddings...")
    x_esm_emb_test = get_plm_embeddings(
        x_esm_tok_test, esm_base_model, wt_esm_input_ids, esm_attention_mask, 
        mode="mean", device=device, verbose=True
    )

    assert x_esm_emb_test.shape == (400, 1280), x_esm_emb_test.shape
    assert x_prosst_emb_test.shape == (400, 768), x_prosst_emb_test.shape
    x_combined_test = torch.cat([x_esm_emb_test, x_prosst_emb_test], dim=-1)  # Pay attention to correct order!
    x_combined_train = torch.cat([x_esm_emb_train, x_prosst_emb_train], dim=-1)
    assert x_combined_test.shape == x_combined_train.shape == (400, 2048), (
        x_combined_test.shape, x_combined_train.shape)
    print("Training models...\n-------------------\nESM...")
    esm_model = get_gp_kernel_model(
        y_train, x_tokseqs_seq_kernel_train=x_esm_emb_train, 
        device=device, train=True
    )
    print("ProSST...")
    # Using seq kernel 
    prosst_model_1 = get_gp_kernel_model(
        y_train, x_tokseqs_seq_kernel_train=x_prosst_emb_train, 
        device=device, train=True
    )
    # Using struct kernel
    prosst_model_2 = get_gp_kernel_model(
        y_train, x_tokseqs_struct_kernel_train=x_prosst_emb_train, 
        device=device, train=True
    )
    print("Combined...")
    comb_model = get_gp_kernel_model(
        y_train=y_train, x_tokseqs_seq_kernel_train=x_esm_emb_train, 
        x_tokseqs_struct_kernel_train=x_prosst_emb_train, 
        device=device, train=True
    )
    comb_model_2 = get_gp_kernel_model(
        y_train=y_train, x_tokseqs_seq_kernel_train=x_combined_train, 
        device=device, train=True
    )

    for i, (model, x_test) in enumerate(
        zip(
            [esm_model, prosst_model_1, prosst_model_2, comb_model, comb_model_2], 
            [x_esm_emb_test, x_prosst_emb_test, x_prosst_emb_test, x_combined_test, x_combined_test]
        )
    ):
        print("~~~ " + ["ESM", "ProSST_1", "ProSST_2", "ESM + ProSST combined seq. + struc. kernel", 
                        "ESM + Prosst combined seq. kernel only"][i] + " GP Test ~~~")
        likelihood = model.likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_test))
            y_pred = pred.mean

        spear_rho = spearmanr(y_test.cpu(), y_pred.cpu().numpy())[0]
        pear_r = pearsonr(y_test.cpu(), y_pred.cpu().numpy())[0]
        print("Spearman's rho SciPy TEST:                 ", spear_rho)
        print("Correlation loss Spearman TEST:            ", hybrid_corr_mse_loss(y_test, y_pred, method="spearman"))
        print("Pearson's r SciPy TEST:                    ", pear_r)
        print("Correlation loss Pearson TEST:             ", hybrid_corr_mse_loss(y_test, y_pred, method="pearson"))
        print("Correlation hybrid MSE-Spearman loss TEST: ", hybrid_corr_mse_loss(y_test, y_pred, method='spearman-hybrid'))
        print("Correlation hybrid MSE-Pearson TEST:       ", hybrid_corr_mse_loss(y_test, y_pred, method='pearson-hybrid'))
        print("MSE:                                       ", hybrid_corr_mse_loss(y_test, y_pred, alpha=0.0))
        if i == 2:
            try:
                np.testing.assert_almost_equal(spear_rho, 0.6321742635891474, decimal=2)
            except AssertionError:
                np.testing.assert_almost_equal(spear_rho, 0.6390131813323833, decimal=2)
        else:
            np.testing.assert_almost_equal(
                spear_rho, 
                [0.7021152007200044, 0.69065778648575, None, 0.7670016687604297, 0.7773472334202088][i], 
                decimal=3
            )


    #################################################
    ##################### KERMUT ####################
    #################################################
        
    aa_cond_probs = plm_inference(
        attention_mask=prosst_attention_mask,
        wt_input_ids=wt_prosst_input_ids,
        model=prosst_base_model,
        tokenized_sequences = None,
        extract_probs=True, 
        wt_structure_input_ids=wt_structure_input_ids,
        extract_conditional_aa_prob=True,
        tokenizer=prosst_tokenizer
    )

    struct_coords = extract_pdb_coords(
        pdb_blat_ecolx,
        target_len=len(wt_seq)
    )

    np.testing.assert_almost_equal(np.sum(struct_coords), -2125.667, decimal=2)

    # Run PROSST Inference
    x_prosst_tok_train = torch.tensor(x_prosst_tok_train)
    x_prosst_tok_test = torch.tensor(x_prosst_tok_test)
        
    x_zero_shot_train_prosst = plm_inference(
        x_prosst_tok_train, wt_prosst_input_ids, prosst_attention_mask, 
        prosst_base_model, wt_structure_input_ids=wt_structure_input_ids
    )
    x_zero_shot_train_esm = plm_inference(x_esm_tok_train, wt_esm_input_ids, esm_attention_mask, esm_base_model)
    x_zero_shot_test_prosst = plm_inference(
        x_prosst_tok_test, wt_prosst_input_ids, prosst_attention_mask, 
        prosst_base_model, wt_structure_input_ids=wt_structure_input_ids
    )
    x_zero_shot_test_esm = plm_inference(x_esm_tok_test, wt_esm_input_ids, esm_attention_mask, esm_base_model)

    print("Zero-shot ProSST Spearman Train:", spearmanr(y_train.cpu().numpy(), x_zero_shot_train_prosst.cpu().numpy()))
    print("Zero-shot ProSST Spearman Test:", spearmanr(y_test.cpu().numpy(), x_zero_shot_test_prosst.cpu().numpy()))


    # Only ProSST
    #############
    train_inputs = prepare_kermut_inputs(
        seqs=s_train_blat,
        x_embed=x_prosst_emb_train,
        x_zero_shot=x_zero_shot_train_prosst,
        device=device
    )

    # Option A: Using RBF (Default)
    gp, likelihood = instantiate_gp(
        train_inputs=train_inputs,
        train_targets=y_train,
        gp_inputs={
            "aa_cond_probs": aa_cond_probs, 
            "struct_coords": struct_coords, 
            "wt_seq": wt_seq
        },
        use_structure_kernel=True,
        use_sequence_kernel=True,
        sequence_kernel_type="RBF",
        use_zero_shot=True,
        device=device
    )

    # Train
    gp, likelihood = optimize_gp(
        gp, likelihood, train_inputs, y_train, lr=0.05, n_steps=150
    )

    # Predict
    test_inputs = prepare_kermut_inputs(
        seqs=s_test_blat,
        x_embed=x_prosst_emb_test,
        x_zero_shot=x_zero_shot_test_prosst,
        device=device
    )
    test_means_pred, _test_variances = predict(gp, likelihood, test_inputs)
    print("Supervised Kermut Spearman ProSST Train on Test:", spearmanr(y_test.cpu(), test_means_pred.cpu())[0])
    np.testing.assert_almost_equal(spearmanr(y_test.cpu(), test_means_pred.cpu())[0], 0.8460823505146906, decimal=3)

    # Only ESM
    ##########
    train_inputs = prepare_kermut_inputs(
        seqs=s_train_blat,
        x_embed=x_esm_emb_train,
        x_zero_shot=x_zero_shot_train_esm,
        device=device
    )

    # Option A: Using RBF (Default)
    gp, likelihood = instantiate_gp(
        train_inputs=train_inputs,
        train_targets=y_train,
        gp_inputs={
            "aa_cond_probs": aa_cond_probs, 
            "struct_coords": struct_coords, 
            "wt_seq": wt_seq
        },
        use_structure_kernel=True,
        use_sequence_kernel=True,
        sequence_kernel_type="RBF",
        use_zero_shot=True,
        device=device
    )

    # Train
    gp, likelihood = optimize_gp(
        gp, likelihood, train_inputs, y_train, lr=0.05, n_steps=150
        )

    # Predict
    test_inputs = prepare_kermut_inputs(
        seqs=s_test_blat,
        x_embed=x_esm_emb_test,
        x_zero_shot=x_zero_shot_test_esm,
        device=device
    )
    test_means_pred, _test_variances = predict(gp, likelihood, test_inputs)
    print("Supervised Kermut Spearman ESM Train on Test:", spearmanr(y_test.cpu(), test_means_pred.cpu())[0])

    # ESM(Seq.) + PROSST(Struct.)
    train_inputs = prepare_kermut_inputs(
        seqs=s_train_blat,
        x_embed=x_esm_emb_train,
        x_zero_shot=x_zero_shot_train_prosst,
        device=device
    )

    # Option A: Using RBF (Default)
    gp, likelihood = instantiate_gp(
        train_inputs=train_inputs,
        train_targets=y_train,
        gp_inputs={
            "aa_cond_probs": aa_cond_probs, 
            "struct_coords": struct_coords, 
            "wt_seq": wt_seq
        },
        use_structure_kernel=True,
        use_sequence_kernel=True,
        sequence_kernel_type="RBF",
        use_zero_shot=True,
        device=device
    )

    # Train
    gp, likelihood = optimize_gp(
        gp, likelihood, train_inputs, y_train, lr=0.05, n_steps=150
    )

    # Predict
    test_inputs = prepare_kermut_inputs(
        seqs=s_test_blat,
        x_embed=x_esm_emb_test,
        x_zero_shot=x_zero_shot_test_prosst,
        device=device
    )
    test_means_pred, _test_variances = predict(gp, likelihood, test_inputs)
    print("Supervised Kermut Spearman ESM-Seq. + ProSST-Struct. Train on Test:", spearmanr(y_test.cpu(), test_means_pred.cpu())[0])

    ######## Concat. Combined

    # Align token dimensions and create positional mapping
    x_combined_embeddings_train = torch.cat(
        (x_prosst_emb_train, x_esm_emb_train), 
        dim=-1
    )

    x_combined_embeddings_test = torch.cat(
        (x_prosst_emb_test, x_esm_emb_test), 
        dim=-1
    )

    # Safely expand 1D zero-shot vectors to 2D [400, 1] matrices before combining
    x_zs_train_prosst = x_zero_shot_train_prosst.unsqueeze(-1) if x_zero_shot_train_prosst.dim() == 1 else x_zero_shot_train_prosst
    x_zs_train_esm = x_zero_shot_train_esm.unsqueeze(-1) if x_zero_shot_train_esm.dim() == 1 else x_zero_shot_train_esm

    x_combined_zs_train = torch.cat(
        (x_zs_train_prosst, x_zs_train_esm), 
        dim=-1
    ) # Vertically aligned columns -> Shape: [400, 2]

    x_zs_test_prosst = x_zero_shot_test_prosst.unsqueeze(-1) if x_zero_shot_test_prosst.dim() == 1 else x_zero_shot_test_prosst
    x_zs_test_esm = x_zero_shot_test_esm.unsqueeze(-1) if x_zero_shot_test_esm.dim() == 1 else x_zero_shot_test_esm

    x_combined_zs_test = torch.cat(
        (x_zs_test_prosst, x_zs_test_esm), 
        dim=-1
    ) # Vertically aligned columns -> Shape: [test_samples, 2]

    train_inputs = prepare_kermut_inputs(
        seqs=s_train_blat,
        x_embed=x_combined_embeddings_train,
        x_zero_shot=x_combined_zs_train,
        device=device
    )

    # Option A: Using RBF (Default)
    gp, likelihood = instantiate_gp(
        train_inputs=train_inputs,
        train_targets=y_train,
        gp_inputs={
            "aa_cond_probs": aa_cond_probs, 
            "struct_coords": struct_coords, 
            "wt_seq": wt_seq
        },
        use_structure_kernel=True,
        use_sequence_kernel=True,
        sequence_kernel_type="RBF",
        use_zero_shot=True,
        device=device
    )

    # Train
    gp, likelihood = optimize_gp(
        gp, likelihood, train_inputs, y_train, lr=0.05, n_steps=150
    )

    # Predict
    test_inputs = prepare_kermut_inputs(
        seqs=s_test_blat,
        x_embed=x_combined_embeddings_test,
        x_zero_shot=x_combined_zs_test,
        device=device
    )
    test_means_pred, _test_variances = predict(gp, likelihood, test_inputs)

    print("Supervised Kermut Spearman ESM + ProSST Concat. Train on Test:", spearmanr(y_test.cpu(), test_means_pred.cpu())[0])
    np.testing.assert_almost_equal(spearmanr(y_test.cpu(), test_means_pred.cpu())[0], 0.8664669154182213, decimal=3)
    t2 = time.time()
    print(f"Function run time: {t2 - t1:.2f} s\n")

if __name__ == "__main__":
    test_gremlin_avgfp()
    test_hybrid_model_dca_llm_aneh()
    test_hybrid_model_dca_llm_avgfp()
    test_dataset_b_results()
    test_plm_corr_blat_ecolx()
    test_gaussian_process_opt()
    