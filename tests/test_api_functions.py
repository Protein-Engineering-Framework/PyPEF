
# Run me from parent dir:
#   Linux
#       export PYTHONPATH="${PYTHONPATH}:${PWD}" && python -m pytest ./tests/
#   Windows
#       $env:PYTHONPATH = "${PWD};${env:PYTHONPATH}";python -m pytest .\tests\


import os.path
import numpy as np
from scipy.stats import spearmanr

from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path, get_regressor_performances
from pypef.dca.gremlin_inference import GREMLIN
from pypef.utils.variant_data import get_sequences_from_file
from pypef.llm.esm_lora_tune import get_esm_models, esm_tokenize_sequences, corr_loss, get_batches, esm_train, esm_test, esm_infer
from pypef.hybrid.hybrid_model import DCALLMHybridModel


msa_file_avgfp = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
    )
)


msa_file_aneh = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/ANEH/ANEH_jhmmer.a2m'
    )
)

ls_b = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/ANEH/LS_B.fasl'
    )
)

ts_b = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/ANEH/TS_B.fasl'
    )
)


train_seqs, _train_vars, train_ys = get_sequences_from_file(ls_b)
test_seqs, _test_vars, test_ys = get_sequences_from_file(ts_b)


def test_gremlin():
    g = GREMLIN(
        alignment=msa_file_avgfp,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()  # only 1 decimal place for TensorFlow result
    np.testing.assert_almost_equal(wt_score, 952.1, decimal=1)
    y_pred = g.get_scores(np.append(train_seqs, test_seqs))
    np.testing.assert_almost_equal(
        spearmanr(np.append(train_ys, test_ys), y_pred)[0], 
        0.4516502675400598, 
        decimal=3
    )


def test_hybrid_model_dca_esm():
    g = GREMLIN(
        alignment=msa_file_aneh,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    x_dca_train = g.get_scores(train_seqs, encode=True)
    print(spearmanr(
        train_ys,
        np.sum(x_dca_train, axis=1)
    ), len(train_ys))

    print(len(train_seqs[0]), train_seqs[0])
    assert len(train_seqs[0]) == len(g.wt_seq)
    base_model, lora_model, tokenizer, optimizer = get_esm_models()
    x_train_esm, attention_mask_esm = esm_tokenize_sequences(
        list(train_seqs), tokenizer, max_length=len(train_seqs[0]))
    x_esm_b, train_ys_b = (
        get_batches(x_train_esm, dtype=int), 
        get_batches(train_ys, dtype=float)
    )
    y_true, y_pred_esm = esm_test(
        x_esm_b, attention_mask_esm, train_ys_b, 
        loss_fn=corr_loss, model=base_model
    )
    print(spearmanr(
        y_true,
        y_pred_esm
    ), len(y_true))

    llm_dict_esm = {
        'esm1v': {
            'llm_base_model': base_model,
            'llm_model': lora_model,
            'llm_optimizer': optimizer,
            'llm_train_function': esm_train,
            'llm_inference_function': esm_infer,
            'llm_loss_function': corr_loss,
            'x_llm' : x_train_esm,
            'llm_attention_mask':  attention_mask_esm
        }
    }

    hm = DCALLMHybridModel(
        x_train_dca=np.array(x_dca_train), 
        y_train=train_ys,
        llm_model_input=llm_dict_esm,
        x_wt=g.x_wt,
        seed=42
    )

    x_dca_test = g.get_scores(test_seqs, encode=True)
    encoded_seqs_test, attention_masks_test = esm_tokenize_sequences(
        list(test_seqs), tokenizer, max_length=len(test_seqs[0]))
    y_pred_test = hm.hybrid_prediction(x_dca=x_dca_test, x_llm=encoded_seqs_test)
    print(hm.beta1, hm.beta2, hm.beta3, hm.beta4, hm.ridge_opt)
    print('hm.y_dca_ttest', spearmanr(hm.y_ttest, hm.y_dca_ttest), len(hm.y_ttest))
    print('hm.y_dca_ridge_ttest', spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest), len(hm.y_ttest))
    print('hm.y_llm_ttest', spearmanr(hm.y_ttest, hm.y_llm_ttest), len(hm.y_ttest))
    print('hm.y_llm_lora_ttest', spearmanr(hm.y_ttest, hm.y_llm_lora_ttest), len(hm.y_ttest))
    print('Hybrid', spearmanr(test_ys, y_pred_test), len(test_ys))
    np.testing.assert_almost_equal(spearmanr(
        hm.y_ttest, hm.y_dca_ttest)[0], -0.5342743713116743, decimal=5)
    np.testing.assert_almost_equal(spearmanr(
        hm.y_ttest, hm.y_dca_ridge_ttest)[0], 0.717333573331078, decimal=5)
    np.testing.assert_almost_equal(spearmanr(
        hm.y_ttest, hm.y_llm_ttest)[0], -0.21761360470606333, decimal=5)
    # Nondeterministic behavior, should be about ~ 0.8, checking if not NaN
    # Torch reproducibility documentation: https://pytorch.org/docs/stable/notes/randomness.html
    assert -1.0 <= spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0] <= 1.0  
    assert -1.0 <= spearmanr(test_ys, y_pred_test)[0] <= 1.0  


def test_dataset_b_results():
    aaindex = "WOLR810101.txt"
    x_fft_train, _ = AAIndexEncoding(
        full_aaidx_txt_path(aaindex), train_seqs
    ).collect_encoded_sequences()
    x_fft_test, _ = AAIndexEncoding(
        full_aaidx_txt_path(aaindex), test_seqs
    ).collect_encoded_sequences()
    performances = get_regressor_performances(
        x_learn=x_fft_train, 
        x_test=x_fft_test, 
        y_learn=train_ys, 
        y_test=test_ys, 
        regressor='pls_loocv'
    )  
    # Dataset B PLS_LOOCV results: R², RMSE, NRMSE, Pearson's r, Spearman's rho 
    # RMSE, in Python 3.10 14.669 and from Python 3.11 on 14.17:
    np.testing.assert_almost_equal(performances[1], 14.48, decimal=0)
    # R²
    np.testing.assert_almost_equal(performances[0], 0.72, decimal=2)
    #  NRMSE, Pearson's r, Spearman's rho
    np.testing.assert_almost_equal(performances[2:5], [0.52, 0.86, 0.89], decimal=2)


if __name__ == "__main__":
    test_gremlin()
    test_hybrid_model_dca_esm()
    test_dataset_b_results()
    