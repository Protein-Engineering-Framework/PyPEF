
# Run me from parent dir:
#   Linux
#       export PYTHONPATH="${PYTHONPATH}:${PWD}" && python -m pytest ./tests/
#   Windows
#       $env:PYTHONPATH = "${PWD};${env:PYTHONPATH}";python -m pytest .\tests\


import os.path
import numpy as np
from scipy.stats import spearmanr
import torch

from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path, get_regressor_performances
from pypef.dca.gremlin_inference import GREMLIN
from pypef.utils.variant_data import get_sequences_from_file, get_wt_sequence
from pypef.llm.esm_lora_tune import esm_setup
from pypef.llm.prosst_lora_tune import prosst_setup
from pypef.llm.inference import inference, llm_embedder
from pypef.hybrid.hybrid_model import DCALLMHybridModel


torch.manual_seed(42)
np.random.seed(42)

msa_file_avgfp = os.path.abspath(os.path.join(
    __file__, '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
))

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

train_seqs_aneh, _train_vars_aneh, train_ys_aneh = get_sequences_from_file(ls_b)
test_seqs_aneh, _test_vars_aneh, test_ys_aneh = get_sequences_from_file(ts_b)


def test_gremlin_avgfp():
    g = GREMLIN(
        alignment=msa_file_avgfp,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()  
    np.testing.assert_almost_equal(wt_score, 952.1102220697624, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)


def test_hybrid_model_dca_llm():
    g = GREMLIN(
        alignment=msa_file_aneh,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()
    np.testing.assert_almost_equal(wt_score, 1743.2087199198131, decimal=1)
    assert wt_score == g.wt_score == np.sum(g.x_wt)
    y_pred = g.get_scores(np.append(train_seqs_aneh, test_seqs_aneh))
    np.testing.assert_almost_equal(
        spearmanr(np.append(train_ys_aneh, test_ys_aneh), y_pred)[0], 
        -0.5528510930046211, 
        decimal=7
    )
    x_dca_train = g.get_scores(train_seqs_aneh, encode=True)
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, np.sum(x_dca_train, axis=1))[0],
        -0.5556053466180598,
        decimal=7
    )
    assert len(train_seqs_aneh[0]) == len(g.wt_seq)

    y_pred_esm = inference(train_seqs_aneh, 'esm')
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, y_pred_esm)[0], 
        -0.21073416060442696, 
        decimal=7
    )
    aneh_wt_seq = get_wt_sequence(wt_seq_file_aneh)
    y_pred_prosst = inference(
        train_seqs_aneh, 'prosst', 
        pdb_file=pdb_file_aneh, wt_seq=aneh_wt_seq
    )
    np.testing.assert_almost_equal(
        spearmanr(train_ys_aneh, y_pred_prosst)[0], 
        -0.7425657069861902, 
        decimal=7
    )

    x_dca_test = g.get_scores(test_seqs_aneh, encode=True)
    for i, setup in enumerate([esm_setup, prosst_setup]):
        print(['~~~ ESM ~~~', '~~~ ProSST ~~~'][i])
        if setup == esm_setup:
            llm_dict = setup(sequences=train_seqs_aneh)
        else:  # elif setup == prosst_setup:
            llm_dict = setup(
                aneh_wt_seq, pdb_file_aneh, sequences=train_seqs_aneh)
        x_llm_test = llm_embedder(llm_dict, test_seqs_aneh)
        hm = DCALLMHybridModel(
            x_train_dca=np.array(x_dca_train), 
            y_train=train_ys_aneh,
            llm_model_input=llm_dict,
            x_wt=g.x_wt,
            seed=42
        )

        y_pred_test = hm.hybrid_prediction(x_dca=x_dca_test, x_llm=x_llm_test)
        print(hm.beta1, hm.beta2, hm.beta3, hm.beta4, hm.ridge_opt)
        print('hm.y_dca_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ttest), len(hm.y_ttest))
        print('hm.y_dca_ridge_ttest:', spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest), len(hm.y_ttest))
        print('hm.y_llm_ttest:', spearmanr(hm.y_ttest, hm.y_llm_ttest), len(hm.y_ttest))
        print('hm.y_llm_lora_ttest:', spearmanr(hm.y_ttest, hm.y_llm_lora_ttest), len(hm.y_ttest))
        print('Hybrid prediction:', spearmanr(test_ys_aneh, y_pred_test), len(test_ys_aneh))
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ttest)[0], -0.5342743713116743, 
            decimal=7
        )
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_dca_ridge_ttest)[0], 0.717333573331078, 
            decimal=7
        )
        np.testing.assert_almost_equal(
            spearmanr(hm.y_ttest, hm.y_llm_ttest)[0], 
            [-0.21761360470606333, -0.8330644449247571][i],
            decimal=7
        )  
        # Nondeterministic behavior (without setting seed), should be about ~0.7 to ~0.9, 
        # but as sample size is so low the following is only checking if not NaN / >=-1.0 and <=1.0,
        # Torch reproducibility documentation: https://pytorch.org/docs/stable/notes/randomness.html
        assert -1.0 <= spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0] <= 1.0  
        assert -1.0 <= spearmanr(test_ys_aneh, y_pred_test)[0] <= 1.0
        # With seed 42 for numpy and torch for implemented LLM's and on local machine:
        if setup == esm_setup:
            continue  # TODO: Make new/overloaded pytest decorator function
            #try: # Different values on different machines (TODO) has to be investigated
            #    np.testing.assert_almost_equal(
            #        spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7772102863835341, decimal=7
            #    )
            #except AssertionError as ae1:
            #    try:
            #        np.testing.assert_almost_equal(
            #            spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7239938685054149, decimal=7
            #        )
            #    except AssertionError as ae2:
            #        raise AssertionError(
            #            f"Neither condition passed:\nFirst comparison failed:\n{ae1}\n"
            #            f"Second comparison failed:\n{ae2}"
            #        )
            #try:
            #    np.testing.assert_almost_equal(
            #        spearmanr(test_ys_aneh, y_pred_test)[0], 0.8004896406836318, decimal=7
            #    )
            #except AssertionError as ae1:
            #    try:
            #        np.testing.assert_almost_equal(
            #            spearmanr(test_ys_aneh, y_pred_test)[0], 0.8338711936729409, decimal=7
            #        )
            #    except AssertionError as ae2:
            #        raise AssertionError(
            #            f"Neither condition passed:\nFirst comparison failed:\n{ae1}\n"
            #            f"Second comparison failed:\n{ae2}"
            #        )

        elif setup == prosst_setup:
            continue
            #try:
            #    np.testing.assert_almost_equal(
            #        spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7770124558338013, decimal=7
            #    )
            #except AssertionError as ae1:
            #    try: 
            #        np.testing.assert_almost_equal(  
            #        spearmanr(hm.y_ttest, hm.y_llm_lora_ttest)[0], 0.7239938685054149, decimal=7
            #        )                                
            #    except AssertionError as ae2:
            #        raise AssertionError(
            #            f"Neither condition passed:\nFirst comparison failed:\n{ae1}\n"
            #            f"Second comparison failed:\n{ae2}"
            #        )
            #np.testing.assert_almost_equal(
            #    spearmanr(test_ys_aneh, y_pred_test)[0], 0.8291977762544377, decimal=7
            #)


def test_dataset_b_results():
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


if __name__ == "__main__":
    test_gremlin_avgfp()
    test_hybrid_model_dca_llm()
    test_dataset_b_results()
    