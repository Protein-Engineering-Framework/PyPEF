
import os.path
import numpy as np

from pypef.ml.regression import AAIndexEncoding, full_aaidx_txt_path, get_regressor_performances
from pypef.dca.gremlin_inference import GREMLIN
from pypef.utils.variant_data import get_sequences_from_file



msa_file = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
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


def test_gremlin():
    g = GREMLIN(
        alignment=msa_file,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()  # only 1 decimal place for TensorFlow result
    np.testing.assert_almost_equal(wt_score, 1203.549234202937, decimal=1)
    

def test_dataset_b_results():
    train_seqs, train_vars, train_ys = get_sequences_from_file(ls_b)
    test_seqs, test_vars, test_ys = get_sequences_from_file(ts_b)
    aaindex = "WOLR810101.txt"
    x_fft_train, _ = AAIndexEncoding(full_aaidx_txt_path(aaindex), train_seqs).collect_encoded_sequences()
    x_fft_test, _ = AAIndexEncoding(full_aaidx_txt_path(aaindex), test_seqs).collect_encoded_sequences()
    performances = get_regressor_performances(
        x_learn=x_fft_train, 
        x_test=x_fft_test, 
        y_learn=train_ys, 
        y_test=test_ys, 
        regressor='pls_loocv'
    )  
    # Dataset B PLS_LOOCV results: R², RMSE, NRMSE, Pearson's r, Spearman's rho 
    np.testing.assert_almost_equal(performances[:5], [0.72, 14.48, 0.52, 0.86, 0.89], decimal=2)
