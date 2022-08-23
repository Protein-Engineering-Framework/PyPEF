"""
An exemplary script for using PyPEF as an API for encoding sequences to
train and test ML models and the hybrid model.
"""
import os
import pandas as pd
import numpy as np
from copy import copy
from scipy.stats import spearmanr

from sklearn.model_selection import KFold
from pypef.utils.learning_test_sets import get_seqs_from_var_name
from pypef.utils.variant_data import remove_nan_encoded_positions
from pypef.ml.regression import (
    OneHotEncoding, AAIndexEncoding, DCAEncoding, get_regressor_performances,
    path_aaindex_dir, full_aaidx_txt_path
)
from pypef.dca.hybrid_model import DCAHybridModel

# avGFP wild type sequence
wt_sequence = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL' \
              'VTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV' \
              'NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD' \
              'HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
variant_fitness_data = pd.read_csv('avGFP.csv', sep=';')  # taking the avGFP dataset
variants = variant_fitness_data.iloc[:2000, 0]  # "just" using 2000 variants for faster processing
fitnesses = variant_fitness_data.iloc[:2000, 1]

variants_split = []
for variant in variants:
    variants_split.append(variant.split('/'))

variants, fitnesses, sequences = get_seqs_from_var_name(wt_sequence, variants_split, fitnesses)

# Splitting in sets for training+validation and testing
kf = KFold(n_splits=10, random_state=42, shuffle=True)
train_val_splits_indices, test_splits_indices = [], []


# 1st example: A. DCA-based encoding
# -------------------------------------------------------------------------------
# 1st example, since this approach reduces the amount of data for variant fitness,
# as not all variants can be encoded using the DCA approach). To compare the test
# performance with the other encoding techniques to be tested, we start with DCA
# and split the variant fitness data so that sizes of the data sets are same for
# all encoding techniques tested.
print('\nTesting DCA-based sequence encoding...')
dca_encoder = DCAEncoding(
    params_file='./workflow/test_dataset_avgfp/uref100_avgfp_jhmmer_119_plmc_42.6.params',
    verbose=False
)
x_dca_ = dca_encoder.collect_encoded_sequences(variants)
x_dca, fitnesses = remove_nan_encoded_positions(copy(x_dca_), fitnesses)
x_dca, variants = remove_nan_encoded_positions(copy(x_dca_), variants)
print(f'N Variants remaining after excluding non-DCA-encodable positions = {len(x_dca)}')
assert len(x_dca) == len(fitnesses) == len(variants)

for train_val_indices, test_indices in kf.split(variants):  # several variants are not included in the data anymore
    train_val_splits_indices.append(train_val_indices)
    test_splits_indices.append(test_indices)
print(f'Total number of variant-fitness data for training and '
      f'validation-based hyperparameter tuning: {len(train_val_indices)}'  # indices only vary by +- 1 for the splits
      f'\nVariants for testing: {len(test_indices)}\n')

ten_split_performance_ml, ten_split_performance_hybrid = [], []
for i, indices in enumerate(train_val_splits_indices):
    x_train_val = np.array(x_dca)[indices]
    y_train_val = np.array(fitnesses)[indices]
    x_test = np.array(x_dca)[test_splits_indices[i]]
    y_test = np.array(fitnesses)[test_splits_indices[i]]
    # get_regressor_performances() already splits train_val data in data for fitting (training) and validation
    # and after each CV-round shifts to the next hyperparameter of the regressor hyperparameter grid
    performances = get_regressor_performances(x_train_val, x_test, y_train_val, y_test, regressor='pls')
    ten_split_performance_ml.append(performances[4])
    print(f'Split {i + 1}/{len(train_val_splits_indices)}:\nSpearmans rho (ML) = {performances[4]:.3f}')
    # B. Hybrid modeling
    # -------------------------------------------------------------------------------
    # WT defined by substitution to itself at an encodable position
    x_wt = dca_encoder.collect_encoded_sequences(['A110A'])
    hybrid_model = DCAHybridModel(X_train=x_train_val, y_train=y_train_val, X_wt=x_wt)
    beta_1, beta_2, regressor = hybrid_model.settings(X_train=x_train_val, y_train=y_train_val)
    y_test_pred = hybrid_model.hybrid_prediction(X=x_test, reg=regressor, beta_1=beta_1, beta_2=beta_2)
    ten_split_performance_hybrid.append(spearmanr(y_test, y_test_pred)[0])
    print(f'Spearmans rho (Hybrid) = {spearmanr(y_test, y_test_pred)[0]:.3f}')
print('-'*60 + f'\n10-fold mean Spearmans rho (ML)= {np.mean(ten_split_performance_ml):.3f} '
      f'+- {np.std(ten_split_performance_ml, ddof=1):.3f}\n'
      f'10-fold mean Spearmans rho (Hybrid)= {np.mean(ten_split_performance_hybrid):.3f} '
      f'+- {np.std(ten_split_performance_ml, ddof=1):.3f}')


# 2nd example: AAindex encoding over all 566 amino acid descriptor sets
# -------------------------------------------------------------------------------
print('\n\nTesting AAindex-based sequence encoding...')
spearmans_rhos_aaidx, aa_index = [], []
# e.g., looping over the 566 AAindex entries, encode with each AAindex and test performance
# which can be seen as a AAindex hyperparameter search on the test set
aa_indices = [file for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]
mean_perfromances, ten_split_performance_std_dev, aa_indices_collected = [], [], []
for index, aaindex in enumerate(aa_indices):
    aaidx_encoder = AAIndexEncoding(full_aaidx_txt_path(aaindex), sequences)
    # two encoding options possible, FFT-ed and 'raw' encoded sequences (here we keep using 'raw' encoded sequences)
    x_aaidx_fft, x_aaidx_no_fft = aaidx_encoder.collect_encoded_sequences()
    ten_split_performance = []
    if x_aaidx_no_fft == 'skip':
        continue
    for i, indices in enumerate(train_val_splits_indices):
        x_train_val = np.array(x_aaidx_no_fft)[indices]
        y_train_val = np.array(fitnesses)[indices]
        x_test = np.array(x_aaidx_no_fft)[test_splits_indices[i]]
        y_test = np.array(fitnesses)[test_splits_indices[i]]
        performances = get_regressor_performances(x_train_val, x_test, y_train_val, y_test, regressor='pls')
        ten_split_performance.append(performances[4])
    print(f'{index + 1}/{len(aa_indices)}: AAindex {aaindex}, '
          f'10-fold mean Spearmans rho = {np.mean(ten_split_performance):.3f} '
          f'+- {np.std(ten_split_performance, ddof=1):.3f}')
    mean_perfromances.append(np.mean(ten_split_performance))

    ten_split_performance_std_dev.append(np.std(ten_split_performance, ddof=1))
    aa_indices_collected.append(aaindex)
max_value = max(mean_perfromances)
max_idx = mean_perfromances.index(max_value)
max_value_std = ten_split_performance_std_dev[max_idx]
max_value_aaidx = aa_indices_collected[max_idx]
print('-'*60 + f'\nBest 10-fold mean Spearmans rho = {max_value:.3f} '
      f'+- {max_value_std:.3f}, AAindex descriptor set = {max_value_aaidx}\n')


# 3rd example: OneHot encoding
# -------------------------------------------------------------------------------
print('\nTesting OneHot sequence encoding...')
onehot_encoder = OneHotEncoding(sequences)
x_onehot = onehot_encoder.collect_encoded_sequences()
ten_split_performance = []
for i, indices in enumerate(train_val_splits_indices):
    x_train_val = np.array(x_onehot)[indices]
    y_train_val = np.array(fitnesses)[indices]
    x_test = np.array(x_onehot)[test_splits_indices[i]]
    y_test = np.array(fitnesses)[test_splits_indices[i]]
    performances = get_regressor_performances(x_train_val, x_test, y_train_val, y_test, regressor='pls')
    ten_split_performance.append(performances[4])
    print(f'Split {i + 1}/{len(train_val_splits_indices)}: Spearmans rho = {performances[4]:.3f}')
print('-'*60 + f'\n10-fold mean Spearmans rho = {np.mean(ten_split_performance):.3f} '
      f'+- {np.std(ten_split_performance, ddof=1):.3f}\n')
