"""
An example script for using PyPEF as an API for encoding sequences
to train and test ML models and the hybrid model MERGE.

Encoding of sequences is not parallelized in this script.
"""

import os
import pandas as pd
import numpy as np
from copy import copy
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
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


# 1st example: A. DCA-based encoding
# -------------------------------------------------------------------------------
# 1st example, since this approach reduces the amount of data (variant-fitness pairs),
# as not all variants can necessarily be encoded using the DCA approach. To compare
# the test performance with the other encoding techniques to be tested, we start with
# DCA and split the variant fitness data so that sizes of the data sets are same for
# all encoding techniques tested.
print('\nRunning script... which takes ~ 4h in total...\n\nTesting DCA-based sequence encoding...')
try:
    dca_encoder = DCAEncoding(
        params_file='./test_dataset_avgfp/uref100_avgfp_jhmmer_119_plmc_42.6.params',
        verbose=False
    )
except (ValueError, FileNotFoundError):
    if not os.path.isdir('test_dataset_avgfp'):
        os.mkdir('test_dataset_avgfp')
    print('Did not find required files for DCA-based encoding. Downloading required files...')
    import urllib.request
    # Single substituted encoded variants: CSV files including variant name and true fitness
    url = 'https://github.com/Protein-Engineering-Framework/PyPEF/raw/master/workflow/test_dataset_avgfp/avGFP.csv'
    urllib.request.urlretrieve(url, './test_dataset_avgfp/avGFP.csv')
    # Getting plmc parameter file
    url = 'https://github.com/niklases/PyPEF/raw/main/workflow/test_dataset_avgfp/uref100_avgfp_jhmmer_119_plmc_42.6.params'
    urllib.request.urlretrieve(url, './test_dataset_avgfp/uref100_avgfp_jhmmer_119_plmc_42.6.params')  # File size: 71.2 MB
    print('Successfully downloaded all required files!')
    dca_encoder = DCAEncoding(
        params_file='./test_dataset_avgfp/uref100_avgfp_jhmmer_119_plmc_42.6.params',
        verbose=False
    )

variant_fitness_data = pd.read_csv('./test_dataset_avgfp/avGFP.csv', sep=';')  # loading the avGFP dataset

variants = variant_fitness_data.iloc[:2000, 0]  # "just" using 2000 variants for faster processing
fitnesses = variant_fitness_data.iloc[:2000, 1].tolist()

# Splitting in sets for training (fitting and hyperparameter validation) and testing
# Change number of applied splits for training and testing here, default: n_splits = 5
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
train_val_splits_indices, test_splits_indices = [], []


x_dca_ = dca_encoder.collect_encoded_sequences(variants)
# removing not DCA-encodable positions (and also fitnesses, variants, and sequences)
x_dca, fitnesses = remove_nan_encoded_positions(copy(x_dca_), fitnesses)
_, variants = remove_nan_encoded_positions(copy(x_dca_), variants)

variants_split = []
for variant in variants:
    variants_split.append(variant.split('/'))
variants, fitnesses, sequences = get_seqs_from_var_name(wt_sequence, variants_split, fitnesses)
print(f'N Variants remaining after excluding non-DCA-encodable positions: {len(x_dca)}')
assert len(x_dca) == len(fitnesses) == len(variants) == len(sequences)

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
    performances = get_regressor_performances(x_train_val, x_test, y_train_val, y_test, regressor='ridge')
    # performances[4] is Spearmans correlation between y_true and y_pred
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
print('-'*80 + f'\n{n_splits}-fold mean Spearmans rho (ML)= {np.mean(ten_split_performance_ml):.3f} '
      f'+- {np.std(ten_split_performance_ml, ddof=1):.3f}\n'
      f'{n_splits}-fold mean Spearmans rho (Hybrid)= {np.mean(ten_split_performance_hybrid):.3f} '
      f'+- {np.std(ten_split_performance_ml, ddof=1):.3f}')


# 2nd example: AAindex encoding over all 566 amino acid descriptor sets
# -------------------------------------------------------------------------------
print('\n\nTesting AAindex-based sequence encoding...')
spearmans_rhos_aaidx, aa_index = [], []
# e.g., looping over the 566 AAindex entries, encode with each AAindex and test performance
# which can be seen as a AAindex hyperparameter search on the test set, i.e.,
# not totally fair comparison to onehot- and DCA-based encoding techniques.
aa_indices = [file for file in os.listdir(path_aaindex_dir()) if file.endswith('.txt')]
mean_perfromances, ten_split_performance_std_dev, aa_indices_collected = [], [], []
for index, aaindex in enumerate(aa_indices):
    aaidx_encoder = AAIndexEncoding(full_aaidx_txt_path(aaindex), sequences)
    # two encoding options possible, FFT-ed and 'raw' encoded sequences (here we keep using 'raw' encoded sequences)
    x_aaidx_fft, x_aaidx_no_fft = aaidx_encoder.collect_encoded_sequences()
    ten_split_performance = []
    if x_aaidx_no_fft == 'skip':
        print(f'Skipped AAindex {aaindex}')
        continue
    for i, indices in enumerate(train_val_splits_indices):
        x_train_val = np.array(x_aaidx_no_fft)[indices]
        y_train_val = np.array(fitnesses)[indices]
        x_test = np.array(x_aaidx_no_fft)[test_splits_indices[i]]
        y_test = np.array(fitnesses)[test_splits_indices[i]]
        performances = get_regressor_performances(x_train_val, x_test, y_train_val, y_test, regressor='ridge')
        ten_split_performance.append(performances[4])
    print(f'{index + 1}/{len(aa_indices)}: AAindex {aaindex}, '
          f'{n_splits}-fold mean Spearmans rho = {np.mean(ten_split_performance):.3f} '
          f'+- {np.std(ten_split_performance, ddof=1):.3f}')
    mean_perfromances.append(np.mean(ten_split_performance))

    ten_split_performance_std_dev.append(np.std(ten_split_performance, ddof=1))
    aa_indices_collected.append(aaindex)
max_value = max(mean_perfromances)
max_idx = mean_perfromances.index(max_value)
max_value_std = ten_split_performance_std_dev[max_idx]
max_value_aaidx = aa_indices_collected[max_idx]
print('-'*80 + f'\nBest {n_splits}-fold mean Spearmans rho = {max_value:.3f} '
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
    performances = get_regressor_performances(x_train_val, x_test, y_train_val, y_test, regressor='ridge')
    ten_split_performance.append(performances[4])
    print(f'Split {i + 1}/{len(train_val_splits_indices)}: Spearmans rho = {performances[4]:.3f}')
print('-'*80 + f'\n{n_splits}-fold mean Spearmans rho = {np.mean(ten_split_performance):.3f} '
      f'+- {np.std(ten_split_performance, ddof=1):.3f}\n')


# 4th example: Low-N and plotting using all encoding techniques and all data
# -------------------------------------------------------------------------------
print('Lastly, encoding all variants and performing "low-N" protein engineering task.\n'
      'This could require some time... ~ 2 hours left...\n'
      'Encoding sequences...')

variants = variant_fitness_data.iloc[:, 0]
fitnesses = variant_fitness_data.iloc[:, 1].tolist()

x_dca_ = dca_encoder.collect_encoded_sequences(variants)
# removing not DCA-encodable positions (and also reduce fitnesses, variants, and sequences accordingly)
x_dca, fitnesses = remove_nan_encoded_positions(copy(x_dca_), fitnesses)
_, variants = remove_nan_encoded_positions(copy(x_dca_), variants)

variants_split = []
for variant in variants:
    variants_split.append(variant.split('/'))

variants, fitnesses, sequences = get_seqs_from_var_name(wt_sequence, variants_split, fitnesses)
print(f'N Variants remaining after excluding non-DCA-encodable positions = {len(x_dca)}')

# using the best identified index, i.e., QIAN880130
print(f'AAIndex-based encoding of the {len(x_dca)} variants (using index QIAN880130)...')
x_aaindex_fft, x_aaindex_no_fft = AAIndexEncoding(
    full_aaidx_txt_path('QIAN880130.txt'), sequences).collect_encoded_sequences()
print(f'One-hot encoding of the {len(x_dca)} variants...')
x_onehot = OneHotEncoding(sequences).collect_encoded_sequences()

assert len(x_dca) == len(x_aaindex_no_fft) == len(x_onehot) == len(fitnesses) == len(variants) == len(sequences)

all_mean_performances_dca, all_mean_performances_hybrid, \
    all_mean_performances_aaidx, all_mean_performances_onehot = [], [], [], []
all_stddevs_dca, all_stddevs_hybrid, all_stddevs_aaidx, all_stddevs_onehot = [], [], [], []
low_n_train = np.arange(50, 1001, 50)
print(f'Testing low N performance, with N_train = {list(low_n_train)}...')
for n_train in tqdm(low_n_train):
    performances_dca, performances_hybrid, performances_aaidx, performances_onehot = [], [], [], []
    for rnd_state in [42, 213, 573, 917, 823]:
        x_dca_train, x_dca_test, y_train, y_test = train_test_split(
            x_dca, fitnesses, train_size=n_train, random_state=rnd_state)
        performances_dca.append(get_regressor_performances(
            x_dca_train, x_dca_test, y_train, y_test, regressor='ridge')[4])  # [4] defines spearmanr

        x_wt = dca_encoder.collect_encoded_sequences(['A110A'])
        hybrid_model = DCAHybridModel(X_train=x_dca_train, y_train=y_train, X_wt=x_wt)
        beta_1, beta_2, hybrid_regressor = hybrid_model.settings(X_train=x_dca_train, y_train=y_train)
        y_hybrid_pred = hybrid_model.hybrid_prediction(X=x_dca_test, reg=hybrid_regressor, beta_1=beta_1, beta_2=beta_2)
        performances_hybrid.append(spearmanr(y_test, y_hybrid_pred)[0])

        x_aaidx_train, x_aaidx_test, y_train, y_test = train_test_split(
            x_aaindex_no_fft, fitnesses, train_size=n_train, random_state=rnd_state)
        performances_aaidx.append(get_regressor_performances(
            x_aaidx_train, x_aaidx_test, y_train, y_test, regressor='ridge')[4])


        x_onehot_train, x_onehot_test, y_train, y_test = train_test_split(
            x_onehot, fitnesses, train_size=n_train, random_state=rnd_state)
        performances_onehot.append(get_regressor_performances(
            x_onehot_train, x_onehot_test, y_train, y_test, regressor='ridge')[4])

    all_mean_performances_dca.append(np.mean(performances_dca))
    all_stddevs_dca.append(np.std(performances_dca, ddof=1))
    all_mean_performances_hybrid.append(np.mean(performances_hybrid))
    all_stddevs_hybrid.append(np.std(performances_hybrid, ddof=1))
    all_mean_performances_aaidx.append(np.mean(performances_aaidx))
    all_stddevs_aaidx.append(np.std(performances_aaidx, ddof=1))
    all_mean_performances_onehot.append(np.mean(performances_onehot))
    all_stddevs_onehot.append(np.std(performances_onehot, ddof=1))

# Plotting all the achieved "low-N" performances
plt.plot(low_n_train, all_mean_performances_dca, 'o--', color='tab:orange', label='DCA encoding (pure ML)')
plt.fill_between(low_n_train,
                 np.array(all_mean_performances_dca) - np.array(all_stddevs_dca),
                 np.array(all_mean_performances_dca) + np.array(all_stddevs_dca),
                 color='tab:orange', alpha=0.2)
plt.plot(low_n_train, all_mean_performances_hybrid, 'o--', color='tab:blue', label='Hybrid DCA model (MERGE)')
plt.fill_between(low_n_train,
                 np.array(all_mean_performances_hybrid) - np.array(all_stddevs_hybrid),
                 np.array(all_mean_performances_hybrid) + np.array(all_stddevs_hybrid),
                 color='tab:blue', alpha=0.2)
plt.plot(low_n_train, all_mean_performances_aaidx, 'o--', color='tab:green', label='AAindex (QIAN880130)')
plt.fill_between(low_n_train,
                 np.array(all_mean_performances_aaidx) - np.array(all_stddevs_aaidx),
                 np.array(all_mean_performances_aaidx) + np.array(all_stddevs_aaidx),
                 color='tab:green', alpha=0.2)
plt.plot(low_n_train, all_mean_performances_onehot, 'o--', color='tab:grey', label='One-hot')
plt.fill_between(low_n_train,
                 np.array(all_mean_performances_onehot) - np.array(all_stddevs_onehot),
                 np.array(all_mean_performances_onehot) + np.array(all_stddevs_onehot),
                 color='tab:grey', alpha=0.2)
plt.legend()
plt.xlabel(r'$N_\mathrm{Train}$')
plt.ylabel(r'$\rho$')
plt.savefig('low_N_avGFP_extrapolation.png', dpi=500)
print('\nDone!')
