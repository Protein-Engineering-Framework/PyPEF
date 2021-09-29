#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – an Integrated Framework for Data-driven Protein Engineering
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

import pickle
from exemplary_dataset_B import all_sequences, all_labels
from pypef.encoding_api import AAIndexEncoding, get_performance
from pypef.encoding_api import pls_cv_regressor, svr_cv_regressor, rf_cv_regressor, mlp_cv_regressor
# Or just use a sklearn BaseEstimator, e.g. PLSRegression():
from sklearn.cross_decomposition import PLSRegression  # Needed for model=eval(best_model[-2]), import also for other
                                                       # regression options if required.

# Split all data (sequences and corresponding fitness labels), e.g. in 120 entries
# for learning and 30 entries for validation (20% validation set size).
learn_sequences, y_learn = all_sequences[:120], all_labels[:120]
valid_sequences, y_valid = all_sequences[120:], all_labels[120:]

performances, _cv_performances = get_performance(                            # _cv_performances not used herein.
    learn_sequences, y_learn, valid_sequences, y_valid,                      # Next to GridSearchCV-based Regression,
    fft=True, tqdm_=True, kfold_cv_on_all=0, regressor=pls_cv_regressor(),   # you can use just a sklearn BaseEstimator,
    finally_train_on_all=False, save_models=5, sort=1                        # e.g., PLSRegression().
)   # If kfold_cv_on_all is not set to 0/False, the given int determines the k in k-fold CV on all data: Output will be
# the _cv_performances list (quite dependent performances on data distribution inside splits) – increases run time.
# Besides, save_models saves int best models according to performance parameters (defined by sort=int: {1: R^2, 2: RMSE,
# 3: NRMSE, 4: Pearson's r, 5: Spearman's rank} [default: 1]). tqdm_ True/False switches progress bar on/off.

# New or any variant sequence (or list of many sequences) to predict
seq_to_predict = [
    'MSAPFAKFPSSASISPNPFTVSIPDEQLDDLKTLVRLSKIAPPTYESLQADGRFGITSEWLTTMREKWLSEFDWRPFEARLNSFPQFTTEIEGLTIHFAALFSEREDAVPIALL'
    'HGWPGSFVEFYPILQLFREEYTPETLPFHLVVPSLPGYTFSSGPPLDKDFGLMDNARVVDQLMKDLGFGSGYIIQGGDIGSFVGRLLGVGFDACKAVHLNFCAMDAPPEGPSIE'
    'SLSAAEKEGIARMEKVMTDGIAYAMEHSTRPSTIGHVLSSSPIALLAWIGEKYLQWVDKPLPSETILEMVSLYWLTESFPRAIHTYRECFPTASAPNGATMLQKELYIHKPFGF'
    'SFFPKDVHPVPRSWIATTGNLVFFRDHAEGGHFAALERPRELKTDLTAFVEQVWQK'
]

# Printing top 10 models: used indices for aa encoding, achieved performance values, and model parameters.
print('Top ten models (ranked by R2):')
for i, p in enumerate(performances[:10]):
    print(i+1, p[:7])

# The best model regarding R2 (if get_performance(sort='1')) is performances[0].
best_model = performances[0]
print('\nBest model according to R2 on validation set: {}\n'.format(best_model[:7]))

# The best model was encoded by amino acid index stored in the eighths entry of the list: best_model[8].
use_aaindex = best_model[8]


######## EXAMPLE 1: Reconstruct best model from performance list ########

# Reconstruct model parameters in list entry best_model[7].
model = eval(best_model[7])

# Get fft-ed or raw-encoded sequences for fitting and prediction, use raw_encoded_num_seq_to_learn if no fft was used.
fft_encoded_num_seq_to_learn, raw_encoded_num_seq_to_learn = AAIndexEncoding(use_aaindex, learn_sequences).get_x_and_y()
fft_encoded_num_seq_to_pred, raw_encoded_num_seq_to_pred = AAIndexEncoding(use_aaindex, seq_to_predict).get_x_and_y()
fft_encoded_num_seq_to_valid, raw_encoded_num_seq_to_valid = AAIndexEncoding(use_aaindex, valid_sequences).get_x_and_y()

# Refitting model reconstructed from string in performance list on learning data.
model.fit(fft_encoded_num_seq_to_learn, y_learn)

y_pred = model.predict(fft_encoded_num_seq_to_valid)
print('Observed vs. [Predicted] entries of the validation set:')
for i, x in enumerate(y_pred):
    print(y_valid[i], y_pred[i])

# Predict (list of unknown/any) sequence(s) to estimate the fitness.
print('\nPredicted fitness of sequence with reconstructed model: {}\n'.format(model.predict(fft_encoded_num_seq_to_pred)))


######## EXAMPLE 2: Reload saved top model stored in folder Models/ (if not get_performance(save_model=0) ########

# Get name of AAindex from best model of performance list.
aaindex_name = best_model[0]

# Load model stored in folder /Models.
model = pickle.load(open('Models/' + aaindex_name + '.sav', 'rb'))

# Get fft-ed (and raw for get_performance(fft=False)) encoded sequence to predict.
fft_encoded_num_seq_to_pred, raw_encoded_num_seq_to_pred = AAIndexEncoding(use_aaindex, seq_to_predict).get_x_and_y()

# Predict (list of) (unknown) sequence(s) to estimate the fitness.
print('Predicted fitness of sequence with loaded model (should be the same result, '
      'except when finally trained model on all data): {}\n'.format(model.predict(fft_encoded_num_seq_to_pred)))

print('Done!')
