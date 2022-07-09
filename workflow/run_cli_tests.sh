#!/bin/bash

### Bash script for testing some PyPEF CLI commands 
### based on the two datasets provided (ANEH and avGFP)

set -x  # echo on
set -e  # exit on (PyPEF) errors

### RUN ME WITH
### $ ./run_cli_tests.sh                      # printing STDOUT and STDERR to terminal
### $ ./run_cli_tests.sh &> test_run_log.log  # writing STDOUT and STDERR to log file

### if using downloaded/locally stored pypef .py files:
############### CHANGE THIS PATHS AND USED THREADS, REQUIRES PYTHON ENVIRONMENT WITH PRE-INSTALLED MODULES ###############
export PYTHONPATH=${PYTHONPATH}:/path/to/pypef-main                                                                      #
pypef='python3 /path/to/pypef-main/pypef/main.py'                                                                        #
threads=16                                                                                                               #
##########################################################################################################################
### else just use pip-installed pypef version (uncomment):                                                               #
#pypef=pypef                                                                                                             # 
##########################################################################################################################


### threads=1 shows progress bar where possible
### CV-based mlp and rf regression option optimization take a long time and related testing commands are commented out/not included herein


### Pure ML (and some hybrid model) tests on ANEH dataset
cd 'test_dataset_aneh'

$pypef --version
$pypef -h
$pypef mklsts -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta

$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor pls
$pypef ml --show
$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor pls_loocv
$pypef ml --show
$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor ridge
$pypef ml --show
$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor lasso
$pypef ml --show
$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor elasticnet
$pypef ml --show
#$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor mlp
#$pypef ml --show
#$pypef ml -e onehot -l LS.fasta -t TS.fasta --regressor rf
#$pypef ml --show

$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor pls --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor pls_loocv --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor ridge --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor lasso --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor elasticnet --threads $threads
$pypef ml --show

$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor pls --nofft --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor pls_loocv --nofft --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor ridge --nofft --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor lasso --nofft --threads $threads
$pypef ml --show
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor elasticnet --nofft --threads $threads
$pypef ml --show

$pypef ml -e dca -l LS.fasta -t TS.fasta --regressor pls --params ANEH_72.6.params --threads $threads
$pypef ml --show
$pypef ml -e dca -l LS.fasta -t TS.fasta --regressor pls_loocv --params ANEH_72.6.params --threads $threads
$pypef ml --show
$pypef ml -e dca -l LS.fasta -t TS.fasta --regressor ridge --params ANEH_72.6.params --threads $threads
$pypef ml --show
#$pypef ml -e dca -l LS.fasta -t TS.fasta --regressor lasso --params ANEH_72.6.params --threads $threads  # programmed error due to no positive R2
#$pypef ml --show
#$pypef ml -e dca -l LS.fasta -t TS.fasta --regressor elasticnet --params ANEH_72.6.params --threads $threads  # programmed error due to no positive R2
#$pypef ml --show

$pypef ml --show

$pypef ml -e aaidx -m FAUJ880104 -f TS.fasta
$pypef ml -e onehot -m ONEHOTMODEL -f TS.fasta
$pypef ml -e dca -m DCAMODEL -f TS.fasta --params ANEH_72.6.params --threads $threads 
$pypef ml -e aaidx -m FAUJ880104 -f TS.fasta --label
$pypef ml -e onehot -m ONEHOTMODEL -f TS.fasta --label
$pypef ml -e dca -m DCAMODEL -f TS.fasta --label --params ANEH_72.6.params --threads $threads
$pypef ml -e aaidx -m FAUJ880104 -f TS.fasta --color --y_wt -1.5
$pypef ml -e onehot -m ONEHOTMODEL -f TS.fasta --color --y_wt -1.5
$pypef ml -e dca -m DCAMODEL -f TS.fasta --params ANEH_72.6.params --color --y_wt -1.5 --threads $threads

$pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
$pypef ml -e aaidx -m FAUJ880104 -p 37_ANEH_variants_prediction_set.fasta
$pypef ml -e onehot -m ONEHOTMODEL -p 37_ANEH_variants_prediction_set.fasta
$pypef ml -e dca -m DCAMODEL -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads

$pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --drecomb --trecomb --qarecomb --qirecomb --ddiverse
$pypef ml -e aaidx -m FAUJ880104 --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
$pypef ml -e onehot -m ONEHOTMODEL --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
$pypef ml -e dca -m DCAMODEL --params ANEH_72.6.params --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse --threads $threads

$pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative
$pypef ml -e onehot directevo -m ONEHOTMODEL -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative
$pypef ml -e dca directevo -m DCAMODEL -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative --params ANEH_72.6.params
$pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative
$pypef ml -e onehot directevo -m ONEHOTMODEL -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative
$pypef ml -e dca directevo -m DCAMODEL -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
$pypef ml -e aaidx directevo -m FAUJ880104 -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative
$pypef ml -e onehot directevo -m ONEHOTMODEL -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative
$pypef ml -e dca directevo -m DCAMODEL -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params

$pypef ml -e aaidx -l LS.fasta -t TS.fasta --regressor pls --nofft
$pypef ml --show
$pypef ml -e aaidx directevo -m WEBA780101 -w Sequence_WT_ANEH.fasta -y -1.5 --negative --nofft

$pypef encode -i 37_ANEH_variants.csv -e aaidx -m FAUJ880104 -w Sequence_WT_ANEH.fasta
$pypef encode -i 37_ANEH_variants.csv -e onehot -w Sequence_WT_ANEH.fasta
$pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads

$pypef ml low_n -i 37_ANEH_variants_aaidx_encoded.csv
$pypef ml low_n -i 37_ANEH_variants_onehot_encoded.csv
$pypef ml low_n -i 37_ANEH_variants_dca_encoded.csv

$pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv
$pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv
$pypef ml extrapolation -i 37_ANEH_variants_dca_encoded.csv

$pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv --conc
$pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv --conc
$pypef ml extrapolation -i 37_ANEH_variants_dca_encoded.csv --conc

$pypef hybrid train_and_save -i 37_ANEH_variants.csv --params ANEH_72.6.params --fit_size 0.66 --threads $threads
$pypef hybrid -l LS.fasta -t TS.fasta --params ANEH_72.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL -t TS.fasta --params ANEH_72.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL -f TS.fasta --params ANEH_72.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL -f TS.fasta --params ANEH_72.6.params --label --threads $threads

$pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
$pypef hybrid -m HYBRIDMODEL -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL --params ANEH_72.6.params --pmult --drecomb --threads $threads

$pypef hybrid directevo -m HYBRIDMODEL -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative --params ANEH_72.6.params
$pypef hybrid directevo -m HYBRIDMODEL -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
$pypef hybrid directevo -m HYBRIDMODEL -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params

$pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
$pypef hybrid low_n -i 37_ANEH_variants_dca_encoded.csv
$pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv
$pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv --conc


### Hybrid model (and some pure ML) tests on avGFP dataset 
cd '../test_dataset_avgfp'

$pypef encode -i avGFP.csv -e dca -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
$pypef encode -i avGFP.csv -e onehot -w P42212_F64L.fasta
$pypef mklsts -i avGFP.csv -w P42212_F64L.fasta
$pypef ml -e aaidx -l LS.fasta -t TS.fasta --threads $threads
$pypef ml --show
$pypef encode -i avGFP.csv -e aaidx -m GEIM800103 -w P42212_F64L.fasta 

$pypef hybrid train_and_save -i avGFP.csv --params uref100_avgfp_jhmmer_119_plmc_42.6.params --fit_size 0.66 --threads $threads

$pypef hybrid -l LS.fasta -t TS.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL -t TS.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL -f TS.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
$pypef hybrid -m HYBRIDMODEL -f TS.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --label --threads $threads

$pypef mkps -i avGFP.csv -w P42212_F64L.fasta
$pypef hybrid -m HYBRIDMODEL -p avGFP_prediction_set.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
$pypef mkps -i avGFP.csv -w P42212_F64L.fasta --drecomb
#$pypef hybrid -m HYBRIDMODEL --params uref100_avgfp_jhmmer_119_plmc_42.6.params --pmult --drecomb --threads $threads  # many single variants for recombination, takes too long

$pypef hybrid directevo -m HYBRIDMODEL -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params
$pypef hybrid directevo -m HYBRIDMODEL -w P42212_F64L.fasta --numiter 10 --numtraj 8 --params uref100_avgfp_jhmmer_119_plmc_42.6.params
$pypef hybrid directevo -m HYBRIDMODEL -i avGFP.csv -w P42212_F64L.fasta --temp 0.1 --usecsv --csvaa --params uref100_avgfp_jhmmer_119_plmc_42.6.params

$pypef hybrid low_n -i avGFP_dca_encoded.csv
$pypef hybrid extrapolation -i avGFP_dca_encoded.csv
$pypef hybrid extrapolation -i avGFP_dca_encoded.csv --conc

$pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
$pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
$pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge

$pypef ml low_n -i avGFP_onehot_encoded.csv --regressor pls
$pypef ml extrapolation -i avGFP_onehot_encoded.csv --regressor pls
$pypef ml extrapolation -i avGFP_onehot_encoded.csv --conc --regressor pls

$pypef ml low_n -i avGFP_aaidx_encoded.csv --regressor ridge
$pypef ml extrapolation -i avGFP_aaidx_encoded.csv --regressor ridge
$pypef ml extrapolation -i avGFP_aaidx_encoded.csv --conc --regressor ridge

echo 'All tests finished without error!'
