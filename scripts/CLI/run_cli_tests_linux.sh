#!/bin/bash

### Bash script for testing some PyPEF CLI commands 
### based on the two datasets provided (ANEH and avGFP)
### REQUIRES MINICONDA OR ANACONDA BEING INSTALLED
printf 'For successful running, following files are required:\n\nin test_dataset_aneh/\n\tSequence_WT_ANEH.fasta\n\t37_ANEH_variants.csv
\tANEH_jhmmer.a2m\n\tANEH_72.6.params (generated using PLMC or dowloaded from https://github.com/niklases/PyPEF/blob/main/datasets/ANEH/ANEH_72.6.params)\n
in test_dataset_avgfp/\n\tP42212_F64L.fasta\n\tavGFP.csv\n\turef100_avgfp_jhmmer_119.a2m
\turef100_avgfp_jhmmer_119_plmc_42.6.params (generated using PLMC or dowloaded from https://github.com/niklases/PyPEF/blob/main/datasets/AVGFP/uref100_avgfp_jhmmer_119_plmc_42.6.params)\n\n'

set -x  # echo on
set -e  # exit on (PyPEF) errors
export PS4='+(Line ${LINENO}): '  # echo script line numbers 

### RUN ME FROM CURRENT FILE DIRECTORY:
### $ ./run_cli_tests_linux.sh                      # printing STDOUT and STDERR to terminal
### $ ./run_cli_tests_linux.sh &> test_cli_run.log  # writing STDOUT and STDERR to log file


cd '../'
path=$( echo ${PWD%/*} )
cd 'CLI'  
### if using downloaded/locally stored pypef .py files:
##########################################################################################################################
yes | conda env remove -n pypef || true                                                                                  #
conda create -n pypef python=3.12 -y                                                                                     #
eval "$(conda shell.bash hook)"                                                                                          #
conda activate pypef                                                                                                     #
python -m pip install -r "$path/requirements.txt"                                                                        #
#pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128               # ONLY IF NIGHTLY IS NEEDED, E.G., NEW BLACKWELL GPU GENERATION 
export PYTHONPATH=${PYTHONPATH}:$path                                                                                    #
pypef="python3 $path/pypef/main.py"                                                                                      #
##########################################################################################################################
### else just use pip-installed pypef version (uncomment):                                                               #
#pypef=pypef                                                                                                             #
##########################################################################################################################
# threads are only used for some parallelization of AAindex and DCA-based sequence encoding                              # 
# if pypef/settings.py defines USE_RAY = True                                                                            #
threads=12                                                                                                               #
##########################################################################################################################

### threads=1 shows progress bar where possible
### CV-based mlp and rf regression option take a long time and related testing commands are commented out/not included herein

### Pure ML (and some hybrid model) tests on ANEH dataset
cd "$path/datasets/ANEH"
#######################################################################
echo

$pypef --version
echo
$pypef -h
echo
$pypef mklsts -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
echo

$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor pls --label
echo
$pypef ml --show
echo
$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor pls_loocv
echo
$pypef ml --show
echo
$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor ridge
echo
$pypef ml --show
echo
$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor lasso
echo
$pypef ml --show
echo
$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor elasticnet
echo
$pypef ml --show
echo
#$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor mlp
#$pypef ml --show
#$pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor rf
#$pypef ml --show

$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls_loocv --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor ridge --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor lasso --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor elasticnet --threads $threads
echo
$pypef ml --show
echo

$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --nofft --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls_loocv --nofft --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor ridge --nofft --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor lasso --nofft --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor elasticnet --nofft --threads $threads
echo
$pypef ml --show
echo

$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params ANEH_72.6.params --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls_loocv --params ANEH_72.6.params --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor ridge --params ANEH_72.6.params --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor lasso --params ANEH_72.6.params --threads $threads
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor elasticnet --params ANEH_72.6.params --threads $threads
echo
$pypef ml --show
echo

$pypef param_inference --msa ANEH_jhmmer.a2m --opt_iter 100
echo
$pypef save_msa_info --msa ANEH_jhmmer.a2m -w Sequence_WT_ANEH.fasta --opt_iter 100
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params GREMLIN
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls_loocv --params GREMLIN
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor ridge --params GREMLIN
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor lasso --params GREMLIN
echo
$pypef ml --show
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor elasticnet --params GREMLIN
echo
$pypef ml --show
echo

$pypef ml -e aaidx -m FAUJ880104 -t TS.fasl
echo
$pypef ml -e onehot -m ONEHOT -t TS.fasl
echo
$pypef ml -e dca -m MLPLMC -t TS.fasl --params ANEH_72.6.params --threads $threads 
echo
$pypef ml -e aaidx -m FAUJ880104 -t TS.fasl --label
echo
$pypef ml -e onehot -m ONEHOT -t TS.fasl --label
echo
$pypef ml -e dca -m MLPLMC -t TS.fasl --label --params ANEH_72.6.params --threads $threads
echo

$pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
echo
$pypef ml -e aaidx -m FAUJ880104 -p 37_ANEH_variants_prediction_set.fasta
echo
$pypef ml -e onehot -m ONEHOT -p 37_ANEH_variants_prediction_set.fasta
echo
$pypef ml -e dca -m MLPLMC -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
echo
$pypef ml -e dca -m MLgremlin -p 37_ANEH_variants_prediction_set.fasta --params GREMLIN
echo

$pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --drecomb --trecomb --qarecomb --qirecomb --ddiverse
echo
$pypef ml -e aaidx -m FAUJ880104 --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
echo
$pypef ml -e onehot -m ONEHOT --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
echo
$pypef ml -e dca -m MLPLMC --params ANEH_72.6.params --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse --threads $threads
echo
$pypef ml -e dca -m MLgremlin --params GREMLIN --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
echo

$pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta --negative
echo
$pypef ml -e onehot directevo -m ONEHOT -w Sequence_WT_ANEH.fasta --negative
echo
$pypef ml -e dca directevo -m MLPLMC -w Sequence_WT_ANEH.fasta --negative --params ANEH_72.6.params
echo
$pypef ml -e dca directevo -m MLgremlin -w Sequence_WT_ANEH.fasta --negative --params GREMLIN
echo
$pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative
echo
$pypef ml -e onehot directevo -m ONEHOT -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative
echo
$pypef ml -e dca directevo -m MLPLMC -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
echo
$pypef ml -e dca directevo -m MLgremlin -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative --params GREMLIN
echo
$pypef ml -e aaidx directevo -m FAUJ880104 -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --temp 0.1 --usecsv --csvaa --negative
echo
$pypef ml -e onehot directevo -m ONEHOT -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --temp 0.1 --usecsv --csvaa --negative
echo
$pypef ml -e dca directevo -m MLPLMC -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params
echo
$pypef ml -e dca directevo -m MLgremlin -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --temp 0.1 --usecsv --csvaa --negative --params GREMLIN
echo

$pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --nofft
echo
$pypef ml --show
echo
$pypef ml -e aaidx directevo -m WEBA780101 -w Sequence_WT_ANEH.fasta --negative --nofft
echo

$pypef encode -i 37_ANEH_variants.csv -e aaidx -m FAUJ880104 -w Sequence_WT_ANEH.fasta
echo
$pypef encode -i 37_ANEH_variants.csv -e onehot -w Sequence_WT_ANEH.fasta
echo
$pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
echo
mv 37_ANEH_variants_dca_encoded.csv 37_ANEH_variants_plmc_dca_encoded.csv || true
echo
$pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params GREMLIN
echo
mv 37_ANEH_variants_dca_encoded.csv 37_ANEH_variants_gremlin_dca_encoded.csv || true
echo

$pypef ml low_n -i 37_ANEH_variants_aaidx_encoded.csv
echo
$pypef ml low_n -i 37_ANEH_variants_onehot_encoded.csv
echo
$pypef ml low_n -i 37_ANEH_variants_plmc_dca_encoded.csv
echo
$pypef ml low_n -i 37_ANEH_variants_gremlin_dca_encoded.csv
echo

$pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv
echo
$pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv
echo
$pypef ml extrapolation -i 37_ANEH_variants_plmc_dca_encoded.csv
echo
$pypef ml extrapolation -i 37_ANEH_variants_gremlin_dca_encoded.csv
echo

$pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv --conc
echo
$pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv --conc
echo
$pypef ml extrapolation -i 37_ANEH_variants_plmc_dca_encoded.csv --conc
echo
$pypef ml extrapolation -i 37_ANEH_variants_gremlin_dca_encoded.csv --conc
echo

$pypef hybrid -l LS.fasl -t TS.fasl --params ANEH_72.6.params --threads $threads
echo
$pypef hybrid -m HYBRIDPLMC -t TS.fasl --params ANEH_72.6.params --threads $threads
echo

$pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
echo
$pypef hybrid -m HYBRIDGREMLIN -t TS.fasl --params GREMLIN
echo

$pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
echo
$pypef hybrid -m HYBRIDPLMC -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
echo
$pypef hybrid -m HYBRIDPLMC --params ANEH_72.6.params --pmult --drecomb --threads $threads
echo

$pypef hybrid directevo -m HYBRIDPLMC -w Sequence_WT_ANEH.fasta --negative --params ANEH_72.6.params
echo
$pypef hybrid directevo -m HYBRIDPLMC -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
echo
$pypef hybrid directevo -m HYBRIDPLMC -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params
echo

$pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
echo
$pypef hybrid low_n -i 37_ANEH_variants_dca_encoded.csv
echo
$pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv
echo
$pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv --conc
echo

# 0.4.0 features: hybrid DCA-LLM modeling
$pypef hybrid --ls LS.fasl --ts TS.fasl --params GREMLIN --llm esm
echo
$pypef hybrid -m HYBRIDGREMLINESM1V --ts TS.fasl --params GREMLIN --llm esm
echo
$pypef mkps -i 37_ANEH_variants.csv --wt Sequence_WT_ANEH.fasta
echo
$pypef hybrid -m HYBRIDGREMLINESM1V --ps 37_ANEH_variants_prediction_set.fasta --params GREMLIN --llm esm
echo
$pypef hybrid directevo -m HYBRIDGREMLINESM1V -w Sequence_WT_ANEH.fasta --negative --params GREMLIN
echo

rm 37_ANEH_variants_plmc_dca_encoded.csv
echo
rm 37_ANEH_variants_gremlin_dca_encoded.csv
echo

### Hybrid model (and some pure ML and pure DCA) tests on avGFP dataset 
cd '../AVGFP'
#######################################################################
echo

$pypef mklsts -i avGFP.csv -w P42212_F64L.fasta
echo
$pypef param_inference --msa uref100_avgfp_jhmmer_119.a2m --opt_iter 100
echo
# Check MSA coevolution info
$pypef save_msa_info --msa uref100_avgfp_jhmmer_119.a2m -w P42212_F64L.fasta --opt_iter 100
###
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --params GREMLIN
echo
$pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
echo
# Similar to line above
$pypef hybrid -t TS.fasl --params GREMLIN
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --params GREMLIN
echo
$pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
echo
$pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
echo
# pure statistical
$pypef hybrid -t TS.fasl --params GREMLIN
echo

# using .params file
$pypef ml -e dca -l LS.fasl -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo
# ML LS/TS
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params GREMLIN
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params uref100_avgfp_jhmmer_119_plmc_42.6.params
echo
# Transforming .params file to DCAEncoding and using DCAEncoding Pickle; output file: Pickles/MLPLMC.
# That means using uref100_avgfp_jhmmer_119_plmc_42.6.params or PLMC as params file is identical.
$pypef param_inference --params uref100_avgfp_jhmmer_119_plmc_42.6.params
echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params PLMC --threads $threads
echo
# ml only TS
$pypef ml -e dca -m MLPLMC -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo
$pypef ml -e dca -m MLgremlin -t TS.fasl --params GREMLIN --threads $threads
echo

echo
$pypef ml -e dca -l LS.fasl -t TS.fasl --params PLMC --threads $threads
echo
$pypef hybrid -l LS.fasl -t TS.fasl --params PLMC --threads $threads
echo
$pypef hybrid -m PLMC -t TS.fasl --params PLMC --threads $threads
echo

# No training set given: Statistical prediction, Hybrid: pure statistical
$pypef hybrid -t TS.fasl --params PLMC --threads $threads
echo
$pypef hybrid -p TS.fasl --params PLMC --threads $threads
echo
# Same as above command
$pypef hybrid -p TS.fasl -m PLMC --params PLMC --threads $threads
echo
$pypef hybrid -t TS.fasl --params GREMLIN
echo
$pypef hybrid -p TS.fasl --params GREMLIN
echo
$pypef hybrid -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo

# Same as above command
$pypef hybrid -p TS.fasl -m GREMLIN --params GREMLIN
echo
$pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
echo
$pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
echo
$pypef save_msa_info --msa uref100_avgfp_jhmmer_119.a2m -w P42212_F64L.fasta --opt_iter 100
# Encode CSV
$pypef encode -e dca -i avGFP.csv --wt P42212_F64L.fasta --params GREMLIN
echo
$pypef encode --encoding dca -i avGFP.csv --wt P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads 12

#Extrapolation
echo
$pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
echo
$pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
echo
$pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge
echo

# Direct Evo
$pypef ml -e dca directevo -m MLgremlin --wt P42212_F64L.fasta --params GREMLIN
echo
$pypef ml -e dca directevo -m MLPLMC --wt P42212_F64L.fasta --params PLMC
echo
$pypef hybrid directevo -m GREMLIN --wt P42212_F64L.fasta --params GREMLIN
echo
$pypef hybrid directevo -m PLMC --wt P42212_F64L.fasta --params PLMC
echo
$pypef hybrid directevo --wt P42212_F64L.fasta --params GREMLIN
echo
$pypef hybrid directevo --wt P42212_F64L.fasta --params PLMC
echo

# Hybrid prediction using GREMLIN parameters
$pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
echo 
$pypef hybrid -m HYBRIDGREMLIN -t TS.fasl --params GREMLIN
echo 


$pypef encode -i avGFP.csv -e dca -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo
$pypef encode -i avGFP.csv -e onehot -w P42212_F64L.fasta
echo
$pypef ml -e aaidx -l LS.fasl -t TS.fasl --threads $threads
echo
$pypef ml --show
echo
$pypef encode -i avGFP.csv -e aaidx -m GEIM800103 -w P42212_F64L.fasta 
echo

$pypef hybrid -l LS.fasl -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo
$pypef hybrid -m HYBRIDPLMC -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo

$pypef ml -e dca -m MLPLMC -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --label --threads $threads
echo

$pypef mkps -i avGFP.csv -w P42212_F64L.fasta
echo
$pypef hybrid -m HYBRIDPLMC -p avGFP_prediction_set.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
echo
$pypef mkps -i avGFP.csv -w P42212_F64L.fasta --drecomb
echo
# many single variants for recombination, takes too long
#$pypef hybrid -m HYBRIDPLMC --params uref100_avgfp_jhmmer_119_plmc_42.6.params --pmult --drecomb --threads $threads  
#echo
$pypef hybrid -m HYBRIDGREMLIN --params GREMLIN --pmult --drecomb
echo

$pypef hybrid directevo -m HYBRIDPLMC -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params
echo
$pypef hybrid directevo -m HYBRIDPLMC -w P42212_F64L.fasta --numiter 10 --numtraj 8 --params uref100_avgfp_jhmmer_119_plmc_42.6.params
echo
$pypef hybrid directevo -m HYBRIDPLMC -i avGFP.csv -w P42212_F64L.fasta --temp 0.1 --usecsv --csvaa --params uref100_avgfp_jhmmer_119_plmc_42.6.params

# 0.4.0 features: hybrid DCA-LLM modeling
$pypef hybrid --ls LS.fasl --ts TS.fasl --params GREMLIN --llm esm
echo
$pypef hybrid -m HYBRIDGREMLINESM1V --ts TS.fasl --params GREMLIN --llm esm
echo

$pypef hybrid --ls LS.fasl --ts TS.fasl --params GREMLIN --llm prosst --wt P42212_F64L.fasta  --pdb GFP_AEQVI.pdb
echo
$pypef hybrid -m HYBRIDGREMLINPROSST --ts TS.fasl --params GREMLIN --llm esm --llm prosst --wt P42212_F64L.fasta  --pdb GFP_AEQVI.pdb
echo

$pypef hybrid directevo -m HYBRIDGREMLINESM1V -w P42212_F64L.fasta --params GREMLIN --llm esm
ExitOnExitCode
Write-Host
$pypef hybrid directevo -m HYBRIDGREMLINPROSST -w P42212_F64L.fasta --params GREMLIN --llm prosst --pdb GFP_AEQVI.pdb
ExitOnExitCode
Write-Host

# Takes long.. better delete 7 out of the 8 recomb txt files
#$pypef hybrid -m HYBRIDGREMLINESM1V -w P42212_F64L.fasta --params GREMLIN --llm esm --pmult --drecomb
#echo
#$pypef hybrid -m HYBRIDGREMLINPROSST -w P42212_F64L.fasta --params GREMLIN --llm prosst --pdb GFP_AEQVI.pdb --pmult --drecomb
#echo
$pypef hybrid -m HYBRIDGREMLINESM1V -w P42212_F64L.fasta --params GREMLIN --llm esm -p avGFP_prediction_set.fasta
echo
$pypef hybrid -m HYBRIDGREMLINPROSST -w P42212_F64L.fasta --params GREMLIN --llm prosst --pdb GFP_AEQVI.pdb -p avGFP_prediction_set.fasta
echo




$pypef hybrid low_n -i avGFP_dca_encoded.csv
echo
$pypef hybrid extrapolation -i avGFP_dca_encoded.csv
echo
$pypef hybrid extrapolation -i avGFP_dca_encoded.csv --conc
echo

$pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
echo
$pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
echo
$pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge
echo

$pypef ml low_n -i avGFP_onehot_encoded.csv --regressor pls
echo
$pypef ml extrapolation -i avGFP_onehot_encoded.csv --regressor pls
echo
$pypef ml extrapolation -i avGFP_onehot_encoded.csv --conc --regressor pls
echo

$pypef ml low_n -i avGFP_aaidx_encoded.csv --regressor ridge
echo
$pypef ml extrapolation -i avGFP_aaidx_encoded.csv --regressor ridge
echo
$pypef ml extrapolation -i avGFP_aaidx_encoded.csv --conc --regressor ridge
echo

echo 'All runs finished without error!'
