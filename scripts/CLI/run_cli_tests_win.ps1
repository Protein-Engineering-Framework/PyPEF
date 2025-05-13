### PowerShell script for testing some PyPEF CLI commands 
### based on the two datasets provided (ANEH and avGFP)
### REQUIRES MINICONDA OR ANACONDA BEING INSTALLED
Write-Host "For successful running, following files are required:`n`nin test_dataset_aneh/
`tSequence_WT_ANEH.fasta`n`t37_ANEH_variants.csv`n`tANEH_jhmmer.a2m
`tANEH_72.6.params (generated using PLMC or dowloaded from https://github.com/niklases/PyPEF/blob/main/datasets/ANEH/ANEH_72.6.params)
`nin test_dataset_avgfp/`n`tP42212_F64L.fasta`n`tavGFP.csv
`turef100_avgfp_jhmmer_119.a2m
`turef100_avgfp_jhmmer_119_plmc_42.6.params (generated using PLMC or dowloaded from https://github.com/niklases/PyPEF/blob/main/datasets/AVGFP/uref100_avgfp_jhmmer_119_plmc_42.6.params)`n`n"

Set-PSDebug -Trace 1
$ErrorActionPreference = "Stop"
$PSDefaultParameterValues = @{
    'Write-Debug:Separator' = " (Line $($MyInvocation.ScriptLineNumber)): "
}

# exit on (PyPEF) errors
function ExitOnExitCode { if ($LastExitCode) { 
    Write-Host "PyPEF command error; terminating execution."; 
    Set-Location -Path $path'/scripts/CLI'; 
    exit } }

### RUN ME IN POWERSHELL WITH
### $ .\run_cli_tests_win.ps1                      # printing STDOUT and STDERR to terminal


$path=Get-Location                                                                                                       #
$path=Split-Path -Path $path -Parent                                                                                     #
$path=Split-Path -Path $path -Parent                                                                                     #
### if using downloaded/locally stored pypef .py files:
##########################################################################################################################
Write-Output Y | conda env remove -n pypef                                                                               #
conda create -n pypef python=3.12 -y                                                                                     #
conda activate pypef                                                                                                     #
python -m pip install -r $path\requirements.txt                                                                          #
#pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128               # ONLY IF NIGHTLY IS NEEDED, E.G., NEW BLACKWELL GPU GENERATION 
$env:PYTHONPATH=$path                                                                                                    #
function pypef { python $path\pypef\main.py @args }                                                                      #
##########################################################################################################################
### else just use pip-installed pypef version (uncomment):                                                               #
#pypef = pypef                                                                                                           #
##########################################################################################################################
# threads are only used for some parallelization of AAindex and DCA-based sequence encoding                              # 
# if pypef/settings.py defines USE_RAY = True                                                                            #
$threads = 12                                                                                                            #
##########################################################################################################################

### threads=1 shows progress bar where possible
### CV-based mlp and rf regression option take a long time and related testing commands are commented out/not included herein

### Pure ML (and some hybrid model) tests on ANEH dataset
Set-Location -Path $path'/datasets/ANEH'
#######################################################################
Write-Host
pypef --version
ExitOnExitCode
Write-Host
pypef -h
ExitOnExitCode
Write-Host
pypef mklsts -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
ExitOnExitCode
Write-Host

pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor pls
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor pls_loocv
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor ridge
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor lasso
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor elasticnet
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
#pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor mlp
#pypef ml --show
#pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor rf
#pypef ml --show

pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls_loocv --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor ridge --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor lasso --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor elasticnet --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host

pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --nofft --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls_loocv --nofft --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor ridge --nofft --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor lasso --nofft --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor elasticnet --nofft --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host

pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls_loocv --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor ridge --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor lasso --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor elasticnet --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef param_inference --msa ANEH_jhmmer.a2m --opt_iter 100
ExitOnExitCode
Write-Host
pypef save_msa_info --msa ANEH_jhmmer.a2m -w Sequence_WT_ANEH.fasta --opt_iter 100
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls_loocv --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor ridge --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor lasso --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor elasticnet --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host

pypef ml -e aaidx -m FAUJ880104 -t TS.fasl
ExitOnExitCode
Write-Host
pypef ml -e onehot -m ONEHOT -t TS.fasl
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLPLMC -t TS.fasl --params ANEH_72.6.params --threads $threads 
ExitOnExitCode
Write-Host
pypef ml -e aaidx -m FAUJ880104 -t TS.fasl --label
ExitOnExitCode
Write-Host
pypef ml -e onehot -m ONEHOT -t TS.fasl --label
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLPLMC -t TS.fasl --label --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host

pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
ExitOnExitCode
Write-Host
pypef ml -e aaidx -m FAUJ880104 -p 37_ANEH_variants_prediction_set.fasta
ExitOnExitCode
Write-Host
pypef ml -e onehot -m ONEHOT -p 37_ANEH_variants_prediction_set.fasta
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLPLMC -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLGREMLIN -p 37_ANEH_variants_prediction_set.fasta --params GREMLIN
ExitOnExitCode
Write-Host

pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --drecomb --trecomb --qarecomb --qirecomb --ddiverse
ExitOnExitCode
Write-Host
pypef ml -e aaidx -m FAUJ880104 --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
ExitOnExitCode
Write-Host
pypef ml -e onehot -m ONEHOT --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLPLMC --params ANEH_72.6.params --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse --threads $threads
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLGREMLIN --params GREMLIN --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
ExitOnExitCode
Write-Host

pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta --negative
ExitOnExitCode
Write-Host
pypef ml -e onehot directevo -m ONEHOT -w Sequence_WT_ANEH.fasta --negative
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLPLMC -w Sequence_WT_ANEH.fasta --negative --params ANEH_72.6.params
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLGREMLIN -w Sequence_WT_ANEH.fasta --negative --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative
ExitOnExitCode
Write-Host
pypef ml -e onehot directevo -m ONEHOT -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLPLMC -w Sequence_WT_ANEH.fasta --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLGREMLIN -w Sequence_WT_ANEH.fasta  --numiter 10 --numtraj 8 --negative --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml -e aaidx directevo -m FAUJ880104 -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta  --temp 0.1 --usecsv --csvaa --negative
ExitOnExitCode
Write-Host
pypef ml -e onehot directevo -m ONEHOT -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta  --temp 0.1 --usecsv --csvaa --negative
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLPLMC -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta  --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLGREMLIN -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta  --temp 0.1 --usecsv --csvaa --negative --params GREMLIN
ExitOnExitCode
Write-Host

pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --nofft
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef ml -e aaidx directevo -m WEBA780101 -w Sequence_WT_ANEH.fasta  --negative --nofft
ExitOnExitCode
Write-Host

pypef encode -i 37_ANEH_variants.csv -e aaidx -m FAUJ880104 -w Sequence_WT_ANEH.fasta
ExitOnExitCode
Write-Host
pypef encode -i 37_ANEH_variants.csv -e onehot -w Sequence_WT_ANEH.fasta
ExitOnExitCode
Write-Host
pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
Move-Item 37_ANEH_variants_dca_encoded.csv 37_ANEH_variants_plmc_dca_encoded.csv -erroraction 'silentlycontinue'
ExitOnExitCode
Write-Host
pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params GREMLIN
ExitOnExitCode
Write-Host
Move-Item 37_ANEH_variants_dca_encoded.csv 37_ANEH_variants_gremlin_dca_encoded.csv  -erroraction 'silentlycontinue'
ExitOnExitCode
Write-Host

pypef ml low_n -i 37_ANEH_variants_aaidx_encoded.csv
ExitOnExitCode
Write-Host
pypef ml low_n -i 37_ANEH_variants_onehot_encoded.csv
ExitOnExitCode
Write-Host
pypef ml low_n -i 37_ANEH_variants_plmc_dca_encoded.csv
ExitOnExitCode
Write-Host
pypef ml low_n -i 37_ANEH_variants_gremlin_dca_encoded.csv
ExitOnExitCode
Write-Host

pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv
ExitOnExitCode
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv
ExitOnExitCode
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_plmc_dca_encoded.csv
ExitOnExitCode
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_gremlin_dca_encoded.csv
ExitOnExitCode
Write-Host

pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv --conc
ExitOnExitCode
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv --conc
ExitOnExitCode
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_plmc_dca_encoded.csv --conc
ExitOnExitCode
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_gremlin_dca_encoded.csv --conc
ExitOnExitCode
Write-Host

pypef hybrid -l LS.fasl -t TS.fasl --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDPLMC -t TS.fasl --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host

pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDGREMLIN -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host

pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDPLMC -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDPLMC --params ANEH_72.6.params --pmult --drecomb --threads $threads
ExitOnExitCode
Write-Host

pypef hybrid directevo -m HYBRIDPLMC -w Sequence_WT_ANEH.fasta --negative --params ANEH_72.6.params
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDPLMC -w Sequence_WT_ANEH.fasta  --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params 
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDPLMC -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta  --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params
ExitOnExitCode
Write-Host

pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid low_n -i 37_ANEH_variants_dca_encoded.csv
ExitOnExitCode
Write-Host
pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv
ExitOnExitCode
Write-Host
pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv --conc
ExitOnExitCode
Write-Host

# 0.4.0 features: hybrid DCA-LLM modeling
pypef hybrid --ls LS.fasl --ts TS.fasl --params GREMLIN --llm esm
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDGREMLINESM1V --ts TS.fasl --params GREMLIN --llm esm
ExitOnExitCode
Write-Host
pypef mkps -i 37_ANEH_variants.csv --wt Sequence_WT_ANEH.fasta
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDGREMLINESM1V --ps 37_ANEH_variants_prediction_set.fasta --params GREMLIN --llm esm
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDGREMLINESM1V -w Sequence_WT_ANEH.fasta --negative --params GREMLIN
ExitOnExitCode
Write-Host


Remove-Item 37_ANEH_variants_plmc_dca_encoded.csv
ExitOnExitCode
Write-Host
Remove-Item 37_ANEH_variants_gremlin_dca_encoded.csv
ExitOnExitCode
Write-Host


### Hybrid model (and some pure ML and pure DCA) tests on avGFP dataset 
Set-Location -Path '../AVGFP'
#######################################################################
Write-Host

pypef mklsts -i avGFP.csv -w P42212_F64L.fasta
ExitOnExitCode
Write-Host
pypef param_inference --msa uref100_avgfp_jhmmer_119.a2m --opt_iter 100
ExitOnExitCode
Write-Host
# Check MSA coevolution info
pypef save_msa_info --msa uref100_avgfp_jhmmer_119.a2m -w P42212_F64L.fasta --opt_iter 100
###
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
# Similar to line above
pypef hybrid -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
# pure statistical
pypef hybrid -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host


# using .params file
pypef ml -e dca -l LS.fasl -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host
# ML LS/TS
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params uref100_avgfp_jhmmer_119_plmc_42.6.params
ExitOnExitCode
Write-Host
# Transforming .params file to DCAEncoding and using DCAEncoding Pickle; output file: Pickles/MLPLMC.
# That means using uref100_avgfp_jhmmer_119_plmc_42.6.params or PLMC as params file is identical.
pypef param_inference --params uref100_avgfp_jhmmer_119_plmc_42.6.params
ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params PLMC --threads $threads
ExitOnExitCode
Write-Host
# ml only TS
pypef ml -e dca -m MLPLMC -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLGREMLIN -t TS.fasl --params GREMLIN --threads $threads
ExitOnExitCode
Write-Host


ExitOnExitCode
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --params PLMC --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params PLMC --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -m PLMC -t TS.fasl --params PLMC --threads $threads
ExitOnExitCode
Write-Host

# Hybrid: pure statistical
pypef hybrid -t TS.fasl --params PLMC --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -p TS.fasl --params PLMC --threads $threads
ExitOnExitCode
Write-Host
# Same as above command
pypef hybrid -p TS.fasl -m PLMC --params PLMC --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -p TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
# Same as above command
pypef hybrid -p TS.fasl -m GREMLIN --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host
pypef save_msa_info --msa uref100_avgfp_jhmmer_119.a2m -w P42212_F64L.fasta --opt_iter 100
ExitOnExitCode
Write-Host

# Encode CSV
pypef encode -e dca -i avGFP.csv --wt P42212_F64L.fasta --params GREMLIN
ExitOnExitCode
Write-Host
pypef encode --encoding dca -i avGFP.csv --wt P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads 12

#Extrapolation
ExitOnExitCode
Write-Host
pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge
ExitOnExitCode
Write-Host

# Direct Evo
pypef ml -e dca directevo -m MLGREMLIN --wt P42212_F64L.fasta --params GREMLIN
ExitOnExitCode
Write-Host
pypef ml -e dca directevo -m MLPLMC --wt P42212_F64L.fasta --params PLMC
ExitOnExitCode
Write-Host
pypef hybrid directevo -m GREMLIN --wt P42212_F64L.fasta --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid directevo -m PLMC --wt P42212_F64L.fasta --params PLMC
ExitOnExitCode
Write-Host
pypef hybrid directevo --wt P42212_F64L.fasta --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid directevo --wt P42212_F64L.fasta --params PLMC
ExitOnExitCode
Write-Host

pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host 
pypef hybrid -m HYBRIDGREMLIN -t TS.fasl --params GREMLIN
ExitOnExitCode
Write-Host 

### Similar to old CLI run test from here

pypef encode -i avGFP.csv -e dca -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef encode -i avGFP.csv -e onehot -w P42212_F64L.fasta
ExitOnExitCode
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --threads $threads
ExitOnExitCode
Write-Host
pypef ml --show
ExitOnExitCode
Write-Host
pypef encode -i avGFP.csv -e aaidx -m GEIM800103 -w P42212_F64L.fasta 
ExitOnExitCode
Write-Host

pypef hybrid -l LS.fasl -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDPLMC -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host

# No training set given
pypef hybrid -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef ml -e dca -m MLPLMC -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --label --threads $threads
ExitOnExitCode
Write-Host

pypef mkps -i avGFP.csv -w P42212_F64L.fasta
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDPLMC -p avGFP_prediction_set.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
ExitOnExitCode
Write-Host
pypef mkps -i avGFP.csv -w P42212_F64L.fasta --drecomb
ExitOnExitCode
Write-Host
# many single variants for recombination, takes too long
#pypef hybrid -m HYBRIDPLMC --params uref100_avgfp_jhmmer_119_plmc_42.6.params --pmult --drecomb --threads $threads  
#ExitOnExitCode
#Write-Host
pypef hybrid -m HYBRIDGREMLIN --params GREMLIN --pmult --drecomb
ExitOnExitCode
Write-Host

pypef hybrid directevo -m HYBRIDPLMC -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDGREMLIN -w P42212_F64L.fasta --params GREMLIN
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDPLMC -w P42212_F64L.fasta --numiter 10 --numtraj 8 --params uref100_avgfp_jhmmer_119_plmc_42.6.params
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDPLMC -i avGFP.csv -w P42212_F64L.fasta --temp 0.1 --usecsv --csvaa --params uref100_avgfp_jhmmer_119_plmc_42.6.params
ExitOnExitCode
Write-Host

# 0.4.0 features: hybrid DCA-LLM modeling
pypef mklsts -i avGFP.csv -w P42212_F64L.fasta --ls_proportion 0.02
ExitOnExitCode
Write-Host
pypef hybrid --ls LS.fasl --ts TS.fasl --params GREMLIN --llm esm
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDGREMLINESM1V --ts TS.fasl --params GREMLIN --llm esm
ExitOnExitCode
Write-Host

pypef hybrid --ls LS.fasl --ts TS.fasl --params GREMLIN --llm prosst --wt P42212_F64L.fasta  --pdb GFP_AEQVI.pdb
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDGREMLINPROSST --ts TS.fasl --params GREMLIN --llm prosst --wt P42212_F64L.fasta  --pdb GFP_AEQVI.pdb
ExitOnExitCode
Write-Host

pypef hybrid directevo -m HYBRIDGREMLINESM1V -w P42212_F64L.fasta --params GREMLIN --llm esm
ExitOnExitCode
Write-Host
pypef hybrid directevo -m HYBRIDGREMLINPROSST -w P42212_F64L.fasta --params GREMLIN --llm prosst --pdb GFP_AEQVI.pdb
ExitOnExitCode
Write-Host

# Takes long.. better delete 7 out of the 8 recomb txt files
#pypef hybrid -m HYBRIDGREMLINESM1V -w P42212_F64L.fasta --params GREMLIN --llm esm --pmult --drecomb
#ExitOnExitCode
#Write-Host
#pypef hybrid -m HYBRIDGREMLINPROSST -w P42212_F64L.fasta --params GREMLIN --llm prosst --pdb GFP_AEQVI.pdb --pmult --drecomb
#ExitOnExitCode
#Write-Host
pypef hybrid -m HYBRIDGREMLINESM1V -w P42212_F64L.fasta --params GREMLIN --llm esm -p avGFP_prediction_set.fasta
ExitOnExitCode
Write-Host
pypef hybrid -m HYBRIDGREMLINPROSST -w P42212_F64L.fasta --params GREMLIN --llm prosst --pdb GFP_AEQVI.pdb -p avGFP_prediction_set.fasta
ExitOnExitCode
Write-Host

pypef hybrid low_n -i avGFP_dca_encoded.csv
ExitOnExitCode
Write-Host
pypef hybrid extrapolation -i avGFP_dca_encoded.csv
ExitOnExitCode
Write-Host
pypef hybrid extrapolation -i avGFP_dca_encoded.csv --conc
ExitOnExitCode
Write-Host

pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge
ExitOnExitCode
Write-Host

pypef ml low_n -i avGFP_onehot_encoded.csv --regressor pls
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_onehot_encoded.csv --regressor pls
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_onehot_encoded.csv --conc --regressor pls
ExitOnExitCode
Write-Host

pypef ml low_n -i avGFP_aaidx_encoded.csv --regressor ridge
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_aaidx_encoded.csv --regressor ridge
ExitOnExitCode
Write-Host
pypef ml extrapolation -i avGFP_aaidx_encoded.csv --conc --regressor ridge
ExitOnExitCode
Write-Host

ExitOnExitCode
Write-Host 'All runs finished without error!'