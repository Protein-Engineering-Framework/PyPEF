#!/bin/bash

# Run we with ./conda_install_test.sh

# Easy "installer" for downloading repository and running local Python files (not pip-installed).
# Testing a few PyPEF commands on downloaded test data.
# Requires conda, i.e. Anaconda3 or Miniconda3 [https://docs.conda.io/en/latest/miniconda.html].

# 1. wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
# 2. bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh


# Echo on
set -x  
# Exit on errors
set -e
# echo script line numbers
export PS4='+(Line ${LINENO}): '

# Alternatively to using conda, you can use Python 3.10 and install packages via "python3 -m pip install -r requirements.txt"
#conda update -n base -c defaults conda
conda env remove -n pypef
conda create -n pypef python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate pypef
python3 -m pip install -U pypef
pypef --version

while true; do
    read -p "Test PyPEF installation (downloads PyPEF example test set files and runs a Python test script, ~ 1 h run time) [Y/N]? " yn
    case $yn in
        [Yy]* ) 
			mkdir test_dataset_avgfp
			wget https://raw.githubusercontent.com/Protein-Engineering-Framework/PyPEF/master/workflow/test_dataset_avgfp/avGFP.csv
			wget https://raw.githubusercontent.com/Protein-Engineering-Framework/PyPEF/master/workflow/test_dataset_avgfp/uref100_avgfp_jhmmer_119.a2m
			wget https://raw.githubusercontent.com/Protein-Engineering-Framework/PyPEF/master/workflow/api_encoding_train_test.py
			mv avGFP.csv ./test_dataset_avgfp/avGFP.csv
			mv uref100_avgfp_jhmmer_119.a2m ./test_dataset_avgfp/uref100_avgfp_jhmmer_119.a2m
			python3 ./api_encoding_train_test.py
			break;;
        [Nn]* ) 
			break;;
        * ) 
			echo "Please answer yes or no.";;
    esac
done

echo 'Done!';



