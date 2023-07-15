#!/bin/bash

# Run we with ./easy_install_test.sh

# Easy "installer" for downloading repository and running local Python files (not pip-installed).
# Testing a few PyPEF commands on downloaded example files.
# Requires conda, i.e. Anaconda3 or Miniconda3 [https://docs.conda.io/en/latest/miniconda.html].

# 1. wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
# 2. bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh


# Echo on
set -x  
# Exit on errors
set -e
# echo script line numbers
export PS4='+(Line ${LINENO}): '

wget https://github.com/Protein-Engineering-Framework/PyPEF/archive/refs/heads/master.zip

sudo apt-get update
sudo apt-get install unzip

unzip master.zip

# Alternatively to using conda, you can use Python 3.10 and install packages via "python3 -m pip install -r requirements.txt"
#conda update -n base -c defaults conda
conda env remove -n pypef
#conda env create -f PyPEF-master/linux_env.yml
conda create -n pypef python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate pypef
python3 -m pip install -U pypef

#export PYTHONPATH=${PYTHONPATH}:${PWD}/PyPEF-master
#pypef='python3 '${PWD}'/PyPEF-master/pypef/main.py'
pypef --version

cd 'PyPEF-master/workflow'

python3 ./api_encoding_train_test.py
