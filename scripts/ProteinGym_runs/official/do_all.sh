#!/bin/bash
set -e

# Pre-download models from HuggingFace
#hf download facebook/esm2_t33_650M_UR50D
#hf download AI4Protein/ProSST-2048

# "fold_rand_multiples" "fold_random_5" "fold_modulo_5" "fold_contiguous_5"
split_methods=("fold_random_5" "fold_modulo_5" "fold_contiguous_5")
 
 cd data
./download_data.sh
cd ..
 
./get_py_packages.sh

cd benchmark_runs

for split_method in "${split_methods[@]}"; do
    ./run_over_all.sh split_method=$split_method # > output.log 2>&1 &
    done

python estimate_performance.py