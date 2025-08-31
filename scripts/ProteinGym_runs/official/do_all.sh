#!/bin/bash


split_method=fold_random_5
 

 cd data
./download_data.sh
cd ..
 
./get_py_packages.sh

cd benchmark_runs
./run_over_all.sh split_method=$split_method > output.log 2>&1 &
