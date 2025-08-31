# Run ProteinGym cross-validation benchmark

Download and extract required data from ProteinGym (if not cached already):
```
cd data
chmod a+x ./download_data.sh
./download_data.sh
cd ..
```

Install package requirements:
```
chmod a+x ./get_py_packages.sh
./get_py_packages.sh
```

Example random_fold run for first dataset `DMS_idx=0` (llm options currently implemented: ESM1v (`llm=esm1v`) and ProSST (`llm=prosst`)):
```bash
cd benchmark_runs
python pgym_cv_benchmark.py split_method=fold_random_5 DMS_idx=0 llm=prosst # overwrite=true
```

For benchmarking all datasets for a specific CV split method (e.g., fold_random_5) you can run:
```bash
cd benchmark_runs
chmod a+x ./run_over_all.sh
./run_over_all.sh split_method=fold_random_5
```

All steps can also performed with running `do_all.sh` (writes output to output.log, where running `python remove_tqdm_lines_log.py` is handy for removing the tqdm progress bars from the output log for eased log reading).
