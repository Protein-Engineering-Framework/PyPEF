## Benchmark runs on publicly available ProteinGym protein variant sequence-fitness datasets

Data is taken (script-based download) from 

"DMS Assays"-->"Substitutions" and "Multiple Sequence Alignments"-->"DMS Assays" data 

from https://proteingym.org/download.

Perform the following steps to download and extract the ProteinGym data and then obtain the predictions/performance for these datasets.
Depending on the available GPU/VRAM, the variable `MAX_WT_SEQUENCE_LENGTH` in the scripts must be adjusted according to the available (V)RAM. For example, the results ([results/dca_esm_and_hybrid_opt_results_v0.4.3_clean.csv](results/dca_esm_and_hybrid_opt_results_v0.4.3_clean.csv), shown graphically on the main README page) were calculated with an NVIDIA GeForce RTX 5090 with 32 GB VRAM and the setting `MAX_WT_SEQUENCE_LENGTH = 1000`:

```sh
#python -m pip install -r ../../requirements.txt
python -m pip install seaborn
python download_proteingym_and_extract_data.py
python protgym_hybrid_perf_test_low_n.py
# and / or
#python protgym_hybrid_perf_test_crossval.py
```
