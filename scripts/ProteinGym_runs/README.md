## Benchmark runs on publicly available ProteinGym protein variant sequence-fitness datasets

Data is taken (script-based download) from 

"DMS Assays"-->"Substitutions" and "Multiple Sequence Alignments"-->"DMS Assays" data 

from https://proteingym.org/download.

Perform the following steps to download and extract the ProteinGym data and then obtain the predictions/performance for these datasets.
Depending on the available GPU/VRAM, the variable `MAX_WT_SEQUENCE_LENGTH` in the script [run_performance_tests_proteingym_hybrid_dca_llm.py](run_performance_tests_proteingym_hybrid_dca_llm.py) must be adjusted according to the available (V)RAM. For example, the results ([results/dca_esm_and_hybrid_opt_results.csv](results/dca_esm_and_hybrid_opt_results.csv), shown graphically on the main README page) were calculated with an NVIDIA GeForce RTX 5090 with 32 GB VRAM and the setting `MAX_WT_SEQUENCE_LENGTH = 1000` (GPU power limit set to 520 W):

```sh
#python -m pip install -r ../../requirements.txt
python -m pip install seaborn
python download_proteingym_and_extract_data.py
python run_performance_tests_proteingym_hybrid_dca_llm.py
```
