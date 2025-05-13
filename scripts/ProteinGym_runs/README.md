## Benchmark runs on publicly available ProteinGym protein variant sequence-fitness datasets

Data is taken (script-based download) from "DMS Assays"-->"Substitutions" and "Multiple Sequence Alignments"-->"DMS Assays" data from https://proteingym.org/download.
Run the following to download and extract the ProteinGym data and subsequently to get the predictions/the performance on those datasets.
Based on available GPU/VRAM, variable `MAX_WT_SEQUENCE_LENGTH` in script [run_performance_tests_proteingym_hybrid_dca_llm.py](run_performance_tests_proteingym_hybrid_dca_llm.py) has to adjusted according to available (V)RAM. E.g., results ([results/dca_esm_and_hybrid_opt_results.csv](results/dca_esm_and_hybrid_opt_results.csv), graphically presented on the main page README) were computed with an NVIDIA GeForce RTX 5090 with 32 GB VRAM and setting `MAX_WT_SEQUENCE_LENGTH` to 1000 (GPU power limit set to 520 W):

```sh
#python -m pip install -r ../../requirements.txt
python -m pip install seaborn
python download_proteingym_and_extract_data.py
python run_performance_tests_proteingym_hybrid_dca_llm.py
```
