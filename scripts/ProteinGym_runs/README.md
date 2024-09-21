## Benchmark runs on publicly available ProteinGym protein variant sequence-fitness datasets

Data is taken (script-based download) from "DMS Assays"-->"Substitutions" and "Multiple Sequence Alignments"-->"DMS Assays" data from https://proteingym.org/download.
Run the following to download and extract the ProteinGym data and subsequently to get the predictions/the performance on those datasets.
```
#python -m pip install -r ../../requirements.txt
python -m pip install seaborn
python download_proteingym_and_extract_data.py
python run_performance_tests_proteingym_data.py
```
