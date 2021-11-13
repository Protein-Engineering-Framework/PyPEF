This repository contains supplementary information to

Niklas E. Siedhoff<sup>*1,§*</sup>, Alexander-Maurice Illig<sup>*1,§*</sup>, Ulrich Schwaneberg<sup>*1,2*</sup>, Mehdi D. Davari<sup>*1,\**</sup>, <br>
PyPEF – an Integrated Framework for Data-driven Protein Engineering, *Journal of Chemical Information and Modeling* (2021) <br>
<sup>*1*</sup><sub>Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany</sub> <br>
<sup>*2*</sup><sub>DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany</sub> <br>
<sup>*\**</sup><sub>Corresponding author</sub> <br>
<sup>*§*</sup><sub>Equal contribution</sub> <br>

# An exemplary use of PyPEF code as an API
Scripting examples of using `pypef/api/\*` modules. `pypef/api/\*` modules contain some classes and functions adapted from pypef/cli/regression.py for 
constructing a small API that might be useful for encoding amino acid sequences to generate raw encodings or FFT-ed sequence encodings that can 
subsequently be used for model construction and validation similar to the PyPEF routine. Further, some ready-to-use cross-validation-tuned 
regression options (using `sklearn.model_selection.GridSearchCV`)are provided.

An exemplary running script is provided in api_usage_example.py that uses functions of module `pypef.api`. The exemplary dataset (sequences and 
respective labels) is imported from exemplary_dataset_B.py.

