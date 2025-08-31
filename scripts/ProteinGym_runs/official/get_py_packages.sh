#!/bin/bash

gh repo clone niklases/pypef
cd PyPEF
pip install -r requirements.txt
pip install hydra-core
pip install .
