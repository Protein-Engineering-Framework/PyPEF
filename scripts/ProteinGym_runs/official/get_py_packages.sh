#!/bin/bash

# using GitHub-stored code from main branch so that this script runs everywhere
wget https://github.com/niklases/PyPEF/archive/refs/heads/main.zip
unzip main.zip && rm main.zip
cd PyPEF-main/
pip install -r requirements.txt
pip install hydra-core
pip install .
cd ..
