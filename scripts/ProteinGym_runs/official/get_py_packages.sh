#!/bin/bash

wget https://github.com/niklases/PyPEF/archive/refs/heads/main.zip
unzip main.zip && rm main.zip
cd PyPEF-main/
pip install -r requirements.txt
pip install hydra-core
pip install .
cd ..
