#!/bin/bash
pip install -r requirements.txt
pip install -U pyinstaller pyside6
pip install -e .
pyinstaller \
  --console \
  --noconfirm \
  --collect-data torch \
  --collect-data biotite \
  --collect-all biotite \
  --collect-data torch_geometric \
  --collect-all torch_geometric \
  --hidden-import torch_geometric \
  gui/PyPEFGUIQtWindow.py
  #  --add-data=X/:X/. \