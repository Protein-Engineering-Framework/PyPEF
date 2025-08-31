#!/bin/bash
set -e
pip install pyinstaller
pip install -e .[gui]
pyinstaller \
  --console \
  --noconfirm \
  --paths "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
  --collect-data pypef \
  --collect-all pypef \
  --collect-data torch \
  --collect-data biotite \
  --collect-all biotite \
  --collect-data torch_geometric \
  --collect-all torch_geometric \
  --hidden-import torch_geometric \
  --hidden-import docopt \
  --exclude-module PyQt5 \
  pypef/gui/qt_window.py
