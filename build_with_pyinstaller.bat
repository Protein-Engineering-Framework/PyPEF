REM Up to now pastes DLLs from local Python environment bin's to _internal...
REM alternative?: set PATH=%PATH%;%USERPROFILE%\miniconda3\envs\pypef\Library\bin\;
pip install -r requirements.txt
pip install -U pyinstaller pyside6
pip install -e .
set PATH=%PATH%;%USERPROFILE%\miniconda3\Scripts
pyinstaller^
  --console^
  --noconfirm^
  --collect-data pypef^
  --collect-all pypef^
  --collect-data torch^
  --collect-data biotite^
  --collect-all biotite^
  --collect-data torch_geometric^
  --collect-all torch_geometric^
  --hidden-import torch_geometric^
  pypef\gui\qt_window.py
REM  --add-binary=%USERPROFILE%\miniconda3\envs\pypef\Library\bin\onedal_thread.3.dll:.^
REM  --add-binary=%USERPROFILE%\miniconda3\envs\pypef\Library\bin\tbbbind.dll:.^
REM  --add-binary=%USERPROFILE%\miniconda3\envs\pypef\Library\bin\tbbbind_2_0.dll:.^
REM  --add-binary=%USERPROFILE%\miniconda3\envs\pypef\Library\bin\tbbbind_2_5.dll:.^
REM  --add-binary=%USERPROFILE%\miniconda3\envs\pypef\Library\bin\tbbmalloc.dll:.^
REM  --add-binary=%USERPROFILE%\miniconda3\envs\pypef\Library\bin\tbbmalloc_proxy.dll:.^