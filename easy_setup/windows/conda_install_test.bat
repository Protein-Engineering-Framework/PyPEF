REM Run me via mouse double-click or in CMD.
REM Installing Miniconda3 locally in folder, creating Python 3.10 environment using Miniconda3, 
REM installing PyPEF in conda environment and running low N test api_encoding_train_test.py file. 
REM Requires files 'download_install_miniconda.ps1' and 'download_pypef_test_files.ps1'.
powershell.exe -ExecutionPolicy Bypass -File "download_install_miniconda.ps1"
call .\Miniconda3\Scripts\activate.bat .\Miniconda3
call conda create -n pypef python=3.10 -y
call conda activate pypef
python -V
python -m ensurepip
python -m pip install -U pypef
pypef --version
@echo off
setlocal
:PROMPT
SET /P AREYOUSURE=Test PyPEF installation (Y/[N]) (downloads PyPEF example test set files and runs a Python test script, ~ 1 h run time)?
IF /I "%AREYOUSURE%" NEQ "Y" GOTO END
powershell.exe -ExecutionPolicy Bypass -File "download_pypef_test_files.ps1"
python .\api_encoding_train_test.py
echo Done!
cmd /k

:END
endlocal
echo Done!
cmd /k
