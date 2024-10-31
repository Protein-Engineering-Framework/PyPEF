@echo off
echo Installing PyPEF...

set "python_exe=python"
set "prefix="

for /F "delims=" %%i in ('powershell python --version') do set py_ver=%%i
if not "%py_ver:~0,6%"=="Python" (
    set "true=0"
) else (
    set "true=0"
    if  "%py_ver:~7,5%"=="3.12." (
        echo Found Python version 3.12.
        set "true=1"        
    )
    if  "%py_ver:~7,5%"=="3.11." (
        echo Found Python version 3.11.
        set "true=1"
    )
    if  "%py_ver:~7,5%"=="3.10." (
        echo Found Python version 3.10.
        set "true=1"
    )
    if  "%py_ver:~7,4%"=="3.9." (
        echo Found Python version 3.9.
        set "true=1"
    )
)

if "%true%"=="0" (
    echo Did not find any Python version. Python will be installed locally in the next step...
    set /P AREYOUSURE="Y"
) else ( 
    echo A suitable Python version was found, no local download and Python installtion should be necessary...
    set /P AREYOUSURE="Install and use local Python version (Y/[N]) (downloads Python installer and installs Python locally in the current working directory)? "
)

if /I "%AREYOUSURE%" NEQ "Y" if /I "%AREYOUSURE%" NEQ "y" goto NO_PYTHON

echo Installing Python...
powershell -Command "$ProgressPreference = 'SilentlyContinue';Invoke-WebRequest https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe -OutFile python-3.12.7-amd64.exe"

.\python-3.12.7-amd64.exe /quiet TargetDir="%~dp0Python3127" Include_pip=1 Include_test=0 AssociateFiles=1 PrependPath=0 CompileAll=1 InstallAllUsers=0

REM Not removing Python installer EXE as it can be used for easy uninstall/repair
REM del /Q python-3.12.7-amd64.exe
set "python_exe=.\Python3127\python.exe"
set "prefix=%%~dp0"


:NO_PYTHON

(
    echo:
    echo import sys
    echo import os
    echo:
    echo sys.path.append(os.path.dirname^(os.path.abspath^(__file__^)^)^)
    echo:
    echo from pypef.main import run_main
    echo:
    echo:
    echo if __name__ == '__main__':
    echo     run_main^(^)
) > run.py

powershell -Command "%python_exe% -m pip install -U pypef pyside6"

(
    echo @echo off
    echo:
    echo start /min cmd /c powershell -Command ^"%prefix%%python_exe% %%~dp0gui\qt_window.py^"
 ) > run_pypef_gui.bat

echo Finished installation...
echo +++      Created file       +++ 
echo +++    run_pypef_gui.bat    +++
echo +++ for future GUI starting +++
echo You can close this window now.

REM call .\run_pypef_gui.bat 
start cmd /c ".\run_pypef_gui.bat"
cmd /k
