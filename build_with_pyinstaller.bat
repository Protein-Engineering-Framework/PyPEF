@echo off
REM Build the PyPEF Qt GUI into a standalone folder with PyInstaller (Windows).
REM
REM The full dependency collection lives in the portable qt_window.spec, which
REM is the single source of truth shared with the Linux build. This script only
REM installs the dependencies and invokes that spec.
setlocal
cd /d "%~dp0"

python -m pip install --upgrade pip pyinstaller
python -m pip install -e .[gui]

python -m PyInstaller --noconfirm qt_window.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete. Run the GUI with:
echo   dist\qt_window\qt_window.exe
