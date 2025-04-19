#!/bin/bash
set -e 

# Only required for WSL(?):
# sudo apt-get install -y libxcb-cursor-dev

py_ver="$(python --version)"

if [[ $py_ver == *"3.12."* ]] || [[ $py_ver == *"3.11."* ]] || [[ $py_ver == *"3.10."* ]] || [[ $py_ver == *"3.9."* ]]; then
    echo "Identified Python version should be suitable for installing and running PyPEF..."
else
    echo >&2 "The identified Python version ($py_ver) does not match the required Python versions... you should activate/install a suitable version first, e.g. Python 3.12."; exit 1
fi
python -m pip install -U pypef pyside6

printf "
import sys\nimport os\n
sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n
from pypef.main import run_main\n\n
if __name__ == '__main__':
    run_main()
" > run.py

printf "#!/bin/bash\n
SCRIPT_DIR=\$( cd -- \"\$( dirname -- \"\${BASH_SOURCE[0]}\" )\" &> /dev/null && pwd )\n
python "\${SCRIPT_DIR}/gui/qt_window.py"\n
" > run_pypef_gui.sh

echo "Finished installation..."
echo "+++      Created file       +++"
echo "+++    run_pypef_gui.sh     +++"
echo "+++ for future GUI starting +++"

chmod a+x ./run_pypef_gui.sh
./run_pypef_gui.sh && exit 0
