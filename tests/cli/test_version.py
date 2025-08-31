import os.path
import subprocess
import pytest
import sys


def capture(command):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    return out, err, proc.returncode


@pytest.mark.main_script_specific
def test_main_script_pypef_version():
    pypef_main_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    if not pypef_main_path in sys.path:
        sys.path.insert(0, pypef_main_path)
        # Will seemingly still fail if Pythonpath has not been ex-
        # ported in running terminal before calling this function
    from pypef import __version__
    command = [
        "python", 
        pypef_main_path + os.path.sep + "pypef" + os.path.sep + "main.py", 
        "--version"
    ]
    out, _err, exitcode = capture(command)
    assert str(out).split('\'')[1].split('\\')[0] == __version__
    assert exitcode == 0


@pytest.mark.pip_specific
def test_pip_pypef_version():
    from pypef import __version__
    command = [
        "pypef", 
        "--version"
    ]
    out, _err, exitcode = capture(command)
    assert str(out).split('\'')[1].split('\\')[0] == __version__
    assert exitcode == 0
