import os
import subprocess

from pypef import __version__

pypef_main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def capture(command):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    return out, err, proc.returncode


def test_pypef_version():
    command = ["python", pypef_main_path + "/pypef/main.py", "--version"]
    out, err, exitcode = capture(command)
    assert str(out).split('\'')[1].split('\\')[0] == __version__
    assert exitcode == 0

