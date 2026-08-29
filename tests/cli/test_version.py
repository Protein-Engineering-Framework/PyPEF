import os.path
import subprocess
import sys
import shutil
import pytest


def capture(command):
    # Ensure subprocess inherits current environment and resolves sys.executable directory
    env = os.environ.copy()
    venv_bin = os.path.dirname(sys.executable)
    env["PATH"] = f"{venv_bin}{os.path.pathsep}{env.get('PATH', '')}"

    # If the command is invoking "python", point explicitly to sys.executable
    if command and command[0] == "python":
        command[0] = sys.executable

    # If invoking a binary like "pypef", resolve its path within the current environment
    elif command and not os.path.isabs(command[0]):
        resolved = shutil.which(command[0], path=env["PATH"])
        if resolved:
            command[0] = resolved

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Decodes stdout/stderr to str automatically
        env=env
    )
    out, err = proc.communicate()
    return out, err, proc.returncode


@pytest.mark.main_script_specific
def test_main_script_pypef_version():
    pypef_main_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    if pypef_main_path not in sys.path:
        sys.path.insert(0, pypef_main_path)
        
    from pypef import __version__
    
    script_path = os.path.join(pypef_main_path, "pypef", "main.py")
    command = ["python", script_path, "--version"]
    
    out, err, exitcode = capture(command)
    assert exitcode == 0, f"Process failed with error:\n{err}"
    assert __version__ in out


@pytest.mark.pip_specific
def test_pip_pypef_version():
    from pypef import __version__
    
    command = ["pypef", "--version"]
    out, err, exitcode = capture(command)
    
    assert exitcode == 0, f"Process failed with error:\n{err}"
    assert __version__ in out


if __name__ == "__main__":
    test_main_script_pypef_version()