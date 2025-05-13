
import shutil
import subprocess
import torch

from pypef.settings import USE_RAY


class ConditionalDecorator(object):
    def __init__(self, decorator, condition):
        self.decorator = decorator
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            # Return the function unchanged, not decorated
            return func
        return self.decorator(func)


if USE_RAY:
    import ray
    ray_conditional_decorator = ConditionalDecorator(ray.remote, USE_RAY)
else:
    ray_conditional_decorator = ConditionalDecorator(None, False)


def get_device():
    return (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )


def get_vram(verbose: bool = True):
    if not torch.cuda.is_available():
        if verbose:
            print("No CUDA/GPU device available for VRAM checking.")
        return "No CUDA/GPU device available for VRAM checking."
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    if verbose:
        print(f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']')
    return f"Total VRAM: {total:.2f}GB"


def get_torch_version():
    return torch.__version__


def get_gpu_info():
    if shutil.which("nvidia-smi"):
        output = subprocess.check_output("nvidia-smi").decode("utf-8").split("\n")
        output = "GPU: " + output[8].split('|')[1][6:-6].strip()
    else:
        output = "No nvidia-smi (and hence GPU) found."
    return output
