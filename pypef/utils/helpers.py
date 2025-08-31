# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

import shutil
import subprocess
import torch
import pynvml
from pynvml import NVMLError

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
        msg = "No CUDA/GPU device available for VRAM checking"
        if verbose:
            print(msg + ".")
        return msg, msg
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    total_str = f"Total VRAM: {total:.2f}GB"
    usage_str = f'VRAM: {total - free:.2f}/{total:.2f}GB  VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'
    if verbose:
        print(usage_str)
    return total_str, usage_str


def get_torch_version():
    return torch.__version__


def get_gpu_info_nvidia_smi():
    if shutil.which("nvidia-smi"):
        output = subprocess.check_output("nvidia-smi").decode("utf-8").split("\n")
        output = "GPU: " + output[8].split('|')[1][6:-6].strip()
    else:
        output = "No nvidia-smi (and hence GPU) found."
    return output


def get_nvidia_gpu_info_pynvml():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu = {
                "index": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "driver_version": driver_version.decode() if isinstance(driver_version, bytes) else driver_version,
                "memory_total_MB": mem_info.total // 1024**2,
                "memory_used_MB": mem_info.used // 1024**2,
                "memory_free_MB": mem_info.free // 1024**2,
                "temperature_C": temp,
                "gpu_utilization_percent": util.gpu,
                "memory_utilization_percent": util.memory
            }
            gpus.append(gpu)
        pynvml.nvmlShutdown()
        if not gpus:  # Will dict be partially filled with None/empty strings?
            print("No GPU information found.")
        gpu_0_name = gpus[0]["name"]
        gpu_0_driver = gpus[0]["driver_version"]
        return gpu_0_name, gpu_0_driver
    except NVMLError as e:
        return "NVMLError", e

