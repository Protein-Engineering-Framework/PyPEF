

import os
import sys
import time
import matplotlib.pyplot as plt

pypef_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '../..'
    )
)

sys.path.append(pypef_path)

avgfp_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '../../datasets/AVGFP'
    )
)

# Assuming that the Conda environment 'pypef' exists and installs 
# the PyPEF package into the pypef environment
os.system("conda run -n pypef python -m pip install -U pypef")
# Using'avGFP_shortened.csv' instead of 'avGFP.csv' takes much less computing time
cmd = f"conda run -n pypef python {os.path.join(pypef_path, 'pypef', 'main.py')} "\
      f"encode -i {os.path.join(avgfp_path, 'avGFP.csv')} "\
      f"-e dca -w {os.path.join(avgfp_path, 'P42212_F64L.fasta')} "\
      f"--params {os.path.join(avgfp_path, 'uref100_avgfp_jhmmer_119_plmc_42.6.params')} "\
      f"--threads XX"

print(f"Using up to {os.cpu_count()} cores/threads for parallel processing...")
all_run_times = []
for n_cores in range(1, os.cpu_count() + 1):
    run_time_1 = time.time()
    print(f"Running command:\n================\n{cmd.replace('XX', str(n_cores))}")
    os.system(cmd.replace('XX', str(n_cores)))
    run_time_2 = time.time()
    all_run_times.append(run_time_2 - run_time_1)
plt.plot(range(1, os.cpu_count() + 1), all_run_times, 'o--')
plt.grid()
plt.xlabel('# Cores/Threads')
plt.ylabel('Runtime (s)')
plt.savefig(os.path.join(os.path.dirname(__file__), 'runtimes.png'), dpi=300)
