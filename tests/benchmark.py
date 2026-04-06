#!/usr/bin/env python3
# benchmark.py – CPU vs GPU wall-clock timing
# outputs: benchmark_configs.png
# usage:   python3 tests/benchmark.py   or   make benchmark

import subprocess
import sys
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)

RUNS = 3
IMG  = "tests/images/test_all.png"


def time_cmd(cmd):
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        r  = subprocess.run(cmd, capture_output=True)
        t1 = time.perf_counter()
        if r.returncode != 0:
            print(f"  [error] {' '.join(cmd)}\n{r.stderr.decode().strip()}")
            return None
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[RUNS // 2]


def chart_configs():
    configs = [
        ("blur=1\nt=40", "1", "40"),
        ("blur=3\nt=40", "3", "40"),
        ("blur=5\nt=35", "5", "35"),
        ("blur=5\nt=45", "5", "45"),
    ]

    labels, cpu_times, gpu_times = [], [], []
    for label, blur, thresh in configs:
        print(f"configs | {label.replace(chr(10),' ')} ...", flush=True)
        cpu = time_cmd(["./cpu_reference", IMG, blur])
        gpu = time_cmd(["./pipeline_app",  IMG, blur, thresh])
        if cpu is None or gpu is None:
            continue
        labels.append(label)
        cpu_times.append(cpu)
        gpu_times.append(gpu)
        print(f"  cpu {cpu:7.1f} ms   gpu {gpu:7.1f} ms   speedup {cpu/gpu:.2f}x")

    if not labels:
        print("chart_configs: no data"); return

    x, width = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_c = ax.bar(x - width/2, cpu_times, width, label="CPU", color="#4878CF")
    bars_g = ax.bar(x + width/2, gpu_times, width, label="GPU", color="#6ACC65")

    for bar in (*bars_c, *bars_g):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    for i, (c, g) in enumerate(zip(cpu_times, gpu_times)):
        ax.text(x[i], max(c, g) * 1.06, f"{c/g:.1f}x",
                ha="center", fontsize=9, color="crimson", fontweight="bold")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Wall-clock time (ms)  [median of 3 runs]")
    ax.set_title("CPU vs GPU – Parameter Configurations (512×512)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("benchmark_configs.png", dpi=150)
    print("saved benchmark_configs.png\n")


if __name__ == "__main__":
    chart_configs()
