#!/usr/bin/env python3
# benchmark.py – CPU vs GPU wall-clock timing
# outputs: benchmark_configs.png  benchmark_scaling.png
# usage:   python3 benchmark.py   or   make benchmark

import subprocess
import sys
import time
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)

RUNS = 3


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
    subprocess.run(["./gen_test_image"], check=True, capture_output=True)

    configs = [
        ("blur=1\nt=0.20", "1", "0.20"),
        ("blur=3\nt=0.15", "3", "0.15"),
        ("blur=5\nt=0.10", "5", "0.10"),
        ("blur=5\nt=0.05", "5", "0.05"),
    ]

    labels, cpu_times, gpu_times = [], [], []
    for label, blur, thresh in configs:
        print(f"configs | {label.replace(chr(10),' ')} ...", flush=True)
        cpu = time_cmd(["./cpu_reference", "test_shapes.ppm", blur])
        gpu = time_cmd(["./pipeline",      "test_shapes.ppm", blur, thresh])
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


def chart_scaling():
    sizes = [512, 1024, 2048, 4096]
    blur, thresh = "3", "0.15"
    cpu_times, gpu_times, valid_sizes = [], [], []

    for sz in sizes:
        ppm = f"bench_{sz}.ppm"
        print(f"scaling | {sz}x{sz} ...", flush=True)
        subprocess.run(["./gen_test_image", str(sz), ppm],
                       check=True, capture_output=True)

        cpu = time_cmd(["./cpu_reference", ppm, blur])
        gpu = time_cmd(["./pipeline",      ppm, blur, thresh])

        try: os.remove(ppm)
        except OSError: pass

        if cpu is None or gpu is None:
            continue
        valid_sizes.append(sz)
        cpu_times.append(cpu)
        gpu_times.append(gpu)
        print(f"  cpu {cpu:7.1f} ms   gpu {gpu:7.1f} ms   speedup {cpu/gpu:.2f}x")

    if not valid_sizes:
        print("chart_scaling: no data"); return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(valid_sizes, cpu_times, "o-", color="#4878CF",
            label="CPU", linewidth=2, markersize=6)
    ax.plot(valid_sizes, gpu_times, "s-", color="#6ACC65",
            label="GPU", linewidth=2, markersize=6)

    for sz, c, g in zip(valid_sizes, cpu_times, gpu_times):
        ax.annotate(f"{c/g:.1f}x", xy=(sz, g),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8,
                    color="crimson", fontweight="bold")

    ax.set_xlabel("Image size (square)")
    ax.set_ylabel("Wall-clock time (ms)  [median of 3 runs]")
    ax.set_title("CPU vs GPU – Scaling by Image Size (blur=3, t=0.15)")
    ax.set_xticks(valid_sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in valid_sizes])
    ax.legend(); ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("benchmark_scaling.png", dpi=150)
    print("saved benchmark_scaling.png\n")


if __name__ == "__main__":
    chart_configs()
    chart_scaling()
