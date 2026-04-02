# GPU Image Pre-Processing Pipeline

A CUDA-accelerated image pre-processing pipeline for geometric shape classification.
Uses **NPP** for grayscale conversion and Gaussian blur, and **cuFFT** for
frequency-domain edge detection.

## Pipeline Stages

```
Input image (PPM)
    │
    ▼
[Stage 1 – NPP]   RGBA → 8-bit Grayscale      → out_1_gray.pgm
    │
    ▼
[Stage 2 – NPP]   Gaussian Blur               → out_2_blurred.pgm
    │
    ▼
[Stage 3 – cuFFT] Frequency-domain Edge Detect → out_3_edges.pgm
```

**Stage 3 detail:** the blurred image is forward-transformed (R2C), a high-pass
filter zeroes all frequency coefficients below `edge_threshold` (fraction of max
frequency distance from DC), and an inverse transform (C2R) produces the edge
magnitude image. Low-frequency regions (flat areas) are suppressed; high-frequency
content (shape boundaries) is preserved.

## Requirements

| Requirement | Version |
|---|---|
| CUDA Toolkit | 12.6 (tested) |
| GPU architecture | sm_86 (change `-arch` in Makefile as needed) |
| NPP libraries | `nppig`, `nppif`, `nppic`, `nppc` |
| cuFFT | included with CUDA Toolkit |
| C++ standard | C++17 |
| Compiler | `g++`, `nvcc` |

## Build

```bash
make          # builds pipeline, gen_test_image, validate
make clean    # removes all build artifacts and output PGMs
```

## Run

```bash
./pipeline <input.ppm> [blur_radius] [edge_threshold]
```

| Parameter | Default | Range | Description |
|---|---|---|---|
| `blur_radius` | 3 | 1–5 | Gaussian kernel half-size (3→7×7, 2→5×5, 1→3×3) |
| `edge_threshold` | 0.15 | 0.01–0.99 | Fraction of max frequency distance below which coefficients are zeroed |

Example:
```bash
./pipeline test_shapes.ppm 3 0.15
```

## Test

Generates a synthetic 512×512 test image containing a circle, rectangle, and
triangle, runs the pipeline twice with different parameters, then validates the
output files.

```bash
make test
```

**Pass 1:** `blur=3, threshold=0.15` — default settings
**Pass 2:** `blur=5, threshold=0.10` — aggressive smoothing and wider edge band

Output files (overwritten by Pass 2):

| File | Description |
|---|---|
| `out_1_gray.pgm` | Grayscale result |
| `out_2_blurred.pgm` | Gaussian-blurred result |
| `out_3_edges.pgm` | Edge magnitude image |

## Validation

The `validate` binary checks the three output PGMs for correctness after each
`make test` run. It performs 10 checks:

| Check | Expectation |
|---|---|
| Dimensions | All three outputs match input (512×512) |
| Gray mean | In range [10, 245] — conversion did not saturate |
| Gray stddev | > 5 — image has actual content |
| Gray max | > 100 — has bright pixels |
| Gray min | < 200 — shapes are darker than background |
| Blur stddev | ≤ gray stddev — smoothing reduces variance |
| Blur mean | Within 20 DN of gray — brightness preserved |
| Blur max | > 0 — image not wiped |
| Edge mean | < gray mean — high-pass removed DC component |
| Edge max | > 50 — edges were detected |
| Edge stddev | > 20 — edge map has contrast between regions |

Run manually:
```bash
./validate [width height]   # default: 512 512
```
Exit code 0 = all pass, 1 = one or more failures.

## CPU Reference (no GPU required)

A CPU-only pipeline for timing comparison and validation on machines without a GPU.
Uses a separable box blur and a 3×3 Laplacian edge detector.

```bash
make cpu_reference
make test_cpu
```

Outputs: `cpu_1_gray.pgm`, `cpu_2_blurred.pgm`, `cpu_3_edges.pgm`

Note: the CPU edge output looks different from the GPU output by design — the
Laplacian is a spatial second-derivative operator (sparse thin edges) while the
cuFFT high-pass filter retains all high-frequency content above the threshold.

## Benchmark

Times both pipelines and saves two charts. Requires `matplotlib` and `numpy`.

```bash
pip install matplotlib numpy
make benchmark
```

**`benchmark_configs.png`** — grouped bar chart comparing CPU vs GPU across four
parameter configurations at 512×512.

**`benchmark_scaling.png`** — line chart showing CPU and GPU wall-clock time as
image size grows from 512×512 to 4096×4096 (fixed blur=3, threshold=0.15). The
GPU carries a constant ~190 ms CUDA context initialization cost per process; GPU
compute time alone is faster at all sizes, and wall-clock time converges around
4096×4096.

## File Structure

```
.
├── main.cu             # pipeline entry point
├── npp_stages.cu       # stage 1 (RGBA→gray) and stage 2 (blur) via NPP
├── cufft_edge.cu       # stage 3 (edge detection) via cuFFT
├── image_io.cpp        # PPM load, PGM save
├── pipeline.h          # shared types and declarations
├── validate.cpp        # output validation
├── cpu_reference.cpp   # CPU reference pipeline
├── gen_test_image.cpp  # synthetic test image generator (accepts size arg)
├── benchmark.py        # CPU vs GPU timing + chart generation
├── stb_image.h         # minimal P6 PPM loader stub
├── stb_image_write.h   # minimal PPM writer stub
└── Makefile
```

## Makefile Targets

| Target | Description |
| --- | --- |
| `make` / `make all` | Build `pipeline`, `gen_test_image`, `validate` |
| `make test` | Generate test image, run both pipeline passes, validate |
| `make benchmark` | Build all binaries and run timing script |
| `make cpu_reference` | Build CPU-only reference binary |
| `make test_cpu` | Run CPU reference pipeline on test image |
| `make clean` | Remove all binaries, object files, output PGMs, and charts |
