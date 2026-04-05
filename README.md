# GPU-Accelerated Shape Recognition Using Image Processing and Fourier Descriptors

The goal of this project is to build a GPU-accelerated image processing pipeline that can classify simple geometric shapes (circles, squares, and triangles) from an input image. The algorithm is from a research gate publication [1] that developed the pipeline that consist of the following stages: grayscale conversion, Gaussian blur for noise reduction, Sobel edge detection to isolate shape boundaries, contour extraction to trace the boundary as an ordered set of points, a 1D FFT on the contour signal to produce Fourier Descriptors, and finally a classifier that interprets those descriptors to identify the shape. A previous implementation of this pipeline was demonstrated by Chess Recognition Enhanced and Parallelized Engine (CREPE) Project [2] on chess pieces. 

References
________________________________________
1.	https://www.researchgate.net/publication/263951637_Fast_Generalized_Fourier_Descriptor_for_object_recognition_of_image_using_CUDA
2.	https://github.com/Kawaboongawa/CREPE

# Stage 1:
# GPU Image Pre-Processing Pipeline

A CUDA-accelerated image preprocessing pipeline for geometric shape classification.
Uses **NPP** for all three stages: grayscale conversion, Gaussian blur, and Sobel
edge detection.

## Pipeline Stages

```
Input image (PPM)
    │
    ▼
[Stage 1 – NPP]  RGBA → 8-bit Grayscale  → out_1_gray.pgm
    │
    ▼
[Stage 2 – NPP]  Gaussian Blur           → out_2_blurred.pgm
    │
    ▼
[Stage 3 – NPP]  Sobel Edge Detection    → out_3_edges.pgm
```

**Stage 3 detail:** the blurred image is processed by two NPP 3×3 Sobel passes
(`nppiFilterSobelHorizBorder` and `nppiFilterSobelVertBorder`, 8u→16s, replicated
border) producing signed horizontal and vertical gradients Gx and Gy.  A small
CUDA kernel computes the per-pixel gradient magnitude
`clamp(sqrt(Gx²+Gy²), 0, 255)` into an 8-bit output image.  Flat regions produce
near-zero response; shape boundaries produce bright ridges suitable for contour
tracing in Stage 2.

## Requirements

| Requirement | Version |
|---|---|
| CUDA Toolkit | 12.6 (tested) |
| GPU architecture | sm_86 (change `-arch` in Makefile as needed) |
| NPP libraries | `nppig`, `nppif`, `nppic`, `nppc` |
| C++ standard | C++17 |
| Compiler | `g++`, `nvcc` |

## Build

```bash
make          # builds pipeline, gen_test_image, validate
make clean    # removes all build artifacts and output PGMs
```

## Run

```bash
./pipeline <input.ppm> [blur_radius]
```

| Parameter | Default | Range | Description |
|---|---|---|---|
| `blur_radius` | 3 | 1–5 | Gaussian kernel half-size (3→7×7, 2→5×5, 1→3×3) |

Example:
```bash
./pipeline test_shapes.ppm 3
```

## Test

Generates a synthetic 512×512 test image containing a circle, rectangle, and
triangle, runs the pipeline twice with different parameters, then validates the
output files.

```bash
make test
```

**Pass 1:** `blur=3` — default settings
**Pass 2:** `blur=5` — aggressive smoothing

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
| Edge mean | < gray mean — flat regions near zero in Sobel output |
| Edge max | > 50 — edges were detected |
| Edge stddev | > 20 — edge map has contrast between regions |

Run manually:
```bash
./validate [width height]   # default: 512 512
```
Exit code 0 = all pass, 1 = one or more failures.

## CPU Reference (no GPU required)

A CPU only pipeline for timing comparison and validation on machines without a GPU.
Uses a separable box blur and a 3×3 Laplacian edge detector.

```bash
make cpu_reference
make test_cpu
```

## Benchmark

Times both pipelines and saves two charts. Requires `matplotlib` and `numpy`.

```bash
pip install matplotlib numpy
make benchmark
```

**`benchmark_configs.png`** — grouped bar chart comparing CPU vs GPU across four
parameter configurations at 512×512.

**`benchmark_scaling.png`** — line chart showing CPU and GPU wall-clock time as
image size grows from 512×512 to 4096×4096 (fixed blur=3). The GPU carries a
constant ~190 ms CUDA context initialization cost per process; GPU compute time
alone is faster at all sizes, and wall-clock time converges around 4096×4096.

## Makefile Targets

| Target | Description |
| --- | --- |
| `make` / `make all` | Build `pipeline`, `gen_test_image`, `validate` |
| `make test` | Generate test image, run both pipeline passes, validate |
| `make benchmark` | Build all binaries and run timing script |
| `make cpu_reference` | Build CPU-only reference binary |
| `make test_cpu` | Run CPU reference pipeline on test image |
| `make clean` | Remove all binaries, object files, output PGMs, and charts |


# Stage 2:
# GPU-Accelerated Shape Recognition
