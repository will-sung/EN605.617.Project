# GPU-Accelerated Shape Recognition Using Image Processing and Fourier Descriptors

The goal of this project is to build a GPU-accelerated pipeline that classifies simple geometric shapes (circles, squares, and triangles) from an input image. The pipeline follows the algorithm described in [1]: grayscale conversion, Gaussian blur, Sobel edge detection, contour extraction, 1D FFT to produce Fourier Descriptors, and a classifier that identifies the shape from those descriptors. A prior implementation of a similar pipeline was done by the CREPE project [2] for chess piece recognition.

**References**

[1] https://www.researchgate.net/publication/263951637_Fast_Generalized_Fourier_Descriptor_for_object_recognition_of_image_using_CUDA

[2] https://github.com/Kawaboongawa/CREPE

---

## Pipeline

```
Input image (PPM)
    │
    ▼
[Step 1 – NPP]   RGBA → 8-bit Grayscale            → out_1_gray.pgm
    │
    ▼
[Step 2 – NPP]   Gaussian Blur                     → out_2_blurred.pgm
    │
    ▼
[Step 3 – NPP]   Sobel Edge Detection              → out_3_edges.pgm
    │
    ▼
[Step 4 – CUDA]  Threshold → Binary Edge Map       → out_4_binary.pgm
    │
    ▼
[Step 5 – CUDA]  Connected Component Labeling      → out_5_labels.pgm
    │
    ▼
[Step 6 – CPU]   Contour Tracing                   → out_6_contours.pgm
```

**Step 3:** runs `nppiFilterSobelHorizBorder` and `nppiFilterSobelVertBorder` (3×3, 8u→16s, replicated border) to get Gx and Gy, then a CUDA kernel computes `clamp(sqrt(Gx²+Gy²), 0, 255)`.

**Step 4:** CUDA kernel — pixels at or above `edge_thresh` become 255, all others 0.

**Step 5:** iterative label propagation with ping-pong device buffers. Each foreground pixel is initialized to its linear index. On each pass, every pixel takes the minimum label of its 8-connected foreground neighbors until no label changes. Components smaller than `CCL_MIN_AREA` pixels are discarded. The label visualization maps each component ID to a distinct gray level (`id × 50`, clamped to 255).

**Step 6:** Moore-neighbor boundary tracing on CPU. For each component, starts at the topmost-leftmost pixel and scans the 8-neighborhood clockwise from the backtrack direction until the path returns to the start.

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
make        # builds pipeline, gen_test_image, validate
make clean  # removes all build artifacts and output PGMs
```

## Run

```bash
./pipeline <input.ppm> [blur_radius] [edge_thresh]
```

| Parameter | Default | Range | Description |
|---|---|---|---|
| `blur_radius` | 3 | 1–5 | Gaussian kernel half-size (3→7×7, 2→5×5, 1→3×3) |
| `edge_thresh` | 40 | 1–254 | Sobel magnitude cutoff for binary edge map |

Example:
```bash
./pipeline test_shapes.ppm 3 40
```

## Test

Generates a synthetic 512×512 test image with a circle, rectangle, and triangle, runs the pipeline twice, then validates all outputs.

```bash
make test
```

**Pass 1:** `blur=3, thresh=40`
**Pass 2:** `blur=5, thresh=35`

Output files (overwritten by Pass 2):

| File | Description |
|---|---|
| `out_1_gray.pgm` | Grayscale result |
| `out_2_blurred.pgm` | Gaussian-blurred result |
| `out_3_edges.pgm` | Sobel gradient magnitude |
| `out_4_binary.pgm` | Binary edge map (0 or 255) |
| `out_5_labels.pgm` | Connected component label visualization |
| `out_6_contours.pgm` | Traced boundary pixels (white on black) |

## Validation

The `validate` binary checks all six output PGMs after each `make test` run (22 checks total).

| Check | Expectation |
|---|---|
| Dimensions | All six outputs match input (512×512) |
| Gray mean | In range [10, 245] |
| Gray stddev | > 5 |
| Gray max | > 100 |
| Gray min | < 200 |
| Blur stddev | ≤ gray stddev |
| Blur mean | Within 20 of gray mean |
| Blur max | > 0 |
| Edge mean | < gray mean |
| Edge max | > 50 |
| Edge stddev | > 20 |
| Binary max | == 255 |
| Binary min | == 0 |
| Binary nonzero | Between 0.5% and 15% |
| Labels min | == 0 |
| Labels max | > 0 |
| Labels nonzero | > 0.5% |
| Labels nonzero | < 15% |
| Contours min | == 0 |
| Contours max | == 255 |
| Contours nonzero | > 0.5% |
| Contours nonzero | < 15% |

```bash
./validate [width height]   # default: 512 512
```

Exit code 0 = all pass, 1 = one or more failures.

## File Structure

| File | Description |
|---|---|
| `main.cu` | Pipeline entry point |
| `npp_stages.cu` | Steps 1–3: grayscale, blur, Sobel via NPP |
| `sobel_threshold.cu` | Step 4: binary threshold kernel |
| `ccl.cu` | Step 5: connected component labeling |
| `contour.cpp` | Step 6: Moore-neighbor boundary tracing |
| `image_io.cpp` | PPM load, PGM save |
| `pipeline.h` | Shared types and declarations |
| `validate.cpp` | Output validation |
| `cpu_reference.cpp` | CPU reference pipeline |
| `gen_test_image.cpp` | Synthetic test image generator |
| `benchmark.py` | CPU vs GPU timing and chart generation |

## CPU Reference

A CPU-only pipeline for timing comparison. Uses a separable box blur and a 3×3 Laplacian edge detector.

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

`benchmark_configs.png` — CPU vs GPU comparison across four parameter configurations at 512×512.

`benchmark_scaling.png` — CPU and GPU wall-clock time as image size scales from 512×512 to 4096×4096.

## Makefile Targets

| Target | Description |
|---|---|
| `make` / `make all` | Build `pipeline`, `gen_test_image`, `validate` |
| `make test` | Generate test image, run both passes, validate |
| `make benchmark` | Build all binaries and run timing script |
| `make cpu_reference` | Build CPU-only reference binary |
| `make test_cpu` | Run CPU reference pipeline on test image |
| `make clean` | Remove all binaries, object files, output PGMs, and charts |
