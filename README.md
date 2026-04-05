# GPU-Accelerated Shape Recognition Using Image Processing and Fourier Descriptors

The goal of this project is to build a GPU-accelerated pipeline that classifies simple geometric shapes (circles, squares, and triangles) from an input image. The pipeline follows the algorithm described in [1]: grayscale conversion, Gaussian blur, Sobel edge detection, contour extraction, 1D FFT to produce Fourier Descriptors, and a classifier that identifies the shape from those descriptors. A prior implementation of a similar pipeline was done by the CREPE project [2] for chess piece recognition.

**References**

[1] https://www.researchgate.net/publication/263951637_Fast_Generalized_Fourier_Descriptor_for_object_recognition_of_image_using_CUDA

[2] https://github.com/Kawaboongawa/CREPE

---

# Stage 1 – GPU Image Pre-Processing Pipeline

Implements color conversion, noise reduction, edge detection, and binarization. Stages 1–3 use NPP; Stage 4 is a custom CUDA kernel.

## Pipeline

```
Input image (PPM)
    │
    ▼
[Stage 1 – NPP]   RGBA → 8-bit Grayscale       → out_1_gray.pgm
    │
    ▼
[Stage 2 – NPP]   Gaussian Blur                → out_2_blurred.pgm
    │
    ▼
[Stage 3 – NPP]   Sobel Edge Detection         → out_3_edges.pgm
    │
    ▼
[Stage 4 – CUDA]  Threshold → Binary Edge Map  → out_4_binary.pgm
```

**Stage 3:** runs `nppiFilterSobelHorizBorder` and `nppiFilterSobelVertBorder` (3×3, 8u→16s, replicated border) to get Gx and Gy, then a CUDA kernel computes `clamp(sqrt(Gx²+Gy²), 0, 255)`.

**Stage 4:** CUDA kernel — pixels at or above `edge_thresh` become 255, all others 0.

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

## Validation

The `validate` binary checks all five output PGMs after each `make test` run (18 checks total).

| Check | Expectation |
|---|---|
| Dimensions | All five outputs match input (512×512) |
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

```bash
./validate [width height]   # default: 512 512
```

Exit code 0 = all pass, 1 = one or more failures.

## File Structure

| File | Description |
|---|---|
| `main.cu` | Pipeline entry point |
| `npp_stages.cu` | Stages 1–3: grayscale, blur, Sobel via NPP |
| `sobel_threshold.cu` | Stage 4: binary threshold kernel |
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


# Stage 2 – GPU-Accelerated Shape Recognition

## Connected Component Labeling

Labels connected groups of edge pixels in the binary image so each shape boundary becomes a separately identified component. Small components (noise) are discarded.

```
[Stage 4 – CUDA]  Binary Edge Map  → out_4_binary.pgm
    │
    ▼
[Stage 5 – CUDA]  Connected Component Labeling  → out_5_labels.pgm
```

**Implementation:** iterative label propagation with ping-pong device buffers. Each foreground pixel is initialized to its linear index. On each pass, every pixel takes the minimum label of its 8-connected foreground neighbors. Passes repeat until no label changes. After convergence the label map is downloaded, components are assigned sequential IDs on the CPU, and any component smaller than `CCL_MIN_AREA` pixels is discarded as noise.

The label visualization (`out_5_labels.pgm`) maps each component ID to a distinct gray level (`id × 50`, clamped to 255) for visual inspection.

| Constant | Default | Description |
|---|---|---|
| `CCL_MIN_AREA` | 50 | Minimum component size in pixels |
