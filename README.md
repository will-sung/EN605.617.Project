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
    │
    ▼
[Step 7 – cuFFT] Fourier Descriptors + Classify    → stdout
```

### Step 1 — Grayscale (`out_1_gray.pgm`)

Uses NPP's `nppiRGBToGray_8u_C3C1R` to convert the 3-channel RGB input to a single 8-bit luminance channel using the standard weighted formula (0.299R + 0.587G + 0.114B). Color information is discarded because subsequent gradient operators only need intensity.

**Output image:** a single-channel grayscale version of the input. Brighter pixels correspond to lighter-colored regions in the original.

---

### Step 2 — Gaussian Blur (`out_2_blurred.pgm`)

Uses NPP's `nppiFilterGauss_8u_C1R` with a `(2r+1) × (2r+1)` kernel (default r=3, giving a 7×7 kernel). Blurring reduces high frequency pixel noise before the gradient computation, which prevents spurious single pixel responses from appearing as edges.

**Output image:** a smoothed version of the grayscale image. Fine texture and noise are suppressed, while large scale boundaries are preserved but slightly softened.

---

### Step 3 — Sobel Edge Detection (`out_3_edges.pgm`)

Runs `nppiFilterSobelHorizBorder` and `nppiFilterSobelVertBorder` (3×3 kernel, 8u→16s, replicated border) to compute the horizontal gradient Gx and vertical gradient Gy. A CUDA kernel then computes the gradient magnitude per pixel: `clamp(sqrt(Gx² + Gy²), 0, 255)` and stores it as 8-bit.

**Output image:** a gradient magnitude map. Bright pixels mark locations of strong intensity change (edges). Flat, uniform regions appear dark. The brighter a pixel, the stronger the edge response at that location.

---

### Step 4 — Binary Threshold (`out_4_binary.pgm`)

A CUDA kernel compares each pixel in the Sobel magnitude image to `edge_thresh`. Pixels at or above the threshold are set to 255 (white); all others are set to 0 (black). This isolates only the strongest edges and discards weak gradient noise.

**Output image:** a strictly binary image. White pixels are candidate edge pixels; everything else is black. The density of white pixels is typically 0.5–15% of the total image area.

---

### Step 5 — Connected Component Labeling (`out_5_labels.pgm`)

Each foreground pixel is initialized to its linear pixel index as a unique label. An iterative GPU kernel then propagates labels: on every pass, each pixel adopts the minimum label among its 8 connected foreground neighbors. Two device buffers alternate (ping-pong) each iteration to avoid read/write races. The loop runs until a device side flag reports no label changed. After convergence, a CPU pass assigns compact sequential IDs and discards components smaller than `CCL_MIN_AREA` pixels (default 50), removing noise specks.

**Output image:** each surviving connected edge region is drawn in a distinct gray level (`component_id × 50`, clamped to 255). Background is black (0). This makes it easy to visually verify how many separate shapes were found, one distinct gray shade per detected shape outline.

---

### Step 6 — Contour Tracing (`out_6_contours.pgm`)

For each labeled component, the CPU locates the topmost leftmost pixel and runs a Moore-neighbor boundary trace. Starting from a backtrack direction of West (direction 4), the algorithm scans the 8 neighbors clockwise, stepping to the first foreground pixel found. The backtrack direction is updated to point from the new pixel back toward the previous background pixel. Tracing ends when the path returns to the starting pixel.

**Output image:** only the traced boundary pixels are drawn in white on a black background. Unlike the binary edge map (which may be several pixels thick), the contour is a single-pixel-wide outline around each component, the exact sequence of points used as input to the Fourier Descriptor step.

---

### Step 7 — Fourier Descriptors + Classification (stdout)

Each contour is treated as a 1D complex signal `z[n] = x[n] + j*y[n]`, where x and y are the coordinates of the n-th boundary point. A forward 1D cuFFT (C2C) transforms the sequence into the frequency domain. The spectrum is normalized by `|Z[1]|` to achieve scale and rotation invariance.

The normalized magnitude of each harmonic reflects the shape's rotational symmetry:
- **k=2 (or N-2):** large for a triangle — the 3-fold symmetry produces a strong 2nd harmonic
- **k=3 (or N-3):** large for a rectangle — the 2-fold/4-fold symmetry produces a strong 3rd harmonic
- **all k>1 small:** characteristic of a circle, which has no discrete angular symmetry

Both positive and mirror (negative) frequencies are checked (`max(|Z[k]|, |Z[N-k]|)`) to handle contours traced in either direction.

**Classification rule:**
- `d2 > 0.08 && d2 > d3` → **triangle**
- `d3 > 0.08 && d3 > d2` → **rectangle**
- otherwise → **circle**

Results are printed to stdout with the label index, contour length, d2/d3 values, and the assigned shape name.

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

Generates all 7 test images, runs the pipeline twice on `test_all.ppm` (circle + rectangle + triangle), then validates all outputs.

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
| `fourier_desc.cu` | Step 7: cuFFT Fourier Descriptors + classifier |
| `image_io.cpp` | PPM load, PGM save |
| `pipeline.h` | Shared types and declarations |
| `validate.cpp` | Output validation |
| `cpu_reference.cpp` | CPU reference pipeline |
| `gen_test_images.cpp` | Generates 7 test PPMs (each shape alone, pairs, all three) |
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
| `make` / `make all` | Build `pipeline`, `gen_test_images`, `validate` |
| `make test` | Generate test image, run both passes, validate |
| `make test_images` | Build `gen_test_images` and generate all 7 individual test PPMs |
| `make benchmark` | Build all binaries and run timing script |
| `make cpu_reference` | Build CPU-only reference binary |
| `make test_cpu` | Run CPU reference pipeline on test image |
| `make clean` | Remove all binaries, object files, output PGMs, and charts |
