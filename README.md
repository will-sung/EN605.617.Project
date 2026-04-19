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

`nppiRGBToGray_8u_C3C1R` — standard luminance weights (0.299R + 0.587G + 0.114B).

---

### Step 2 — Gaussian Blur (`out_2_blurred.pgm`)

`nppiFilterGauss_8u_C1R` with a `(2r+1) × (2r+1)` kernel (default r=3 → 7×7). Smooths noise before gradient computation.

---

### Step 3 — Sobel Edge Detection (`out_3_edges.pgm`)

`nppiFilterSobelHorizBorder` + `nppiFilterSobelVertBorder` (3×3, 8u→16s, replicated border). A CUDA kernel computes `clamp(sqrt(Gx²+Gy²), 0, 255)` and writes it back as 8-bit.

---

### Step 4 — Binary Threshold (`out_4_binary.pgm`)

CUDA kernel: pixels ≥ `edge_thresh` → 255, rest → 0.

---

### Step 5 — Connected Component Labeling (`out_5_labels.pgm`)

Each foreground pixel starts with its linear index as a label. An iterative kernel propagates the minimum label across 8-connected neighbors using ping-pong buffers until nothing changes. A CPU pass then compacts the IDs and drops components smaller than `CCL_MIN_AREA` (default 50 px).

Output colors each component `id × 50` (clamped to 255) so separate shapes are visually distinct.

---

### Step 6 — Contour Tracing (`out_6_contours.pgm`)

Moore-neighbor boundary trace on each component. Starts at the topmost-leftmost pixel, backtrack direction West, scans neighbors clockwise until the path closes. Result is a single-pixel-wide contour used as input to the FFT step.

---

### Step 7 — Fourier Descriptors + Classification (stdout)

Contour points are packed into a complex signal `z[n] = x[n] + j*y[n]` and transformed with cuFFT (1D C2C). The spectrum is normalized by `|Z[1]|` for scale/rotation invariance. Both `|Z[k]|` and `|Z[N-k]|` are checked to handle either trace direction.

Key descriptors:
- `d2 = max(|Z[2]|, |Z[N-2]|)` — strong for triangles
- `d3 = max(|Z[3]|, |Z[N-3]|)` — strong for rectangles

**Classification:**
- `d2 > 0.08 && d2 > d3` → **triangle**
- `d3 > 0.08 && d3 > d2` → **rectangle**
- otherwise → **circle**

Results printed to stdout: label index, contour length, d2/d3, shape name.

## GUI

A tkinter-based front-end for running the pipeline interactively.

```bash
pip install Pillow
make gui
```

`make gui` builds `pipeline_app` if needed, then launches the GUI. The left panel has controls for selecting an input image and adjusting blur radius and edge threshold. Clicking **Run Pipeline** executes the pipeline in a background thread and displays the result in the canvas. Use the stage radio buttons to flip between the six intermediate output images. Shape classification results appear in the scrollable text box at the bottom of the panel.

Requires Python 3 with `tkinter` (standard library) and `Pillow`.

---

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
./pipeline_app <input> [blur_radius] [edge_thresh]
```

Accepts any format supported by stb_image: **PNG, JPG, BMP, TGA, GIF, PPM/PGM**.

| Parameter | Default | Range | Description |
|---|---|---|---|
| `blur_radius` | 3 | 1–5 | Gaussian kernel half-size (3→7×7, 2→5×5, 1→3×3) |
| `edge_thresh` | 40 | 1–254 | Sobel magnitude cutoff for binary edge map |

Example:
```bash
./pipeline_app tests/images/test_all.png 3 40
```

## Test

Generates all 7 test images, runs the pipeline twice on `test_all.png` (circle + rectangle + triangle), then validates all outputs.

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

```
pipeline/               source for the GPU pipeline application
  main.cu               entry point
  npp_stages.cu         Steps 1–3: grayscale, blur, Sobel via NPP
  sobel_threshold.cu    Step 4: binary threshold kernel
  ccl.cu                Step 5: connected component labeling
  contour.cpp           Step 6: Moore-neighbor boundary tracing
  fourier_desc.cu       Step 7: cuFFT Fourier Descriptors + classifier
  image_io.cpp          image load (stb_image) and PGM save
  pipeline.h            shared types and declarations
  stb_image.h           stb_image v2.30 (JPEG/PNG/BMP/TGA/GIF/PNM)

gui/
  gui.py                tkinter GUI front-end

tests/                  test and verification tools
  validate.cpp          checks all six output PGMs
  gen_test_images.cpp   generates 7 test PNGs
  cpu_reference.cpp     CPU-only reference pipeline for timing comparison
  benchmark.py          CPU vs GPU timing charts
  stb_image_write.h     stb_image_write v1.16
  images/               generated test PNGs (created by make test_images)
```

Binaries are built into the project root: `pipeline_app`, `gen_test_images`, `validate`, `cpu_reference`.

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
| `make` / `make all` | Build `pipeline_app`, `gen_test_images`, `validate` |
| `make test` | Generate test image, run both passes, validate |
| `make test_images` | Build `gen_test_images` and generate all 7 individual test PPMs |
| `make gui` | Build `pipeline_app` and launch the GUI |
| `make benchmark` | Build all binaries and run timing script |
| `make cpu_reference` | Build CPU-only reference binary |
| `make test_cpu` | Run CPU reference pipeline on test image |
| `make clean` | Remove all binaries, object files, output PGMs, and charts |
