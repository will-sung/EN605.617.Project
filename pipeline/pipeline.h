#pragma once
// pipeline.h – shared types and function declarations

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

static constexpr int MAX_DIM      = 4096;
static constexpr int BLUR_RADIUS  = 3;    // default -> 7x7 kernel
static constexpr int EDGE_THRESH  = 40;   // default Sobel magnitude threshold
static constexpr int CCL_MIN_AREA = 50;   // minimum component size (pixels)

struct GrayImage {
    int      width  = 0;
    int      height = 0;
    uint8_t* data   = nullptr;  // row-major, host memory
};

uint8_t* load_image(const std::string& path, int& out_w, int& out_h);

bool save_pgm(const std::string& path, const GrayImage& img);

// stage 1 (NPP): RGBA -> 8-bit grayscale
GrayImage npp_rgba_to_gray(const uint8_t* h_rgba, int width, int height);

// stage 2 (NPP): gaussian blur
GrayImage npp_gaussian_blur(const GrayImage& src, int radius);

// stage 3 (NPP + CUDA): Sobel edge detection -> gradient magnitude image
GrayImage npp_sobel_edges(const GrayImage& src);

// stage 4 (CUDA): threshold gradient magnitude -> binary edge map (0 or 255)
GrayImage threshold_edges(const GrayImage& src, int thresh_val);

// result of connected component labeling
struct CCLResult {
    int       width          = 0;
    int       height         = 0;
    uint32_t* labels         = nullptr;  // host memory, 0 = background
    int       num_components = 0;
};

// stage 5 (CUDA): label connected components in a binary image
// components smaller than min_area pixels are treated as background
CCLResult ccl_label(const GrayImage& binary, int min_area = CCL_MIN_AREA);
void      ccl_free(CCLResult& r);

// ordered boundary of one labeled component
struct Contour {
    int                              label = 0;
    std::vector<std::pair<int,int>>  points;   // (x, y) in trace order
};

// stage 6 (CPU): trace the boundary of every component in a CCLResult
std::vector<Contour> trace_contours(const CCLResult& ccl);

// classification result for one contour
struct ShapeResult {
    int         label;
    int         contour_len;
    const char* shape;   // "circle", "rectangle", "triangle", "unknown"
    float       d2;      // normalized |Z[2]| (triangle indicator)
    float       d3;      // normalized |Z[3]| (rectangle indicator)
};

// stage 7 (cuFFT): compute Fourier Descriptors and classify each contour
std::vector<ShapeResult> classify_shapes(const std::vector<Contour>& contours);
