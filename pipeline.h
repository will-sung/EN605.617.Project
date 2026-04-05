#pragma once
// pipeline.h – shared types and function declarations

#include <cstdint>
#include <string>

static constexpr int MAX_DIM     = 4096;
static constexpr int BLUR_RADIUS = 3;    // default -> 7x7 kernel

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
