// npp_stages.cu – NPP image processing stages
// stage 1: RGBA -> grayscale (nppiRGBToGray)
// stage 2: gaussian blur    (nppiFilterGauss)

#include "pipeline.h"

#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cassert>
#include <stdexcept>

static void chk_cuda(cudaError_t e, const char* loc) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << loc << " : "
                  << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

static void chk_npp(NppStatus s, const char* loc) {
    if (s != NPP_SUCCESS) {
        std::cerr << "[NPP] " << loc
                  << " status=" << static_cast<int>(s) << "\n";
        throw std::runtime_error("NPP error");
    }
}

static NppStreamContext make_npp_ctx() {
    NppStreamContext ctx{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    ctx.hStream                            = nullptr;
    ctx.nCudaDeviceId                      = dev;
    ctx.nMultiProcessorCount               = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor       = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock                = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock                 = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    cudaStreamGetFlags(nullptr, &ctx.nStreamFlags);
    return ctx;
}

// stage 1 – RGBA to 8-bit grayscale
GrayImage npp_rgba_to_gray(
    const uint8_t* h_rgba,
    int width,
    int height
) {
    const size_t rgba_bytes = static_cast<size_t>(width) * height * 4;
    const size_t gray_bytes = static_cast<size_t>(width) * height;

    Npp8u* d_rgba = nullptr;
    Npp8u* d_gray = nullptr;
    int rgba_step = width * 4;
    int gray_step = width;

    chk_cuda(cudaMalloc(&d_rgba, rgba_bytes), "malloc rgba");
    chk_cuda(cudaMalloc(&d_gray, gray_bytes), "malloc gray");
    chk_cuda(cudaMemcpy(d_rgba, h_rgba,
                        rgba_bytes, cudaMemcpyHostToDevice), "H2D rgba");

    NppiSize roi = { width, height };
    NppStreamContext nppCtx = make_npp_ctx();
    chk_npp(
        nppiRGBToGray_8u_AC4C1R_Ctx(
            d_rgba, rgba_step,
            d_gray, gray_step,
            roi, nppCtx
        ),
        "nppiRGBToGray"
    );

    GrayImage out;
    out.width  = width;
    out.height = height;
    out.data   = new uint8_t[gray_bytes];
    chk_cuda(cudaMemcpy(out.data, d_gray,
                        gray_bytes, cudaMemcpyDeviceToHost), "D2H gray");

    cudaFree(d_rgba);
    cudaFree(d_gray);

    std::cout << "[NPP] Grayscale  " << width << "x" << height << "\n";
    return out;
}

// stage 2 – gaussian blur
GrayImage npp_gaussian_blur(const GrayImage& src, int radius)
{
    assert(radius >= 1 && radius <= 5);
    const int w = src.width, h = src.height;
    const size_t bytes = static_cast<size_t>(w) * h;
    const int    step  = w;

    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    chk_cuda(cudaMalloc(&d_src, bytes), "blur malloc src");
    chk_cuda(cudaMalloc(&d_dst, bytes), "blur malloc dst");
    chk_cuda(cudaMemcpy(d_src, src.data,
                        bytes, cudaMemcpyHostToDevice), "blur H2D");

    // radius 1->3x3, 2->5x5, 3+->7x7
    NppiMaskSize mask;
    switch (radius) {
        case 1:  mask = NPP_MASK_SIZE_3_X_3; break;
        case 2:  mask = NPP_MASK_SIZE_5_X_5; break;
        default: mask = NPP_MASK_SIZE_7_X_7; break;
    }

    // seed d_dst so border pixels (outside ROI) keep original values
    chk_cuda(cudaMemcpy(d_dst, d_src,
                        bytes, cudaMemcpyDeviceToDevice), "blur D2D seed");

    int border   = radius;
    int srcOff   = border * step + border;
    NppiSize roi = { w - 2*border, h - 2*border };

    NppStreamContext nppCtx = make_npp_ctx();
    chk_npp(
        nppiFilterGauss_8u_C1R_Ctx(
            d_src + srcOff, step,
            d_dst + srcOff, step,
            roi, mask, nppCtx
        ),
        "nppiFilterGauss"
    );

    GrayImage out;
    out.width  = w;
    out.height = h;
    out.data   = new uint8_t[bytes];
    chk_cuda(cudaMemcpy(out.data, d_dst,
                        bytes, cudaMemcpyDeviceToHost), "blur D2H");

    cudaFree(d_src);
    cudaFree(d_dst);

    std::cout << "[NPP] Blur  " << w << "x" << h
              << "  radius=" << radius << "\n";
    return out;
}
