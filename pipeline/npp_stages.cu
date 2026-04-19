// npp_stages.cu – NPP image processing stages
// stage 1: RGBA -> grayscale
// stage 2: gaussian blur
// stage 3: Sobel edge detection

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

void device_gray_free(DeviceGrayImage& img) {
    cudaFree(img.d_data);
    img.d_data = nullptr;
    img.width  = 0;
    img.height = 0;
}

GrayImage device_gray_download(const DeviceGrayImage& img) {
    const size_t bytes = static_cast<size_t>(img.width) * img.height;
    GrayImage out;
    out.width  = img.width;
    out.height = img.height;
    out.data   = new uint8_t[bytes];
    chk_cuda(cudaMemcpy(out.data, img.d_data, bytes,
                        cudaMemcpyDeviceToHost), "D2H gray");
    return out;
}

// stage 1 – RGBA to 8-bit grayscale
DeviceGrayImage npp_rgba_to_gray(
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

    cudaFree(d_rgba);

    std::cout << "[NPP] Grayscale  " << width << "x" << height << "\n";
    return { width, height, d_gray };
}

// stage 2 – gaussian blur
DeviceGrayImage npp_gaussian_blur(const DeviceGrayImage& src, int radius)
{
    assert(radius >= 1 && radius <= 5);
    const int w = src.width, h = src.height;
    const size_t bytes = static_cast<size_t>(w) * h;
    const int    step  = w;

    Npp8u* d_dst = nullptr;
    chk_cuda(cudaMalloc(&d_dst, bytes), "blur malloc dst");

    chk_cuda(cudaMemcpy(d_dst, src.d_data,
                        bytes, cudaMemcpyDeviceToDevice), "blur D2D seed");

    // radius 1->3x3, 2->5x5, 3+->7x7
    NppiMaskSize mask;
    switch (radius) {
        case 1:  mask = NPP_MASK_SIZE_3_X_3; break;
        case 2:  mask = NPP_MASK_SIZE_5_X_5; break;
        default: mask = NPP_MASK_SIZE_7_X_7; break;
    }

    int border   = radius;
    int srcOff   = border * step + border;
    NppiSize roi = { w - 2*border, h - 2*border };

    NppStreamContext nppCtx = make_npp_ctx();
    chk_npp(
        nppiFilterGauss_8u_C1R_Ctx(
            src.d_data + srcOff, step,
            d_dst      + srcOff, step,
            roi, mask, nppCtx
        ),
        "nppiFilterGauss"
    );

    std::cout << "[NPP] Blur  " << w << "x" << h
              << "  radius=" << radius << "\n";
    return { w, h, d_dst };
}

// stage 3 – Sobel edge detection
// clamp(sqrt(Gx^2 + Gy^2), 0, 255) -> 8u

__global__ static void sobel_magnitude_kernel(
    const int16_t* __restrict__ gx,
    const int16_t* __restrict__ gy,
    uint8_t*       __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mag = sqrtf((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
    out[i] = static_cast<uint8_t>(fminf(mag, 255.0f));
}

DeviceGrayImage npp_sobel_edges(const DeviceGrayImage& src)
{
    const int    w       = src.width,  h = src.height;
    const size_t bytes8  = static_cast<size_t>(w) * h;
    const size_t bytes16 = bytes8 * sizeof(int16_t);
    const int    step8   = w;
    const int    step16  = w * static_cast<int>(sizeof(int16_t));

    Npp16s* d_gx  = nullptr;
    Npp16s* d_gy  = nullptr;
    Npp8u*  d_out = nullptr;

    chk_cuda(cudaMalloc(&d_gx,  bytes16),  "sobel malloc gx");
    chk_cuda(cudaMalloc(&d_gy,  bytes16),  "sobel malloc gy");
    chk_cuda(cudaMalloc(&d_out, bytes8),   "sobel malloc out");

    NppiSize         srcSize = { w, h };
    NppiSize         roi     = { w, h };
    NppiPoint        offset  = { 0, 0 };
    NppStreamContext nppCtx  = make_npp_ctx();

    // horizontal gradient
    chk_npp(
        nppiFilterSobelHorizBorder_8u16s_C1R_Ctx(
            src.d_data, step8, srcSize, offset,
            d_gx,       step16, roi,
            NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, nppCtx
        ),
        "nppiFilterSobelHorizBorder"
    );

    // vertical gradient
    chk_npp(
        nppiFilterSobelVertBorder_8u16s_C1R_Ctx(
            src.d_data, step8, srcSize, offset,
            d_gy,       step16, roi,
            NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, nppCtx
        ),
        "nppiFilterSobelVertBorder"
    );

    int n       = w * h;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    sobel_magnitude_kernel<<<blocks, threads>>>(d_gx, d_gy, d_out, n);
    chk_cuda(cudaGetLastError(),      "sobel_magnitude_kernel launch");
    chk_cuda(cudaDeviceSynchronize(), "sobel_magnitude_kernel sync");

    cudaFree(d_gx);
    cudaFree(d_gy);

    std::cout << "[NPP] Sobel   " << w << "x" << h << "\n";
    return { w, h, d_out };
}
