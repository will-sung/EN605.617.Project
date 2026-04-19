// sobel_threshold.cu – Stage 4: binarize the Sobel gradient magnitude image
// pixels >= thresh -> 255 (edge), otherwise 0

#include "pipeline.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

static void chk_cuda(cudaError_t e, const char* loc)
{
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << loc << " : "
                  << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

__global__ static void threshold_kernel(
    const uint8_t* __restrict__ src,
    uint8_t*       __restrict__ dst,
    int     n,
    uint8_t thresh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = (src[i] >= thresh) ? 255u : 0u;
}

DeviceGrayImage threshold_edges(const DeviceGrayImage& src, int thresh_val)
{
    const int    w      = src.width, h = src.height;
    const size_t bytes  = static_cast<size_t>(w) * h;
    const uint8_t thresh = static_cast<uint8_t>(
        thresh_val <   0 ?   0 :
        thresh_val > 255 ? 255 : thresh_val);

    uint8_t* d_dst = nullptr;
    chk_cuda(cudaMalloc(&d_dst, bytes), "thresh malloc dst");

    const int threads = 256;
    const int blocks  = (w * h + threads - 1) / threads;
    threshold_kernel<<<blocks, threads>>>(src.d_data, d_dst, w * h, thresh);
    chk_cuda(cudaGetLastError(),      "threshold_kernel launch");
    chk_cuda(cudaDeviceSynchronize(), "threshold_kernel sync");

    std::cout << "[CUDA] Threshold  " << w << "x" << h
              << "  thresh=" << static_cast<int>(thresh) << "\n";
    return { w, h, d_dst };
}
