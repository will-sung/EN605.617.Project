// cufft_edge.cu – frequency-domain edge detection via cuFFT
// forward R2C -> high-pass filter -> inverse C2R -> normalize to uint8

#include "pipeline.h"

#include <cufft.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void chk_cuda(cudaError_t e, const char* loc) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << loc << ": "
                  << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

static void chk_fft(cufftResult r, const char* loc) {
    if (r != CUFFT_SUCCESS) {
        std::cerr << "[cuFFT] " << loc
                  << " result=" << static_cast<int>(r) << "\n";
        throw std::runtime_error("cuFFT error");
    }
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// each thread scales one frequency bin F[v][u] by its distance from DC
__global__ void highpass_kernel(
    cufftComplex* F,
    int           padH,
    int           padW2,  // complex row width = padW/2 + 1
    float         threshold
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= padW2 || v >= padH) return;

    // centre v coordinate
    float fv = (v <= padH / 2) ? (float)v : (float)(v - padH);
    float fu = (float)u;

    float maxDist = sqrtf((float)(padH/2)*(padH/2) + (float)padW2*padW2);
    float dist    = sqrtf(fv*fv + fu*fu) / maxDist;

    // ramp: zero below threshold, linear ramp to 1 above
    float w = (dist < threshold)
              ? 0.0f
              : (dist - threshold) / (1.0f - threshold);

    int idx = v * padW2 + u;
    F[idx].x *= w;
    F[idx].y *= w;
}

// crop from padded buffer, abs + scale to uint8
__global__ void normalise_kernel(
    const float* in,
    uint8_t*     out,
    int          padW,
    int          srcW,
    int          srcH,
    float        scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= srcW || y >= srcH) return;

    int iv = (int)fabsf(in[y * padW + x] * scale);
    out[y * srcW + x] = (uint8_t)(iv > 255 ? 255 : iv);
}

GrayImage cufft_edge_detect(const GrayImage& src, float threshold)
{
    const int W   = src.width,  H  = src.height;
    const int PW  = next_pow2(W), PH = next_pow2(H);
    const int PW2 = PW / 2 + 1;  // R2C complex row length

    std::cout << "[cuFFT] Edge detect  "
              << W << "x" << H
              << "  pad=" << PW << "x" << PH << "\n";

    const size_t real_sz    = sizeof(float)        * PH * PW;
    const size_t complex_sz = sizeof(cufftComplex) * PH * PW2;

    float*        d_real = nullptr;
    cufftComplex* d_freq = nullptr;
    uint8_t*      d_out  = nullptr;

    chk_cuda(cudaMalloc(&d_real, real_sz),    "fft malloc real");
    chk_cuda(cudaMalloc(&d_freq, complex_sz), "fft malloc freq");
    chk_cuda(cudaMalloc(&d_out,  W * H),      "fft malloc out");
    chk_cuda(cudaMemset(d_real, 0, real_sz),  "fft memset");

    // expand uint8 src into top-left of padded float buffer
    std::vector<float> h_padded(PH * PW, 0.0f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            h_padded[y * PW + x] = (float)src.data[y * W + x];

    chk_cuda(cudaMemcpy(d_real, h_padded.data(),
                        real_sz, cudaMemcpyHostToDevice), "fft H2D");

    cufftHandle plan_fwd, plan_inv;
    chk_fft(cufftPlan2d(&plan_fwd, PH, PW, CUFFT_R2C), "plan R2C");
    chk_fft(cufftExecR2C(plan_fwd, d_real, d_freq),     "exec R2C");

    dim3 block(16, 16);
    dim3 grid_f((PW2+15)/16, (PH+15)/16);
    highpass_kernel<<<grid_f, block>>>(d_freq, PH, PW2, threshold);
    chk_cuda(cudaGetLastError(), "highpass_kernel");

    chk_fft(cufftPlan2d(&plan_inv, PH, PW, CUFFT_C2R), "plan C2R");
    chk_fft(cufftExecC2R(plan_inv, d_freq, d_real),     "exec C2R");

    // cuFFT is unnormalized, scale by 1/(PH*PW) then to [0,255]
    float scale = 255.0f / (float)(PH * PW);
    dim3 grid_n((W+15)/16, (H+15)/16);
    normalise_kernel<<<grid_n, block>>>(d_real, d_out, PW, W, H, scale);
    chk_cuda(cudaGetLastError(), "normalise_kernel");

    GrayImage out;
    out.width  = W;
    out.height = H;
    out.data   = new uint8_t[W * H];
    chk_cuda(cudaMemcpy(out.data, d_out,
                        W * H, cudaMemcpyDeviceToHost), "fft D2H");

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_real);
    cudaFree(d_freq);
    cudaFree(d_out);

    std::cout << "[cuFFT] Edge detect done\n";
    return out;
}
