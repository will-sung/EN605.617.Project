// fourier_desc.cu – Stage 7: Fourier Descriptors + shape classification

#include "pipeline.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static void chk_cuda(cudaError_t e, const char* loc)
{
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << loc << " : "
                  << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

static void chk_fft(cufftResult r, const char* loc)
{
    if (r != CUFFT_SUCCESS) {
        std::cerr << "[cuFFT] " << loc << " status=" << r << "\n";
        throw std::runtime_error("cuFFT error");
    }
}

static float cmag(cufftComplex c)
{
    return sqrtf(c.x * c.x + c.y * c.y);
}

std::vector<ShapeResult> classify_shapes(const std::vector<Contour>& contours)
{
    std::vector<ShapeResult> results;

    // pre-allocate device buffers at max contour size and reuse
    int max_N = 0;
    for (const auto& c : contours)
        if ((int)c.points.size() >= 8)
            max_N = std::max(max_N, (int)c.points.size());

    if (max_N == 0) {
        for (const auto& c : contours)
            results.push_back({c.label, (int)c.points.size(), "unknown", 0.f, 0.f});
        return results;
    }

    cufftComplex* d_in  = nullptr;
    cufftComplex* d_out = nullptr;
    chk_cuda(cudaMalloc(&d_in,  max_N * sizeof(cufftComplex)), "fft malloc in");
    chk_cuda(cudaMalloc(&d_out, max_N * sizeof(cufftComplex)), "fft malloc out");

    for (const auto& contour : contours) {
        const int N = (int)contour.points.size();

        if (N < 8) {
            results.push_back({contour.label, N, "unknown", 0.f, 0.f});
            continue;
        }

        std::vector<cufftComplex> h_in(N);
        for (int n = 0; n < N; n++) {
            h_in[n].x = (float)contour.points[n].first;
            h_in[n].y = (float)contour.points[n].second;
        }

        chk_cuda(cudaMemcpy(d_in, h_in.data(), N * sizeof(cufftComplex),
                            cudaMemcpyHostToDevice), "fft H2D");

        cufftHandle plan;
        chk_fft(cufftPlan1d(&plan, N, CUFFT_C2C, 1), "cufftPlan1d");
        chk_fft(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD), "cufftExecC2C");
        cufftDestroy(plan);

        std::vector<cufftComplex> h_out(N);
        chk_cuda(cudaMemcpy(h_out.data(), d_out, N * sizeof(cufftComplex),
                            cudaMemcpyDeviceToHost), "fft D2H");

        float m1   = cmag(h_out[1]);
        float norm = (m1 > 1e-6f) ? m1 : 1.0f;

        float d2 = std::max(cmag(h_out[2]), cmag(h_out[N - 2])) / norm;
        float d3 = std::max(cmag(h_out[3]), cmag(h_out[N - 3])) / norm;

        const char* shape;
        if      (d2 > 0.08f && d2 > d3) shape = "triangle";
        else if (d3 > 0.08f && d3 > d2) shape = "rectangle";
        else                             shape = "circle";

        results.push_back({contour.label, N, shape, d2, d3});

        std::cout << "[cuFFT] label=" << contour.label
                  << "  len=" << N
                  << "  d2=" << d2
                  << "  d3=" << d3
                  << "  -> " << shape << "\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return results;
}
