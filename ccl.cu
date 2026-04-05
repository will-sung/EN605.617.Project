// ccl.cu – Stage 5: connected component labeling on binary edge map
// uses iterative label propagation with ping-pong buffers

#include "pipeline.h"

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdint>

static void chk_cuda(cudaError_t e, const char* loc)
{
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << loc << " : "
                  << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

static constexpr uint32_t BG = 0xFFFFFFFFu;  // background sentinel

// foreground pixels get label = linear index; background gets BG
__global__ static void ccl_init_kernel(
    const uint8_t* __restrict__ binary,
    uint32_t*      __restrict__ labels,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    labels[i] = (binary[i] > 0) ? (uint32_t)i : BG;
}

// one propagation pass: each foreground pixel takes the minimum label
// of itself and its 8-connected foreground neighbors
__global__ static void ccl_propagate_kernel(
    const uint32_t* __restrict__ src,
    uint32_t*       __restrict__ dst,
    int w, int h,
    int* d_changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int i = y * w + x;
    uint32_t cur = src[i];
    if (cur == BG) { dst[i] = BG; return; }

    uint32_t best = cur;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            int nx = x + dx, ny = y + dy;
            if ((unsigned)nx < (unsigned)w && (unsigned)ny < (unsigned)h) {
                uint32_t nl = src[ny * w + nx];
                if (nl < best) best = nl;
            }
        }
    }
    dst[i] = best;
    if (best < cur) *d_changed = 1;
}

CCLResult ccl_label(const GrayImage& binary, int min_area)
{
    const int    w      = binary.width, h = binary.height;
    const size_t npix   = (size_t)w * h;
    const size_t lbytes = npix * sizeof(uint32_t);

    uint8_t* d_bin = nullptr;
    chk_cuda(cudaMalloc(&d_bin, npix), "ccl malloc bin");
    chk_cuda(cudaMemcpy(d_bin, binary.data, npix,
                        cudaMemcpyHostToDevice), "ccl H2D");

    uint32_t* d_lab[2] = {};
    chk_cuda(cudaMalloc(&d_lab[0], lbytes), "ccl malloc lab0");
    chk_cuda(cudaMalloc(&d_lab[1], lbytes), "ccl malloc lab1");

    int* d_changed = nullptr;
    chk_cuda(cudaMalloc(&d_changed, sizeof(int)), "ccl malloc changed");

    // init labels
    int threads = 256;
    int blocks  = ((int)npix + threads - 1) / threads;
    ccl_init_kernel<<<blocks, threads>>>(d_bin, d_lab[0], (int)npix);
    chk_cuda(cudaGetLastError(),      "ccl_init launch");
    chk_cuda(cudaDeviceSynchronize(), "ccl_init sync");
    cudaFree(d_bin);

    // propagate until convergence
    dim3 blk(16, 16);
    dim3 grd((w + 15) / 16, (h + 15) / 16);
    int ping = 0;

    for (int it = 0, max_it = w + h; it < max_it; it++) {
        int zero = 0;
        chk_cuda(cudaMemcpy(d_changed, &zero, sizeof(int),
                            cudaMemcpyHostToDevice), "ccl reset");

        ccl_propagate_kernel<<<grd, blk>>>(
            d_lab[ping], d_lab[1 - ping], w, h, d_changed);
        chk_cuda(cudaGetLastError(),      "ccl_propagate launch");
        chk_cuda(cudaDeviceSynchronize(), "ccl_propagate sync");
        ping = 1 - ping;

        int changed = 0;
        chk_cuda(cudaMemcpy(&changed, d_changed, sizeof(int),
                            cudaMemcpyDeviceToHost), "ccl read changed");
        if (!changed) break;
    }

    // download
    std::vector<uint32_t> raw(npix);
    chk_cuda(cudaMemcpy(raw.data(), d_lab[ping], lbytes,
                        cudaMemcpyDeviceToHost), "ccl D2H");
    cudaFree(d_lab[0]);
    cudaFree(d_lab[1]);
    cudaFree(d_changed);

    // count component sizes on CPU
    std::unordered_map<uint32_t, int> sizes;
    for (uint32_t l : raw)
        if (l != BG) sizes[l]++;

    // assign sequential IDs to components >= min_area
    std::unordered_map<uint32_t, uint32_t> remap;
    uint32_t next = 1;
    for (auto& [lbl, sz] : sizes)
        if (sz >= min_area) remap[lbl] = next++;

    // write final label map
    uint32_t* out = new uint32_t[npix];
    for (size_t i = 0; i < npix; i++) {
        if (raw[i] == BG) { out[i] = 0; continue; }
        auto it = remap.find(raw[i]);
        out[i] = (it != remap.end()) ? it->second : 0;
    }

    int n_comp = (int)(next - 1);
    std::cout << "[CUDA] CCL    " << w << "x" << h
              << "  components=" << n_comp << "\n";

    CCLResult res;
    res.width          = w;
    res.height         = h;
    res.labels         = out;
    res.num_components = n_comp;
    return res;
}

void ccl_free(CCLResult& r)
{
    delete[] r.labels;
    r.labels         = nullptr;
    r.num_components = 0;
}
