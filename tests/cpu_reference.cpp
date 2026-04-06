// cpu_reference.cpp – CPU pipeline for timing comparison and validation
// stages: load -> grayscale -> blur -> laplacian edges -> save PGMs

#include "stb_image.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

struct Gray {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
};

static bool save_pgm(const std::string& path, const Gray& g)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P5\n" << g.w << " " << g.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(g.px.data()), g.w * g.h);
    std::cout << "[CPU] saved " << path << "  " << g.w << "x" << g.h << "\n";
    return true;
}

// BT.601 coefficients match NPP's nppiRGBToGray
static Gray rgba_to_gray(const uint8_t* rgba, int w, int h)
{
    Gray out; out.w = w; out.h = h;
    out.px.resize(w * h);
    for (int i = 0; i < w * h; ++i) {
        out.px[i] = (uint8_t)(0.299f * rgba[i*4+0] +
                               0.587f * rgba[i*4+1] +
                               0.114f * rgba[i*4+2]);
    }
    std::cout << "[CPU] Grayscale  " << w << "x" << h << "\n";
    return out;
}

// separable box blur, clamped at borders
static Gray gaussian_blur(const Gray& src, int radius)
{
    const int w = src.w, h = src.h;
    Gray tmp; tmp.w = w; tmp.h = h; tmp.px = src.px;
    Gray out; out.w = w; out.h = h; out.px.resize(w * h, 0);

    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
        int sum = 0, cnt = 0;
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = std::max(0, std::min(w-1, x+dx));
            sum += src.px[y*w + nx]; ++cnt;
        }
        tmp.px[y*w + x] = (uint8_t)(sum / cnt);
    }

    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
        int sum = 0, cnt = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            int ny = std::max(0, std::min(h-1, y+dy));
            sum += tmp.px[ny*w + x]; ++cnt;
        }
        out.px[y*w + x] = (uint8_t)(sum / cnt);
    }

    std::cout << "[CPU] Blur  radius=" << radius << "\n";
    return out;
}

// 3x3 Laplacian:  0 -1 0 / -1 4 -1 / 0 -1 0
static Gray laplacian_edges(const Gray& src)
{
    const int w = src.w, h = src.h;
    Gray out; out.w = w; out.h = h;
    out.px.resize(w * h, 0);

    for (int y = 1; y < h-1; ++y)
    for (int x = 1; x < w-1; ++x) {
        int v = 4 * src.px[ y   *w + x  ]
                  - src.px[(y-1)*w + x  ]
                  - src.px[(y+1)*w + x  ]
                  - src.px[ y   *w + x-1]
                  - src.px[ y   *w + x+1];
        out.px[y*w + x] = (uint8_t)std::max(0, std::min(255, abs(v)));
    }

    std::cout << "[CPU] Laplacian edges\n";
    return out;
}

int main(int argc, char* argv[])
{
    const char* path = (argc >= 2) ? argv[1] : "test_shapes.ppm";
    int blur_radius  = (argc >= 3) ? atoi(argv[2]) : 3;

    std::cout << "=== CPU Reference Pipeline ===\n";
    std::cout << "Input: " << path << "\n";

    int w = 0, h = 0, comp = 0;
    uint8_t* rgba = stbi_load(path, &w, &h, &comp, 4);
    if (!rgba) {
        std::cerr << "Load failed: " << stbi_failure_reason() << "\n";
        return 1;
    }
    std::cout << "[CPU] Loaded " << w << "x" << h << "\n";

    Gray gray = rgba_to_gray(rgba, w, h);
    free(rgba);
    save_pgm("cpu_1_gray.pgm", gray);

    Gray blur = gaussian_blur(gray, blur_radius);
    save_pgm("cpu_2_blurred.pgm", blur);

    Gray edges = laplacian_edges(blur);
    save_pgm("cpu_3_edges.pgm", edges);

    std::cout << "=== CPU Reference Complete ===\n";
    return 0;
}
