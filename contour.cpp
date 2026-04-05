// contour.cpp – Stage 6: Moore-neighbor boundary tracing per labeled component

#include "pipeline.h"
#include <iostream>
#include <utility>
#include <vector>

static const int DX[8] = { 1, 1, 0,-1,-1,-1, 0, 1};
static const int DY[8] = { 0, 1, 1, 1, 0,-1,-1,-1};

// trace the ordered boundary of one component
static std::vector<std::pair<int,int>> trace_one(
    const uint32_t* labels, int w, int h, uint32_t lbl)
{
    // topmost-leftmost pixel of this component
    int sx = -1, sy = -1;
    for (int y = 0; y < h && sx < 0; y++)
        for (int x = 0; x < w && sx < 0; x++)
            if (labels[y * w + x] == lbl) { sx = x; sy = y; }

    std::vector<std::pair<int,int>> pts;
    if (sx < 0) return pts;

    int bd = 4;  // initial backtrack: West
    int cx = sx, cy = sy;
    pts.push_back({cx, cy});

    for (int iter = 0, limit = 4 * w * h; iter < limit; iter++) {
        bool found = false;
        for (int i = 1; i <= 8; i++) {
            int d  = (bd + i) % 8;
            int nx = cx + DX[d], ny = cy + DY[d];
            bool fg = ((unsigned)nx < (unsigned)w &&
                       (unsigned)ny < (unsigned)h &&
                       labels[ny * w + nx] == lbl);
            if (fg) {
                int bg_d = (bd + i - 1) % 8;
                int nbx  = cx + DX[bg_d], nby = cy + DY[bg_d];
                int ddx  = nbx - nx,      ddy = nby - ny;
                bd = 0;
                for (int k = 0; k < 8; k++)
                    if (DX[k] == ddx && DY[k] == ddy) { bd = k; break; }
                cx = nx; cy = ny;
                found = true;
                break;
            }
        }
        if (!found || (cx == sx && cy == sy)) break;
        pts.push_back({cx, cy});
    }

    return pts;
}

std::vector<Contour> trace_contours(const CCLResult& ccl)
{
    const int w = ccl.width, h = ccl.height;
    std::vector<Contour> result;

    for (int lbl = 1; lbl <= ccl.num_components; lbl++) {
        Contour c;
        c.label  = lbl;
        c.points = trace_one(ccl.labels, w, h, (uint32_t)lbl);
        if (!c.points.empty()) result.push_back(std::move(c));
    }

    std::cout << "[CPU]  Contours  " << w << "x" << h
              << "  found=" << result.size() << "  lengths:";
    for (const auto& c : result) std::cout << " " << c.points.size();
    std::cout << "\n";

    return result;
}
