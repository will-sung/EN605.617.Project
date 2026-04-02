// gen_test_image.cpp – generates a synthetic test image (circle, rect, triangle)
// usage: ./gen_test_image [size] [output.ppm]
//        defaults: size=512, output=test_shapes.ppm

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

static int W = 512, H = 512;
static const int CH = 4;

static void put_pixel(uint8_t* buf, int x, int y,
                      uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    int idx = (y * W + x) * CH;
    buf[idx+0] = r; buf[idx+1] = g;
    buf[idx+2] = b; buf[idx+3] = 255;
}

static void draw_circle(uint8_t* buf, int cx, int cy, int radius,
                        uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = cy - radius; y <= cy + radius; ++y)
    for (int x = cx - radius; x <= cx + radius; ++x) {
        int dx = x-cx, dy = y-cy;
        if (dx*dx + dy*dy <= radius*radius)
            put_pixel(buf, x, y, r, g, b);
    }
}

static void draw_rect(uint8_t* buf, int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
        put_pixel(buf, x, y, r, g, b);
}

static void draw_triangle(uint8_t* buf,
                          int ax, int ay, int bx, int by, int cx, int cy,
                          uint8_t r, uint8_t g, uint8_t b)
{
    int minX = std::min({ax,bx,cx}), maxX = std::max({ax,bx,cx});
    int minY = std::min({ay,by,cy}), maxY = std::max({ay,by,cy});

    auto sign = [](int p1x,int p1y,int p2x,int p2y,int p3x,int p3y){
        return (p1x-p3x)*(p2y-p3y) - (p2x-p3x)*(p1y-p3y);
    };

    for (int py = minY; py <= maxY; ++py)
    for (int px = minX; px <= maxX; ++px) {
        int d1 = sign(px,py, ax,ay, bx,by);
        int d2 = sign(px,py, bx,by, cx,cy);
        int d3 = sign(px,py, cx,cy, ax,ay);
        bool neg = (d1<0)||(d2<0)||(d3<0);
        bool pos = (d1>0)||(d2>0)||(d3>0);
        if (!(neg && pos)) put_pixel(buf, px, py, r, g, b);
    }
}

int main(int argc, char* argv[])
{
    if (argc >= 2) W = H = std::atoi(argv[1]);
    std::string outfile = (argc >= 3) ? argv[2] : "test_shapes.ppm";

    // scale all shape coords from the baseline 512x512
    float s = W / 512.0f;

    size_t bytes = (size_t)W * H * CH;
    uint8_t* buf = new uint8_t[bytes];
    memset(buf, 230, bytes);
    for (int i = 3; i < (int)bytes; i += 4) buf[i] = 255;

    draw_circle(buf, (int)(s*128), (int)(s*128), (int)(s*80),
                60, 120, 200);
    draw_rect(buf, (int)(s*310), (int)(s*60), (int)(s*450), (int)(s*200),
              200, 80, 80);
    draw_triangle(buf,
                  (int)(s*256), (int)(s*310),
                  (int)(s*140), (int)(s*470),
                  (int)(s*370), (int)(s*470),
                  80, 180, 80);

    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
        if ((x * 17 + y * 31) % 97 == 0)
            put_pixel(buf, x, y, 10, 10, 10);

    stbi_write_png(outfile.c_str(), W, H, CH, buf, W*CH);
    std::cout << "Generated " << outfile << "  " << W << "x" << H << "\n";
    delete[] buf;
    return 0;
}
