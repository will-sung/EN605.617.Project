// gen_test_images.cpp – generates 7 test PPMs for the pipeline

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

static const int W = 512, H = 512, CH = 4;

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
    for (int y = cy - radius; y <= cy + radius; y++)
    for (int x = cx - radius; x <= cx + radius; x++) {
        int dx = x - cx, dy = y - cy;
        if (dx*dx + dy*dy <= radius*radius)
            put_pixel(buf, x, y, r, g, b);
    }
}

static void draw_rect(uint8_t* buf, int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++)
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

    for (int py = minY; py <= maxY; py++)
    for (int px = minX; px <= maxX; px++) {
        int d1 = sign(px,py, ax,ay, bx,by);
        int d2 = sign(px,py, bx,by, cx,cy);
        int d3 = sign(px,py, cx,cy, ax,ay);
        bool neg = (d1<0)||(d2<0)||(d3<0);
        bool pos = (d1>0)||(d2>0)||(d3>0);
        if (!(neg && pos)) put_pixel(buf, px, py, r, g, b);
    }
}

static uint8_t* make_canvas()
{
    size_t bytes = (size_t)W * H * CH;
    uint8_t* buf = new uint8_t[bytes];
    memset(buf, 230, bytes);
    for (int i = 3; i < (int)bytes; i += 4) buf[i] = 255;
    return buf;
}

static void save(uint8_t* buf, const char* path)
{
    stbi_write_png(path, W, H, CH, buf, W * CH);
    std::cout << "Generated " << path << "\n";
}

int main()
{
    uint8_t* buf;

    // --- single shapes ---

    buf = make_canvas();
    draw_circle(buf, 256, 256, 130, 60, 120, 200);
    save(buf, "test_circle.ppm");
    delete[] buf;

    buf = make_canvas();
    draw_rect(buf, 131, 181, 381, 331, 200, 80, 80);
    save(buf, "test_rect.ppm");
    delete[] buf;

    buf = make_canvas();
    draw_triangle(buf, 256, 110, 96, 400, 416, 400, 80, 180, 80);
    save(buf, "test_triangle.ppm");
    delete[] buf;

    // --- pairs ---

    buf = make_canvas();
    draw_circle(buf,  128, 256, 90,  60, 120, 200);
    draw_rect(buf,   290, 185, 470, 325, 200, 80,  80);
    save(buf, "test_circle_rect.ppm");
    delete[] buf;

    buf = make_canvas();
    draw_circle(buf,  128, 256, 90,  60, 120, 200);
    draw_triangle(buf, 384, 120, 270, 400, 498, 400, 80, 180, 80);
    save(buf, "test_circle_tri.ppm");
    delete[] buf;

    buf = make_canvas();
    draw_rect(buf,    30, 185, 220, 325, 200, 80,  80);
    draw_triangle(buf, 384, 120, 270, 400, 498, 400, 80, 180, 80);
    save(buf, "test_rect_tri.ppm");
    delete[] buf;

    // --- all three ---

    buf = make_canvas();
    draw_circle(buf,   128, 128, 80,  60, 120, 200);
    draw_rect(buf,     310,  60, 450, 200, 200, 80,  80);
    draw_triangle(buf, 256, 310, 140, 470, 370, 470, 80, 180, 80);
    save(buf, "test_all.ppm");
    delete[] buf;

    return 0;
}
