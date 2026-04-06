// image_io.cpp – PPM load / PGM save

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "pipeline.h"
#include <fstream>
#include <iostream>
#include <cstring>

uint8_t* load_image(
    const std::string& path,
    int& out_w,
    int& out_h
) {
    int channels = 0;
    uint8_t* raw = stbi_load(path.c_str(), &out_w, &out_h, &channels, 4);

    if (!raw) {
        std::cerr << "[load_image] Failed: " << path << "\n"
                  << "  " << stbi_failure_reason() << "\n";
        return nullptr;
    }
    std::cout << "[load_image] " << path
              << "  " << out_w << "x" << out_h << "\n";

    size_t bytes = static_cast<size_t>(out_w) * out_h * 4;
    uint8_t* buf = new uint8_t[bytes];
    std::memcpy(buf, raw, bytes);
    stbi_image_free(raw);
    return buf;
}

bool save_pgm(const std::string& path, const GrayImage& img)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[save_pgm] Cannot open: " << path << "\n";
        return false;
    }

    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data),
            static_cast<std::streamsize>(img.width) * img.height);

    std::cout << "[save_pgm] " << path
              << "  " << img.width << "x" << img.height << "\n";
    return f.good();
}
