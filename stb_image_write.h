// stb_image_write.h – minimal stub: writes P6 binary PPM only

#pragma once
#include <cstdint>
#include <cstdio>

static inline int stbi_write_png(
    const char* path,
    int w, int h,
    int channels,
    const uint8_t* data,
    int /*stride_bytes*/
) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t npix = (size_t)w * h;
    for (size_t i = 0; i < npix; ++i) {
        const uint8_t* px = data + i * channels;
        fwrite(px, 1, 3, f);  // R,G,B only
    }
    fclose(f);
    return 1;
}

#define STB_IMAGE_WRITE_IMPLEMENTATION
