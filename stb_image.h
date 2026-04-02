// stb_image.h – minimal P6 PPM loader stub

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static const char* stbi__reason_g = "unknown";
static inline const char* stbi_failure_reason() { return stbi__reason_g; }

static inline void ppm_skip_ws(FILE* f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != '\n' && c != EOF) {}
        } else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            ungetc(c, f);
            break;
        }
    }
}

static inline int ppm_read_int(FILE* f) {
    ppm_skip_ws(f);
    int v = 0, c;
    while ((c = fgetc(f)) != EOF) {
        if (c >= '0' && c <= '9') v = v * 10 + (c - '0');
        else { ungetc(c, f); break; }
    }
    return v;
}

static inline uint8_t* stbi_load(
    const char* path,
    int* out_w, int* out_h,
    int* out_comp,
    int  /*desired_channels*/
) {
    FILE* f = fopen(path, "rb");
    if (!f) { stbi__reason_g = "cannot open file"; return nullptr; }

    char magic[3] = {};
    if (fread(magic, 1, 2, f) != 2 ||
        magic[0] != 'P' || magic[1] != '6') {
        stbi__reason_g = "not a P6 PPM";
        fclose(f); return nullptr;
    }

    int W  = ppm_read_int(f);
    int H  = ppm_read_int(f);
    int mx = ppm_read_int(f);
    fgetc(f);  // single whitespace before raster

    if (W <= 0 || H <= 0 || mx != 255) {
        stbi__reason_g = "unsupported PPM (must be 8-bit P6)";
        fclose(f); return nullptr;
    }
    *out_w = W; *out_h = H; *out_comp = 3;

    size_t   npix = (size_t)W * H;
    uint8_t* rgb  = (uint8_t*)malloc(npix * 3);
    uint8_t* rgba = (uint8_t*)malloc(npix * 4);
    if (!rgb || !rgba) {
        stbi__reason_g = "out of memory";
        free(rgb); free(rgba); fclose(f); return nullptr;
    }

    if (fread(rgb, 3, npix, f) != npix) {
        stbi__reason_g = "truncated PPM pixel data";
        free(rgb); free(rgba); fclose(f); return nullptr;
    }
    fclose(f);

    for (size_t i = 0; i < npix; ++i) {
        rgba[i*4+0] = rgb[i*3+0];
        rgba[i*4+1] = rgb[i*3+1];
        rgba[i*4+2] = rgb[i*3+2];
        rgba[i*4+3] = 255;
    }
    free(rgb);
    return rgba;
}

static inline void stbi_image_free(void* p) { free(p); }

// allow #define STB_IMAGE_IMPLEMENTATION without symbol clash
#define STB_IMAGE_IMPLEMENTATION
