// main.cu – shape recognition pipeline
// usage: ./pipeline <image.ppm> [blur_radius] [edge_thresh]

#include "pipeline.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

// map component labels to gray values for visualization
static GrayImage make_label_vis(const CCLResult& ccl)
{
    GrayImage vis;
    vis.width  = ccl.width;
    vis.height = ccl.height;
    size_t n   = (size_t)ccl.width * ccl.height;
    vis.data   = new uint8_t[n];
    for (size_t i = 0; i < n; i++) {
        uint32_t lbl = ccl.labels[i];
        vis.data[i]  = (lbl == 0) ? 0
                     : static_cast<uint8_t>(std::min(lbl * 50u, 255u));
    }
    return vis;
}

static void print_device_info() {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "[CUDA] Device " << dev
              << ": " << prop.name
              << "  SM " << prop.major << "." << prop.minor
              << "  " << (prop.totalGlobalMem >> 20) << " MiB\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <image> [blur_radius]\n";
        return EXIT_FAILURE;
    }

    const std::string input_path = argv[1];
    int blur_radius = (argc >= 3) ? std::atoi(argv[2]) : BLUR_RADIUS;
    int edge_thresh = (argc >= 4) ? std::atoi(argv[3]) : EDGE_THRESH;

    blur_radius = std::max(1,   std::min(blur_radius, 5));
    edge_thresh = std::max(1,   std::min(edge_thresh, 254));

    std::cout << "=== Shape Pre-Processing Pipeline ===\n";
    std::cout << "Input       : " << input_path << "\n";
    std::cout << "Blur radius : " << blur_radius << "\n";
    std::cout << "Edge thresh : " << edge_thresh << "\n";
    print_device_info();

    try {
        int width = 0, height = 0;
        uint8_t* rgba = load_image(input_path, width, height);
        if (!rgba) return EXIT_FAILURE;

        // stage 1: RGBA -> grayscale (result stays on device)
        DeviceGrayImage d_gray = npp_rgba_to_gray(rgba, width, height);
        delete[] rgba;
        { GrayImage h = device_gray_download(d_gray);
          save_pgm("out_1_gray.pgm", h); delete[] h.data; }

        // stage 2: gaussian blur (device -> device)
        DeviceGrayImage d_blurred = npp_gaussian_blur(d_gray, blur_radius);
        device_gray_free(d_gray);
        { GrayImage h = device_gray_download(d_blurred);
          save_pgm("out_2_blurred.pgm", h); delete[] h.data; }

        // stage 3: Sobel edge detection (device -> device)
        DeviceGrayImage d_edges = npp_sobel_edges(d_blurred);
        device_gray_free(d_blurred);
        { GrayImage h = device_gray_download(d_edges);
          save_pgm("out_3_edges.pgm", h); delete[] h.data; }

        // stage 4: threshold gradient magnitude -> binary edge map (device -> device)
        DeviceGrayImage d_binary = threshold_edges(d_edges, edge_thresh);
        device_gray_free(d_edges);
        { GrayImage h = device_gray_download(d_binary);
          save_pgm("out_4_binary.pgm", h); delete[] h.data; }

        // stage 5: connected component labeling (consumes d_binary on device)
        CCLResult ccl = ccl_label(d_binary, CCL_MIN_AREA);
        device_gray_free(d_binary);

        GrayImage label_vis = make_label_vis(ccl);
        save_pgm("out_5_labels.pgm", label_vis);
        delete[] label_vis.data;

        // stage 6: contour tracing (CPU)
        std::vector<Contour> contours = trace_contours(ccl);
        ccl_free(ccl);

        GrayImage contour_img;
        contour_img.width  = width;
        contour_img.height = height;
        contour_img.data   = new uint8_t[(size_t)width * height]();
        for (const auto& c : contours)
            for (const auto& pt : c.points)
                contour_img.data[pt.second * width + pt.first] = 255;
        save_pgm("out_6_contours.pgm", contour_img);
        delete[] contour_img.data;

        // step 7: Fourier Descriptors + classification
        std::cout << "\n--- Shape Classification ---\n";
        std::vector<ShapeResult> shapes = classify_shapes(contours);

        std::cout << "\n=== Results ===\n";
        for (const auto& s : shapes)
            std::cout << "  Shape " << s.label << ": " << s.shape << "\n";

        std::cout << "\n=== Pipeline complete ===\n";
        std::cout << "Outputs: out_1_gray.pgm  out_2_blurred.pgm  out_3_edges.pgm"
                  << "  out_4_binary.pgm  out_5_labels.pgm  out_6_contours.pgm\n";

    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
