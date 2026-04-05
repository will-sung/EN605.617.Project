// main.cu – image pre-processing pipeline
// stages: load -> grayscale (NPP) -> blur (NPP) -> Sobel (NPP) -> threshold (CUDA)
// usage: ./pipeline <image.ppm> [blur_radius] [edge_thresh]

#include "pipeline.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cuda_runtime.h>

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

        // stage 1: RGBA -> grayscale
        GrayImage gray = npp_rgba_to_gray(rgba, width, height);
        delete[] rgba;
        save_pgm("out_1_gray.pgm", gray);

        // stage 2: gaussian blur – reduces noise before edge detection
        GrayImage blurred = npp_gaussian_blur(gray, blur_radius);
        delete[] gray.data;
        save_pgm("out_2_blurred.pgm", blurred);

        // stage 3: Sobel edge detection on the blurred output
        GrayImage edges = npp_sobel_edges(blurred);
        delete[] blurred.data;
        save_pgm("out_3_edges.pgm", edges);

        // stage 4: threshold gradient magnitude -> binary edge map
        GrayImage binary = threshold_edges(edges, edge_thresh);
        delete[] edges.data;
        save_pgm("out_4_binary.pgm", binary);
        delete[] binary.data;

        std::cout << "\n=== Pipeline complete ===\n";
        std::cout << "Outputs: out_1_gray.pgm  out_2_blurred.pgm  "
                  << "out_3_edges.pgm  out_4_binary.pgm\n";

    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
