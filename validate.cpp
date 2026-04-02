// validate.cpp – checks pipeline output PGMs for correctness
// usage: ./validate [width height]   (default 512 512)

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct PGM {
    int width = 0, height = 0;
    std::vector<uint8_t> data;
};

static bool load_pgm(const std::string& path, PGM& img)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "  [LOAD-FAIL] Cannot open: " << path << "\n";
        return false;
    }

    std::string magic;
    f >> magic;
    if (magic != "P5") {
        std::cerr << "  [LOAD-FAIL] Not a P5 PGM: " << path << "\n";
        return false;
    }

    int maxval = 0;
    f >> img.width >> img.height >> maxval;
    f.ignore(1);  // whitespace before raster

    if (img.width <= 0 || img.height <= 0 || maxval != 255) {
        std::cerr << "  [LOAD-FAIL] Bad header: " << path << "\n";
        return false;
    }

    size_t npix = static_cast<size_t>(img.width) * img.height;
    img.data.resize(npix);
    f.read(reinterpret_cast<char*>(img.data.data()),
           static_cast<std::streamsize>(npix));

    if (!f) {
        std::cerr << "  [LOAD-FAIL] Truncated: " << path << "\n";
        return false;
    }
    return true;
}

struct Stats {
    double  mean = 0.0, stddev = 0.0;
    uint8_t minv = 255,  maxv  = 0;
    double  nonzero_frac = 0.0;
};

static Stats compute_stats(const PGM& img)
{
    Stats s;
    double sum = 0, sum2 = 0;
    size_t nz = 0;

    for (uint8_t v : img.data) {
        sum  += v;
        sum2 += (double)v * v;
        if (v < s.minv) s.minv = v;
        if (v > s.maxv) s.maxv = v;
        if (v > 0) ++nz;
    }

    const size_t n = img.data.size();
    s.mean         = sum / (double)n;
    s.stddev       = std::sqrt(sum2 / n - s.mean * s.mean);
    s.nonzero_frac = (double)nz / n;
    return s;
}

static int g_pass = 0, g_fail = 0;

static bool check(bool cond, const std::string& msg)
{
    std::cout << (cond ? "  [PASS] " : "  [FAIL] ") << msg << "\n";
    cond ? ++g_pass : ++g_fail;
    return cond;
}

static void print_stats(const char* label, const Stats& s)
{
    std::cout << "  " << label
              << "  mean="    << s.mean
              << "  stddev="  << s.stddev
              << "  min="     << (int)s.minv
              << "  max="     << (int)s.maxv
              << "  nonzero=" << (int)(s.nonzero_frac * 100) << "%\n";
}

int main(int argc, char* argv[])
{
    const int exp_w = (argc >= 3) ? std::atoi(argv[1]) : 512;
    const int exp_h = (argc >= 3) ? std::atoi(argv[2]) : 512;

    std::cout << "=== Pipeline Output Validation ===\n";
    std::cout << "Expected: " << exp_w << "x" << exp_h << "\n\n";

    PGM gray, blurred, edges;
    bool ok = true;

    std::cout << "--- Loading files ---\n";
    ok &= load_pgm("out_1_gray.pgm",    gray);
    ok &= load_pgm("out_2_blurred.pgm", blurred);
    ok &= load_pgm("out_3_edges.pgm",   edges);

    if (!ok) {
        std::cerr << "[ABORT] Failed to load output files.\n";
        return 1;
    }
    std::cout << "  OK\n\n";

    std::cout << "--- Dimension checks ---\n";
    check(gray.width    == exp_w && gray.height    == exp_h, "gray    " + std::to_string(exp_w) + "x" + std::to_string(exp_h));
    check(blurred.width == exp_w && blurred.height == exp_h, "blurred " + std::to_string(exp_w) + "x" + std::to_string(exp_h));
    check(edges.width   == exp_w && edges.height   == exp_h, "edges   " + std::to_string(exp_w) + "x" + std::to_string(exp_h));

    const Stats sg = compute_stats(gray);
    const Stats sb = compute_stats(blurred);
    const Stats se = compute_stats(edges);

    std::cout << "\n--- Grayscale (out_1_gray.pgm) ---\n";
    print_stats("gray   ", sg);
    check(sg.mean   > 10.0 && sg.mean < 245.0, "mean in range [10,245]");
    check(sg.stddev > 5.0,                      "non-trivial variance");
    check(sg.maxv   > 100,                      "has bright pixels");
    check(sg.minv   < 200,                      "has dark pixels");

    std::cout << "\n--- Gaussian blur (out_2_blurred.pgm) ---\n";
    print_stats("blurred", sb);
    check(sb.stddev <= sg.stddev,               "stddev <= gray (blur reduces variance)");
    check(std::abs(sb.mean - sg.mean) < 20.0,   "mean within 20 of gray");
    check(sb.maxv > 0,                          "not all-zero");

    std::cout << "\n--- Edge detection (out_3_edges.pgm) ---\n";
    print_stats("edges  ", se);
    check(se.mean < sg.mean,   "mean < gray (DC removed by high-pass)");
    check(se.maxv > 50,        "max > 50 (edges detected)");
    check(se.stddev > 20.0,    "stddev > 20 (edge/flat contrast)");

    std::cout << "\n=== " << g_pass << " passed, " << g_fail << " failed ===\n";
    return (g_fail > 0) ? 1 : 0;
}
