CUDA_PATH ?= /usr/local/cuda-12.6
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

NVCC_FLAGS := -std=c++17 -O2 -lineinfo -arch=sm_86
CXX_FLAGS  := -std=c++17 -O2

LDFLAGS := -L$(CUDA_PATH)/lib64 \
           -lcufft -lnppig -lnppif -lnppicc -lnppc -lcudart

CU_SRCS  := main.cu npp_stages.cu sobel_threshold.cu ccl.cu fourier_desc.cu
CPP_SRCS := image_io.cpp contour.cpp

OBJ_CU  := $(CU_SRCS:.cu=.o)
OBJ_CPP := $(CPP_SRCS:.cpp=.o)

PIPELINE := pipeline
GEN_IMGS := gen_test_images
VALIDATE := validate

.PHONY: all test test_images benchmark clean

all: $(PIPELINE) $(GEN_IMGS) $(VALIDATE)

$(PIPELINE): $(OBJ_CU) $(OBJ_CPP)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu pipeline.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp pipeline.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(GEN_IMGS): gen_test_images.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@

test_images: $(GEN_IMGS)
	./$(GEN_IMGS)

$(VALIDATE): validate.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@

test: all test_images
	@echo ""
	@echo ">>> Pass 1: blur=3 thresh=40..."
	./$(PIPELINE) test_all.ppm 3 40
	@echo ""
	@echo ">>> Pass 2: blur=5 thresh=35..."
	./$(PIPELINE) test_all.ppm 5 35
	@echo ""
	@echo ">>> Validating outputs..."
	./$(VALIDATE) 512 512
	@ls -lh out_*.pgm test_all.ppm

benchmark: all cpu_reference
	python3 benchmark.py

cpu_reference: cpu_reference.cpp stb_image.h
	$(CXX) $(CXX_FLAGS) cpu_reference.cpp -I. -o cpu_reference

test_cpu: cpu_reference test_images
	@echo ""
	./cpu_reference test_all.ppm 3
	@echo ""
	./cpu_reference test_all.ppm 5
	@mv cpu_1_gray.pgm    cpu_pass2_1_gray.pgm    2>/dev/null || true
	@mv cpu_2_blurred.pgm cpu_pass2_2_blurred.pgm 2>/dev/null || true
	@mv cpu_3_edges.pgm   cpu_pass2_3_edges.pgm   2>/dev/null || true
	@ls -lh *.pgm test_all.ppm

clean:
	rm -f $(OBJ_CU) $(OBJ_CPP) $(PIPELINE) $(GEN_IMGS) $(VALIDATE) \
	      cpu_reference out_*.pgm test_*.ppm *.o \
	      benchmark_configs.png benchmark_scaling.png cpu_pass2_*.pgm
