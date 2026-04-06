CUDA_PATH ?= /usr/local/cuda-12.6
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

NVCC_FLAGS := -std=c++17 -O2 -lineinfo -arch=sm_86 -I pipeline
CXX_FLAGS  := -std=c++17 -O2 -I pipeline

LDFLAGS := -L$(CUDA_PATH)/lib64 \
           -lcufft -lnppig -lnppif -lnppicc -lnppc -lcudart

CU_SRCS  := pipeline/main.cu pipeline/npp_stages.cu pipeline/sobel_threshold.cu \
            pipeline/ccl.cu pipeline/fourier_desc.cu
CPP_SRCS := pipeline/image_io.cpp pipeline/contour.cpp

OBJ_CU  := $(CU_SRCS:.cu=.o)
OBJ_CPP := $(CPP_SRCS:.cpp=.o)

PIPELINE := pipeline_app
GEN_IMGS := gen_test_images
VALIDATE := validate

.PHONY: all test test_images benchmark clean

all: $(PIPELINE) $(GEN_IMGS) $(VALIDATE)

$(PIPELINE): $(OBJ_CU) $(OBJ_CPP)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

pipeline/%.o: pipeline/%.cu pipeline/pipeline.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

pipeline/%.o: pipeline/%.cpp pipeline/pipeline.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(GEN_IMGS): tests/gen_test_images.cpp
	$(CXX) -std=c++17 -O2 -I tests $< -o $@

test_images: $(GEN_IMGS)
	./$(GEN_IMGS)

$(VALIDATE): tests/validate.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@

test: all test_images
	@echo ""
	@echo ">>> Pass 1: blur=3 thresh=40..."
	./$(PIPELINE) tests/images/test_all.png 3 40
	@echo ""
	@echo ">>> Pass 2: blur=5 thresh=35..."
	./$(PIPELINE) tests/images/test_all.png 5 35
	@echo ""
	@echo ">>> Validating outputs..."
	./$(VALIDATE) 512 512
	@ls -lh out_*.pgm tests/images/test_all.png

benchmark: all cpu_reference
	python3 tests/benchmark.py

cpu_reference: tests/cpu_reference.cpp
	$(CXX) $(CXX_FLAGS) -I tests $< -o cpu_reference

test_cpu: cpu_reference test_images
	@echo ""
	./cpu_reference tests/images/test_all.png 3
	@echo ""
	./cpu_reference tests/images/test_all.png 5
	@mv cpu_1_gray.pgm    cpu_pass2_1_gray.pgm    2>/dev/null || true
	@mv cpu_2_blurred.pgm cpu_pass2_2_blurred.pgm 2>/dev/null || true
	@mv cpu_3_edges.pgm   cpu_pass2_3_edges.pgm   2>/dev/null || true
	@ls -lh *.pgm tests/images/test_all.png

clean:
	rm -f $(OBJ_CU) $(OBJ_CPP) $(PIPELINE) $(GEN_IMGS) $(VALIDATE) \
	      cpu_reference out_*.pgm tests/images/*.png pipeline/*.o \
	      benchmark_configs.png benchmark_scaling.png cpu_pass2_*.pgm
