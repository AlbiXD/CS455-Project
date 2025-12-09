MPICC    = mpic++
NVCC     = nvcc

CFLAGS   = -O3 $(shell pkg-config --cflags opencv4)
LDFLAGS  = $(shell pkg-config --libs opencv4) -lcudart

NVCCFLAGS = -O3 \
            -gencode arch=compute_50,code=sm_50  \
            -gencode arch=compute_86,code=sm_86  \
            -gencode arch=compute_89,code=sm_89


OUTPUT    = video.out
BUILD_DIR = ./build

OBJ_CPP = $(BUILD_DIR)/main.o $(BUILD_DIR)/filters.o
OBJ_CU  = $(BUILD_DIR)/filters_cuda.o

$(OUTPUT): $(OBJ_CPP) $(OBJ_CU)
	$(MPICC) $(OBJ_CPP) $(OBJ_CU) $(LDFLAGS) -o $(OUTPUT)

$(BUILD_DIR)/main.o: ./src/main.cpp
	mkdir -p $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -c ./src/main.cpp -o $(BUILD_DIR)/main.o

$(BUILD_DIR)/filters.o: ./src/filter.cpp ./includes/filter_cuda.hpp
	mkdir -p $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -c ./src/filter.cpp -o $(BUILD_DIR)/filters.o

$(BUILD_DIR)/filters_cuda.o: ./src/filter_cuda.cu ./includes/filter_cuda.hpp
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c ./src/filter_cuda.cu -o $(BUILD_DIR)/filters_cuda.o

clean:
	rm -rf $(OUTPUT) $(BUILD_DIR)
