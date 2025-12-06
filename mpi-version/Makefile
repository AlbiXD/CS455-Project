MPICC = mpic++
NVCC = nvcc

CFLAGS = -O3 `pkg-config --cflags opencv4`
NVCCFLAGS = -O3 `pkg-config --cflags opencv4` -I./includes -gencode=arch=compute_50,code=sm_50
LDFLAGS = `pkg-config --libs opencv4` -lcudart

OUTPUT = video.out
BUILD_DIR = ./build

# Object files
OBJ_CPP = $(BUILD_DIR)/main.o $(BUILD_DIR)/filters.o
OBJ_CU  = $(BUILD_DIR)/filters_cuda.o

# Target
video.out: $(OBJ_CPP) $(OBJ_CU)
	$(MPICC) $(OBJ_CPP) $(OBJ_CU) $(LDFLAGS) -o $(OUTPUT)

# Compile C++ sources
$(BUILD_DIR)/main.o: ./src/main.cpp
	mkdir -p $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -c ./src/main.cpp -o $(BUILD_DIR)/main.o

$(BUILD_DIR)/filters.o: ./src/filter.cpp ./includes/filter_cuda.hpp
	mkdir -p $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -c ./src/filter.cpp -o $(BUILD_DIR)/filters.o

# Compile CUDA sources
$(BUILD_DIR)/filters_cuda.o: ./src/filter_cuda.cu ./includes/filter_cuda.hpp
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c ./src/filter_cuda.cu -o $(BUILD_DIR)/filters_cuda.o

# Clean
clean:
	rm -rf $(OUTPUT) $(BUILD_DIR)/*.o
