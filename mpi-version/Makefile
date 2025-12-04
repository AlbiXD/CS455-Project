MPICC = mpic++
CFLAGS = `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

# Define the output binary name
OUTPUT = my_program

# Target to create the build directory if it doesn't exist
build: main.cpp
	$(MPICC) $(CFLAGS) main.cpp $(LDFLAGS) -o $(OUTPUT)

# Clean up the build directory
clean:
	rm $(OUTPUT)
