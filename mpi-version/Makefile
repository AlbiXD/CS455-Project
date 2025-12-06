MPICC = mpic++
CFLAGS = `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

OUTPUT = my_program

video.out: ./build/main.o ./build/filters.o
	$(MPICC) $(CFLAGS) ./build/main.o ./build/filters.o $(LDFLAGS) -o video.out
./build/main.o: ./src/main.cpp
	mkdir -p ./build
	$(MPICC) $(CFLAGS) -c ./src/main.cpp -o ./build/main.o
./build/filters.o: ./src/filter.cpp
	mkdir -p ./build
	$(MPICC) $(CFLAGS) -c ./src/filter.cpp -o ./build/filters.o

clean:
	rm $(OUTPUT)
