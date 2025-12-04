CC = g++
CFLAGS = -O3 -Wall -Wextra `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`


video.out: ./build/main.o ./build/filters.o
	$(CC) $(CFLAGS) ./build/main.o ./build/filters.o $(LDFLAGS) -o video.out
build/main.o: ./src/main.cpp
	mkdir -p ./build
	$(CC) $(CFLAGS) -c ./src/main.cpp -o ./build/main.o
build/filters.o: ./src/filters.cpp
	mkdir -p ./build
	$(CC) $(CFLAGS) -c ./src/filters.cpp -o ./build/filters.o


clean:
	rm -rf build
	rm	video.out