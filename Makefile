CC = g++
CFLAGS = -Wall -Wextra
OPENCV = `pkg-config --cflags --libs opencv4`

test: test.cpp
	$(CC) $(CFLAGS) test.cpp -o test $(OPENCV)