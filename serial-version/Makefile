CC = g++
CFLAGS = -Wall -Wextra
OPENCV = `pkg-config --cflags --libs opencv4`

a.out: ./src/main.cpp
	$(CC) $(CFLAGS) ./src/main.cpp  $(OPENCV)
