#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#include <opencv2/core.hpp>


void apply_invert(unsigned char *my_frame, int frame_width, int frame_height);

void apply_grayscale(unsigned char *my_frame, int frame_width, int frame_height);

void apply_blur(unsigned char *my_frame, int frame_width, int frame_height);

int  apply_edge(unsigned char *my_frame, int frame_width, int frame_height);

int  apply_filter(int choice, unsigned char *my_frame, int frame_width, int frame_height);


#endif
