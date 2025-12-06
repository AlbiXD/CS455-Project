#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>
#include <math.h>


int apply_invert(cv::VideoCapture cap, cv::VideoWriter out);

int apply_grayscale(cv::VideoCapture cap, cv::VideoWriter out);

int apply_blur(cv::VideoCapture cap, cv::VideoWriter out);

int apply_edge(cv::VideoCapture cap, cv::VideoWriter out);


#endif
