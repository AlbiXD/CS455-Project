#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>



int apply_invert(cv::VideoCapture cap, cv::VideoWriter out);

int apply_grayscale(cv::VideoCapture cap, cv::VideoWriter out);



#endif