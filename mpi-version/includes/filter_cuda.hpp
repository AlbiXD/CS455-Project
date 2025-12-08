#ifndef FILTER_CUDA_HPP
#define FILTER_CUDA_HPP

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
float process_frame_cuda(int choice, unsigned char* pixels_input, unsigned char *pixels_device, unsigned char *pixels_output, int width, int height, cudaStream_t *stream);
void process_video_cuda(int rank, int choice, int width, int height, int start, int end);
//void process_frame_batch_cuda(unsigned char* d_frames, int width, int height, int batch_size);
#endif
