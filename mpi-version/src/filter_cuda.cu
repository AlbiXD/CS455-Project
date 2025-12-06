#include "../includes/filter_cuda.hpp"

__global__ void kernel_process(unsigned char* frame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        // Example processing: invert colors
        frame[idx + 0] = 255 - frame[idx + 0]; // Red
        frame[idx + 1] = 255 - frame[idx + 1]; // Green
        frame[idx + 2] = 255 - frame[idx + 2]; // Blue
    }
}

void process_frame_cuda(int choice, unsigned char* pixels_input, int width, int height) {
    int bx_dim, by_dim, gx_dim, gy_dim;

    bx_dim = by_dim = 4;
    gx_dim = (width + bx_dim-1)/bx_dim;
    gy_dim = (height + by_dim-1)/by_dim;

    dim3 threads(bx_dim, by_dim);
    dim3 grid(gx_dim, gy_dim);

    unsigned char * pixels_device;
    int size = width*height*3*sizeof(unsigned char);
    cudaMalloc(reinterpret_cast<void**>(&pixels_device), size);

    //Host->Device
    cudaMemcpy(pixels_device, pixels_input, size, cudaMemcpyHostToDevice);

    //Kernel
    kernel_process<<<grid, threads>>>(pixels_device, width, height);
    cudaError_t launchErr = cudaGetLastError();
    //if (launchErr != cudaSuccess) {std::cout << "I HATE YOUUU " << std::endl;}
    //Device->Host
    cudaMemcpy(pixels_input, pixels_device, size, cudaMemcpyDeviceToHost);
    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(pixels_device);
}

/*
void process_video_cuda_og(int rank, int choice, int frame_width, int frame_height, int start, int end){

    Mat frame;
	VideoWriter writer;
	int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
	double fps = cap.get(CAP_PROP_FPS);

	writer.open(part, fourcc, fps, Size(frame_width, frame_height), true);


    unsigned char* d_pixels;
    int frame_size = frame_width * frame_height * 3;
    cudaMalloc(&d_pixels, frame_size); // allocate once outside loop

    for(int i = start; i < end; i++){
        if(!cap.read(frame)) break;

        cudaMemcpy(d_pixels, frame.data, frame_size, cudaMemcpyHostToDevice);

        process_frame_cuda_kernel(d_pixels, frame_width, frame_height);

        cudaMemcpy(pixels, d_pixels, frame_size, cudaMemcpyDeviceToHost);

        Mat processed_frame(frame_height, frame_width, CV_8UC3, pixels);
        writer.write(processed_frame);
    }

    /*
    for(int i = start; i < end; i++){

        if(!cap.read(frame)){
            break;
        }
        memcpy(pixels, frame.data, frame_width * frame_height * 3);

        process_frame_cuda(choice, pixels, width, height)
        Mat processed_frame(frame_height, frame_width, CV_8UC3, pixels);
        writer.write(processed_frame);
    }/

}*/

__global__ void kernel_process_batch(unsigned char* frames, int width, int height, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z; // batch index
    int channels = 3;

    if (x < width && y < height && b < batch_size) {
        int frame_size = width * height * channels;
        int idx = b * frame_size + (y * width + x) * channels;

        // Example: zero out all pixels (black)
        frames[idx + 0] = 0;
        frames[idx + 1] = 0;
        frames[idx + 2] = 0;
    }
}

// Host wrapper callable from CPU
void process_frame_batch_cuda(unsigned char* d_frames, int width, int height, int batch_size) {
    dim3 threads(16, 16);
    dim3 grid((width + threads.x - 1)/threads.x, (height + threads.y - 1)/threads.y, batch_size);

    kernel_process_batch<<<grid, threads>>>(d_frames, width, height, batch_size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s\n", cudaGetErrorString(err));
    }
}













