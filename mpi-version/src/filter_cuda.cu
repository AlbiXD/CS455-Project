#include "../includes/filter_cuda.hpp"


__global__ void kernel_blur(const unsigned char* in,
                            unsigned char* out,
                            int width,
                            int height,
                            int radius)

//__global__ void kernel_blur(unsigned char* out, int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;

    if (x >= width || y >= height) return;

    int sumB = 0;
    int sumG = 0;
    int sumR = 0;
    int count = 0;

    // simple box blur (square kernel)
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = x + dx;
            int sy = y + dy;

            // clamp to edge
            if (sx < 0) sx = 0;
            if (sx >= width) sx = width - 1;
            if (sy < 0) sy = 0;
            if (sy >= height) sy = height - 1;

            int src_idx = (sy * width + sx) * channels;
            unsigned char b = in[src_idx + 0];
            unsigned char g = in[src_idx + 1];
            unsigned char r = in[src_idx + 2];

            sumB += b;
            sumG += g;
            sumR += r;
            count++;
        }
    }

    int dst_idx = (y * width + x) * channels;
    out[dst_idx + 0] = static_cast<unsigned char>(sumB / count);
    out[dst_idx + 1] = static_cast<unsigned char>(sumG / count);
    out[dst_idx + 2] = static_cast<unsigned char>(sumR / count);
}

__global__ void cuda_blur(unsigned char* in, unsigned char* out, int width, int height, int radius){
    int sum = 0;
    int sumR, sumG, sumB;
    int idxIn, idxOut;
    sumR = sumG = sumB = 0;
    int in_row, in_col;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y; //no my rank or my work- this process does 1 whole frame?
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if(out_row < height && out_col < width){
        for(in_row = out_row-radius; in_row <= out_row+radius; in_row++){
            if(in_row < 0 || in_row >= height){continue;}
            for(in_col = out_col-radius; in_col <= out_col+radius; in_col++){
                if(in_col < 0 || in_col >= width){continue;}

                    idxIn = (in_row * width + in_col)*3;
                    sumB += in[idxIn + 0];
                    sumG += in[idxIn + 1];
                    sumR += in[idxIn + 2];
                    sum++;

            }
        }
        idxOut = (out_row * width + out_col) * 3;
        out[idxOut + 0] = sumB/9;
        out[idxOut + 1] = sumG/9;
        out[idxOut + 2] = sumR/9;
    }

}
#define BLOCK_H 16
#define BLOCK_W 16

__global__ void cuda_edge_2(unsigned char* in, unsigned char* out, int width, int height, int funky) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int gx = bx * blockDim.x + tx;
    int gy = by * blockDim.y + ty;
    __shared__ float sGx[3][3];
    __shared__ float sGy[3][3];
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        sGx[0][0] = -1; sGx[0][1] = 0; sGx[0][2] = 1;
        sGx[1][0] = -2; sGx[1][1] = 0; sGx[1][2] = 2;
        sGx[2][0] = -1; sGx[2][1] = 0; sGx[2][2] = 1;

        sGy[0][0] = -1; sGy[0][1] = -2; sGy[0][2] = -1;
        sGy[1][0] =  0; sGy[1][1] =  0; sGy[1][2] =  0;
        sGy[2][0] =  1; sGy[2][1] =  2; sGy[2][2] =  1;
    }
    __syncthreads();
    if (gx == 0 || gx == width - 1 || gy == 0 || gy == height - 1) {
        int idx = (gy * width + gx) * 3;
        out[idx + 0] = 0;
        out[idx + 1] = 0;
        out[idx + 2] = 0;
        return;
    }
    if (gx >= 1 && gx < width - 1 && gy >= 1 && gy < height - 1) {
        int rX = 0, rY = 0;
        int gX = 0, gY = 0;
        int bX = 0, bY = 0;

        for (int fy = -1; fy <= 1; fy++) {
            for (int fx = -1; fx <= 1; fx++) {
                int idx = ((gy + fy) * width + (gx + fx)) * 3;
                unsigned char B = in[idx + 0];
                unsigned char G = in[idx + 1];
                unsigned char R = in[idx + 2];
                if(!funky){
                    float L = 0.299f * R + 0.587f * G + 0.114f * B;
                    unsigned char grey = static_cast<unsigned char>(L);
                    R = grey; G = grey; B = grey;
                }
                rX += R * sGx[fx + 1][fy + 1];
                rY += R * sGy[fx + 1][fy + 1];
                gX += G * sGx[fx + 1][fy + 1];
                gY += G * sGy[fx + 1][fy + 1];
                bX += B * sGx[fx + 1][fy + 1];
                bY += B * sGy[fx + 1][fy + 1];
            }
        }

        int idxOut = (gy * width + gx) * 3;
        out[idxOut + 0] = min(255, (int)sqrtf(bX*bX + bY*bY));
        out[idxOut + 1] = min(255, (int)sqrtf(gX*gX + gY*gY));
        out[idxOut + 2] = min(255, (int)sqrtf(rX*rX + rY*rY));
    }
}

__global__ void kernel_process(unsigned char* frame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        frame[idx + 0] = 255 - frame[idx + 0];
        frame[idx + 1] = 255 - frame[idx + 1];
        frame[idx + 2] = 255 - frame[idx + 2];
    }
}
__global__ void cuda_grey(unsigned char* frame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;
    int R, G, B;
    float L;
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        R = frame[idx + 2];
        G = frame[idx + 1];
        B = frame[idx + 0];
        L = 0.299f * R + 0.587f * G + 0.114f * B;
        unsigned char grey = static_cast<unsigned char>(L);
        frame[idx + 0] = grey;
        frame[idx + 1] = grey;
        frame[idx + 2] = grey;
    }
}



float process_frame_cuda(int choice, unsigned char* pixels_input, unsigned char *pixels_device, unsigned char* pixels_output, int width, int height, cudaStream_t *stream) {
    //Timing
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int bx_dim, by_dim, gx_dim, gy_dim;

    if(choice == 4){
        bx_dim = by_dim = 16;
    }
    else{
        bx_dim = by_dim = 4;
    }
    gx_dim = (width + bx_dim-1)/bx_dim;
    gy_dim = (height + by_dim-1)/by_dim;

    dim3 threads(bx_dim, by_dim);
    dim3 grid(gx_dim, gy_dim);


    int size = width*height*3*sizeof(unsigned char);
    //cudaMalloc(reinterpret_cast<void**>(&pixels_device), size);
    cudaEventRecord(ev_start, *stream);

    //Host->Device
    //cudaMemcpy(pixels_device, pixels_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(pixels_device, pixels_input, size, cudaMemcpyHostToDevice, *stream);
    //unsigned char *pixels_output;
    //cudaMalloc(reinterpret_cast<void**>(&pixels_output), size);

    //Kernel
    if(choice == 1){
        cuda_grey<<<grid, threads, 0, *stream>>>(pixels_device ,width, height);
        cudaMemcpyAsync(pixels_input, pixels_device, size, cudaMemcpyDeviceToHost, *stream);

    }
    else if(choice == 2){
        //RADIUS is 1
        cuda_blur<<<grid, threads, 0, *stream>>>(pixels_device, pixels_output,width, height, 1);
        cudaMemcpyAsync(pixels_input, pixels_output, size, cudaMemcpyDeviceToHost, *stream);

    }
    else if(choice == 3){
        kernel_process<<<grid, threads, 0, *stream>>>(pixels_device, width, height);
        cudaMemcpyAsync(pixels_input, pixels_device, size, cudaMemcpyDeviceToHost, *stream);
    }
    else if(choice == 4){
        size_t shared_mem_size = (16 + 2) * (16 + 2) * sizeof(unsigned char);
        //cuda_edge<<<grid, threads, shared_mem_size, *stream>>>(pixels_device, pixels_output,width, height, 1);
        cuda_edge_2<<<grid, threads, shared_mem_size, *stream>>>(pixels_device, pixels_output,width, height, 0);

        cudaMemcpyAsync(pixels_input, pixels_output, size, cudaMemcpyDeviceToHost, *stream);

    }
     else if(choice == 5){
        size_t shared_mem_size = (16 + 2) * (16 + 2) * sizeof(unsigned char);
        //cuda_edge<<<grid, threads, shared_mem_size, *stream>>>(pixels_device, pixels_output,width, height, 1);
        cuda_edge_2<<<grid, threads, shared_mem_size, *stream>>>(pixels_device, pixels_output,width, height, 1);

        cudaMemcpyAsync(pixels_input, pixels_output, size, cudaMemcpyDeviceToHost, *stream);

    }
    cudaError_t launchErr = cudaGetLastError();
    //if (launchErr != cudaSuccess) {std::cout << "I HATE YOUUU " << std::endl;}
    //Device->Host
    //cudaMemcpy(pixels_input, pixels_device, size, cudaMemcpyDeviceToHost);


    // Wait for GPU to finish
    /*cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }*/

    //cudaFree(pixels_device);

        //cudaMemcpyAsync(host_in, dev_buf, frame_bytes, cudaMemcpyDeviceToHost, stream);

    // record stop
    cudaEventRecord(ev_stop, *stream);

    // wait for event to complete
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return ms; // milliseconds
}

float process_frame_cudawhy(int choice,
                         unsigned char* pixels_input,
                         unsigned char* pixels_device_in,
                         unsigned char* pixels_device_out,
                         int width,
                         int height,
                         cudaStream_t* stream)
{
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int bx_dim = 16;
    int by_dim = 16;
    int gx_dim = (width  + bx_dim - 1) / bx_dim;
    int gy_dim = (height + by_dim - 1) / by_dim;

    dim3 threads(bx_dim, by_dim);
    dim3 grid(gx_dim, gy_dim);

    int size = width * height * 3 * sizeof(unsigned char);

    cudaEventRecord(ev_start, *stream);

    // Host -> Device
    cudaMemcpyAsync(pixels_device_in,
                    pixels_input,
                    size,
                    cudaMemcpyHostToDevice,
                    *stream);

    // choose what to do on GPU
    if (choice == 2) {
        // BLUR
        int radius = 20; // 3x3 box blur
        kernel_blur<<<grid, threads, 0, *stream>>>(
            pixels_device_in,
            pixels_device_out,
            width,
            height,
            radius
        );

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            printf("Blur kernel error: %s\n", cudaGetErrorString(launchErr));
        }

        // Device -> Host from OUT buffer
        cudaMemcpyAsync(pixels_input,
                        pixels_device_out,
                        size,
                        cudaMemcpyDeviceToHost,
                        *stream);
    } else {
        // fallback: old invert kernel, in-place
        kernel_process<<<grid, threads, 0, *stream>>>(
            pixels_device_in,
            width,
            height
        );

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            printf("Process kernel error: %s\n", cudaGetErrorString(launchErr));
        }

        cudaMemcpyAsync(pixels_input,
                        pixels_device_in,
                        size,
                        cudaMemcpyDeviceToHost,
                        *stream);
    }

    cudaEventRecord(ev_stop, *stream);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return ms;
}


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

    /*cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s\n", cudaGetErrorString(err));
    }*/
}













