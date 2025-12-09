# Distributed Video Filter (MPI + CUDA)

High-performance video filtering pipeline in C++ using OpenCV, MPI, and CUDA. Implements serial, MPI, CUDA, and hybrid CUDA+MPI versions with benchmarked speedups on multi-node clusters.

## Overview

This project implements a distributed video processing pipeline with four execution modes:

- **Serial** – baseline C++ + OpenCV implementation
- **MPI** – frame-level parallelism across multiple processes / nodes
- **CUDA** – GPU-accelerated filtering on a single machine
- **CUDA + MPI** – hybrid version that distributes work across nodes, each using its GPU
<p align="center">
  <img src="https://github.com/user-attachments/assets/434b6704-904a-4c1a-8e03-a30cc7ec21fa" width="320" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/0d17fec0-6d21-4b17-aea4-a0a4aa21919e" width="320" />
</p>


## Benchmark
### Serial Implementation
<img width="876" height="541" alt="serial_grayscale_5min" src="https://github.com/user-attachments/assets/7715a0a4-6d9a-443f-9c4f-b137235a7707" />

### MPI Implementation
<img width="896" height="519" alt="mpi_grayscale_5min" src="https://github.com/user-attachments/assets/4c14a38a-d1de-438a-b320-3af34b744a81" />

### CUDA Implementation
<img width="876" height="541" alt="cuda_grayscale_5min" src="https://github.com/user-attachments/assets/6b240683-280c-4e71-8099-1bce628746de" />

### CUDA + MPI Implementation
<img width="886" height="497" alt="cuda_mpi-5min" src="https://github.com/user-attachments/assets/4c1c3714-2143-450e-888b-fd168c07a367" />

As expected, the serial implementation was the slowest, with the MPI CPU implementation providing a noticeable speedup. The standalone CUDA version outperformed MPI due to GPU acceleration. In the hybrid CUDA + MPI mode, the additional MPI communication overhead introduced a slight slowdown compared to CUDA alone, though the difference remained small.

## Dependencies

**Requirements:** C++17, FFmpeg, OpenCV 4, OpenMPI, CUDA (optional)

## Build
```bash
cd serial-version && make
cd ../mpi-version && make
cd ../cuda_mpi-version && make
