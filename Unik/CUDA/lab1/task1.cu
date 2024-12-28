
#include "common.hpp"

#include <stdio.h>
#include <iostream>

#define N (1024 * 1024)

__global__ void kernel (float *data)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float x = 2.0f * 3.1415926f * (float)idx / (float)N;
   data[idx] = sinf(sqrtf(x));
}

void print_gpu_info()
{
    int deviceCount;
    cudaDeviceProp devProp;

    check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount fail");
    printf("Found %d devices\n", deviceCount);

    for ( int device = 0; device < deviceCount; device++ ) {
        check(cudaGetDeviceProperties(&devProp, device), "cudaGetDeviceProperties fail");

        printf("Device %d\n", device );
        printf("Compute capability     : %d.%d\n", devProp.major, devProp.minor);
        printf("Name                   : %s\n", devProp.name);
        printf("Total Global Memory    : %lu\n", devProp.totalGlobalMem);
        printf("Shared memory per block: %lu\n", devProp.sharedMemPerBlock);
        printf("Registers per block    : %d\n", devProp.regsPerBlock);
        printf("Warp size              : %d\n", devProp.warpSize);
        printf("Max threads per block  : %d\n", devProp.maxThreadsPerBlock);
        printf("Total constant memory  : %lu\n", devProp.totalConstMem);

        printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
               devProp.maxThreadsDim[0],
               devProp.maxThreadsDim[1],
               devProp.maxThreadsDim[2]);

        printf("Max grid size: x = %d, y = %d, z = %d\n",
               devProp.maxGridSize[0],
               devProp.maxGridSize[1],
               devProp.maxGridSize[2]);
    }
}


int main()
{
	float *a = (float *)malloc(N * sizeof(float));
    float *dev = NULL;
    float gpuTime = 0.0f;

    if (a == NULL)
        printf("host malloc fail\n");

    print_gpu_info();

    check(cudaSetDevice(0), "set device fail");

    check(cudaMalloc(&dev, N * sizeof (float)), "malloc fail");

    timer_start();
    kernel<<<dim3(32, 32, 1), dim3(32, 32, 1)>>> (dev);
    check(cudaGetLastError(), "kernel run fail");
    gpuTime = timer_stop();
    
    check(cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost), "memcpy fail");
    check(cudaFree(dev), "free fail");

    for (int idx = 0; idx < min(10, N); idx++) {
       printf("a[%d] = %.5f\n", idx, a[idx]);
    }
    printf("...\n");
    
    std::cout << "execution time: " << gpuTime << '\n';

    free(a);
    return 0;
}

