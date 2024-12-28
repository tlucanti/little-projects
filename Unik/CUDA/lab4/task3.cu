#include <cuda_runtime.h>
#include <iostream>
#include "common.hpp"

__global__ void dummyKernel()
{
    __shared__ int buff[32];
    buff[threadIdx.x] = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    buff[threadIdx.x] += idx;
}

int main()
{
   int device;
   check(cudaGetDevice(&device), "get device");

   int maxThreadsPerBlock;
   check(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device), "get attribute");
   std::cout << "max threads per block: " << maxThreadsPerBlock << '\n';

   int blockSize = 256;
   int minGridSize;
   int occupancy;

   check(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dummyKernel, 0, 0), "occupancy block size");
   std::cout << "min grid size: " << minGridSize << '\n';
   std::cout << "optimal block size: " << blockSize << '\n';

   int smCount;
   check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, dummyKernel, blockSize, 0), "occupancy active blocks");
   check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device), "get attribute");

   std::cout << "max active blocks: " << occupancy * smCount << std::endl;

    return 0;
}
