
#include "common.hpp"

#include <stdio.h>
#include <iostream>
#include <climits>
#include <algorithm>
#include <vector>

#define MAX_VAL 256
#define BLOCK_SIZE MAX_VAL

__global__ void computeHistogram(int *array, int *hist, int arraySize) {
    __shared__ int buffer[MAX_VAL];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ti = threadIdx.x;
    
    buffer[ti] = 0;
    __syncthreads();

    if (idx < arraySize) {
        atomicAdd(&buffer[array[idx]], 1);
    }
    __syncthreads();
    
    atomicAdd(&hist[ti], buffer[ti]);
}

int main(int argc, char **argv)
{
    const int N = 1000000; 
    const int blockSize = BLOCK_SIZE;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    srand(time(nullptr));
    
    /* fill array with random values from 0 to MAX_VAL - 1 */
    std::vector<int> hostArr(N);
    std::vector<int> hostHist(MAX_VAL);
    std::vector<int> trueHist(MAX_VAL);
    for (int i = 0; i < N; i++) {
        hostArr[i] = rand() % MAX_VAL;
        trueHist.at(hostArr[i]) += 1;
    }
    

    /* allocate and transfer device memory */
    int* dev;
    int *hist;
    
    check(cudaMalloc(&dev, N * sizeof(int)), "malloc");
    check(cudaMalloc(&hist, MAX_VAL * sizeof(int)), "malloc");
    
    check(cudaMemcpy(dev, hostArr.data(), hostArr.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    check(cudaMemcpy(hist, hostHist.data(), hostHist.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy");

    /* run parallel / linear kernel */
    timer_start();
    computeHistogram<<<numBlocks, blockSize>>>(dev, hist, N);
    check(cudaGetLastError(), "kernel run fail");
    
    /* transfer data back to host */
    check(cudaMemcpy(hostHist.data(), hist, hostHist.size() * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    float ms = timer_stop();

    /* print answer */
    std::cout << "time: " << ms << " ms\n";
    std::cout <<   "expected : real\n";
    for (int i = 0; i < MAX_VAL; i++) {
        std::cout << trueHist.at(i) << " : " << hostHist.at(i);
        if (trueHist.at(i) != hostHist.at(i)) {
            std::cout << " [FAIL]";
        }
        std::cout << '\n';
    }
    std::cout << std::endl;

    /* cleanup */
    cudaFree(dev);
    cudaFree(hist);
    return 0;
}
