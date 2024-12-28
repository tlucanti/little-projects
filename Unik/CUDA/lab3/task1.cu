
#include "common.hpp"

#include <stdio.h>
#include <iostream>
#include <climits>
#include <algorithm>
#include <vector>

#ifndef VERIFY
# define VERIFY 0
#endif

// s = 1
// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
// ^=^=^=^ ^=^=^=^ ^=^=^=^ ^=^=^=^ ^
//
// s = 2
// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
// ^-------^-------^-------^       ^
//
// s = 4
// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
// ^-------'       ^-------'       ^
//
// s = 8
// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
// ^---------------'               ^
//
// s = 16
// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6
// ^-------------------------------'

#define BLOCK_SIZE 512

/* do one reduction step */
__global__ void doReduceMin(int *array, int arraySize, int stride) {
    __shared__ int buffer[BLOCK_SIZE];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ti = threadIdx.x;
    
    /* fill shared buffer from global memory */
    buffer[ti] = INT_MAX;
    if (idx * stride < arraySize) {
        buffer[ti] = array[idx * stride];
    }
    __syncthreads();
    
    /* do per block reduction */
    int s = 1;
    while (s <= BLOCK_SIZE / 2) {
        if (ti % (s * 2) == 0 and ti + s < BLOCK_SIZE) {
            buffer[ti] = min(buffer[ti], buffer[ti + s]);
        }
        s *= 2;
        __syncthreads();
    }
    
    /* save block minimum to global memory */
    if (ti == 0) {
        if (idx * stride < arraySize) {
            array[idx * stride] = buffer[0];
        }
    }
}

/* do full reduction (save answer in array[0]) */
void reduceMin(int numBlocks, int blockSize, int *array, int arraySize) {
    for (int stride = 1; stride <= arraySize / 2; stride *= BLOCK_SIZE) {
        doReduceMin<<<numBlocks, blockSize>>>(array, arraySize, stride);
        check(cudaGetLastError(), "kernel run fail");
    }
}

/* do reduction with atomics (save answer in array[0]) */
__global__ void linearMin(int* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
       atomicMin(&input[0], input[idx]);
    }
}

int main(int argc, char **argv)
{
    const int N = 1000000; 
    const int blockSize = BLOCK_SIZE;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    int true_ans = INT_MAX;
    int ans;
    
    /* fill array with random values from -1e8 to 1e8 */
    srand(time(nullptr));
    std::vector<int> host(N);
    for (int i = 0; i < N; i++) {
        host[i] = rand() % 200000000 - 100000000;
        true_ans = min(true_ans, host[i]);
    }

    /* allocate and transfer device memory */
    int* dev;
    check(cudaMalloc(&dev, N * sizeof(int)), "malloc");
    check(cudaMemcpy(dev, host.data(), N * sizeof(int), cudaMemcpyHostToDevice), "memcpy");

    /* run parallel / linear kernel */
    timer_start();
    if (OP == 'r') {
        reduceMin(numBlocks, blockSize, dev, N);
    } else if (OP == 'l') {
        linearMin<<<numBlocks, blockSize>>>(dev, N);
    }
    check(cudaGetLastError(), "kernel run fail");
    
    /* transfer data back to host */
    check(cudaMemcpy(&ans, dev, sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    float ms = timer_stop();

    /* print answer */
    if (OP =='r') {
        std::cout << "minimum (reduce): ";
    } else if (OP == 'l') {
        std::cout << "minimum (linear): ";
    }
    std::cout << ans << ", time: " << ms << " ms\n";
    std::cout << "expeted value: " << true_ans << std::endl;

    /* cleanup */
    cudaFree(dev);
    return 0;
}
