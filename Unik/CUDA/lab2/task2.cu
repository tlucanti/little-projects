
#include "common.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifndef STREAMS
# errof "streams not defined"
#endif

#define N (1024 * 1024 * 512)
#define STREAM_SIZE (N / STREAMS)

__global__ void transpose(int *mem, int stream)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int *a = mem + 0 * N;
    int *b = mem + 1 * N;
    int *c = mem + 2 * N;
    
    i += stream * STREAM_SIZE;
    // printf(">>> %d: %d %d\n", i, a[i], b[i]);
    c[i] = a[i] * b[i];
}

int main()
{
    int *hostMem;
    int *devMem;

    cudaStream_t streams[STREAMS];

    check(cudaSetDevice(0), "set device fail");

    for (int i = 0; i < STREAMS; i++) {
        check(cudaStreamCreate(&streams[i]), "stream create fail");
    }

    check(cudaMallocHost(&hostMem, 3 * N * sizeof(int)), "host malloc fail");
    check(cudaMalloc(&devMem, 3 * N * sizeof(int)), "device malloc fail");

    for (int i = 0; i < N; i++) {
        hostMem[i] = i + 1;
        hostMem[i + N] = i * i;
    }

    timer_start();

    int *devA = devMem + 0 * N;
    int *devB = devMem + 1 * N;
    int *devC = devMem + 2 * N;
    int *hostA = hostMem + 0 * N;
    int *hostB = hostMem + 1 * N;
    int *hostC = hostMem + 2 * N;
    for (int i = 0; i < STREAMS; i++) {
        // std::cout << "transfering array A to stream " << i << std::endl;
        check(cudaMemcpyAsync(devA + i * STREAM_SIZE, hostA + i * STREAM_SIZE, STREAM_SIZE * sizeof(int), cudaMemcpyHostToDevice, streams[i]),
              "transfer array A failed");
        // std::cout << "transfering array B to stream " << i << std::endl;
        check(cudaMemcpyAsync(devB + i * STREAM_SIZE, hostB + i * STREAM_SIZE, STREAM_SIZE * sizeof(int), cudaMemcpyHostToDevice, streams[i]),
              "transfer array B failed");
    }

    for (int i = 0; i < STREAMS; i++) {
        transpose<<<N / 32 / STREAMS, 32, 0, streams[i]>>> (devMem, i);
        // std::cout << "running stream " << i << std::endl;
        check(cudaGetLastError(), "kernel run fail");
    }

    for (int i = 0; i < STREAMS; i++) {
        check(cudaMemcpyAsync(hostC + i * STREAM_SIZE, devC + i * STREAM_SIZE, STREAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost, streams[i]),
              "transfer device to host failed");
    }

    cudaDeviceSynchronize();
    std::cout << STREAMS << ", " << timer_stop() << std::endl;

    // for (int i = 0; i < min(N, 50); i++) {
    //     int *a = hostMem;
    //     int *b = hostMem + N;
    //     int *c = hostMem + 2 * N;
    //     printf("%d: %d %d %d\n", i, a[i], b[i], c[i]);
    // }

    check(cudaFreeHost(hostMem), "host free fail");
    check(cudaFree(devMem), "device free fail");

    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}