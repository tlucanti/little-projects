
#include "common.hpp"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define N (512)

__global__ void kernel (float *a, float *b, float *c)
{
   long idx = threadIdx.x + blockIdx.x * blockDim.x;
   c[idx] = a[idx] + b[idx];
}

int main()
{
    float a[N] = {};
    float b[N] = {};
    float c[N] = {};

    float *ca, *cb, *cc;

    for (long i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = i * i;
    }

    check(cudaSetDevice(0), "set device fail");

    check(cudaMalloc(&ca, N * sizeof(float)), "malloc fail");
    check(cudaMalloc(&cb, N * sizeof(float)), "malloc fail");
    check(cudaMalloc(&cc, N * sizeof(float)), "malloc fail");

    std::cout << "alloc 1\n";
    check(cudaMemcpy(ca, a, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy fail");
    std::cout << "alloc 2\n";
    check(cudaMemcpy(cb, b, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy fail");

    timer_start();
    std::cout << "running\n";
    kernel<<<dim3(1, 1, 1), dim3(512, 1, 1)>>> (ca, cb, cc);
    float gpuTime = timer_stop();

    check(cudaMemcpy(c, cc, N * sizeof(float), cudaMemcpyDeviceToHost), "memcpy fail");

    check(cudaFree(ca), "free fail");
    check(cudaFree(cb), "free fail");
    check(cudaFree(cc), "free fail");

    for (long i = 0; i < N; i++) {
        if (i < 10 || i > N - 10) {
            printf("%f %f %f\n", a[i], b[i], c[i]);
        }
    }
    
    std::cout << "execution time: " << gpuTime << '\n';
}