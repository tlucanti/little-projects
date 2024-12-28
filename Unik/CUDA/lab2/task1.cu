
#include "common.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N (1024 * 512)

#ifndef KERNEL
# error "kernel not defined"
#endif

__global__ void good(int *a, int *b, int *c)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   c[i] = a[i] * b[i];
}

__global__ void bad(int *a, int *b, int *c)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   i = (i * i) % N;

   c[i] = a[i] * b[i];
}

int main()
{
    int a[N] = {};
    int b[N] = {};
    int c[N] = {};

    int *ca, *cb, *cc;

    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = i * i;
    }

    check(cudaSetDevice(0), "set device fail");

    check(cudaMalloc(&ca, N * sizeof(int)), "malloc fail");
    check(cudaMalloc(&cb, N * sizeof(int)), "malloc fail");
    check(cudaMalloc(&cc, N * sizeof(int)), "malloc fail");

    check(cudaMemcpy(ca, a, N * sizeof(int), cudaMemcpyHostToDevice), "memcpy fail");
    check(cudaMemcpy(cb, b, N * sizeof(int), cudaMemcpyHostToDevice), "memcpy fail");

    timer_start();
    KERNEL<<<N / 32, 32>>> (ca, cb, cc);
    float t = timer_stop();

    check(cudaMemcpy(c, cc, N * sizeof(int), cudaMemcpyDeviceToHost), "memcpy fail");

    check(cudaFree(ca), "free fail");
    check(cudaFree(cb), "free fail");
    check(cudaFree(cc), "free fail");

    printf("exec time: %g\n", t);
    for (int i = 0; i < min(N, 20); i++) {
        printf("%d: %d %d %d\n", i, a[i], b[i], c[i]);
    }
    std::cout << "execution time: " << t << '\n';
}