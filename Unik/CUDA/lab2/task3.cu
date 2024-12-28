
#include "common.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifndef TILE_SIZE
# define TILE_SIZE 16
#endif

#ifndef VERIFY
# define VERIFY 0
#endif

void transpose_cpu(const int *src, int *dst, int n) {
   for (int y = 0; y < n; y++) {
       for (int x = 0; x < n; x++) {
           dst[y * n + x] = src[x * n + y];
       }
   }
}

__global__ void transpose_global(const int *src, int *dst, int n) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < n && y < n) {
       dst[y * n + x] = src[x * n + y];
   }
}

__global__ void transpose_shared(const int *src, int *dst, int n) {
   __shared__ int tile[TILE_SIZE][TILE_SIZE];

   int x = blockIdx.x * TILE_SIZE + threadIdx.x;
   int y = blockIdx.y * TILE_SIZE + threadIdx.y;

   if (x < n && y < n) {
       tile[threadIdx.y][threadIdx.x] = src[y * n + x];
   }
   __syncthreads();
   
   x = blockIdx.y * TILE_SIZE + threadIdx.x;
   y = blockIdx.x * TILE_SIZE + threadIdx.y;

   if (x < n && y < n) {
       dst[y * n + x] = tile[threadIdx.x][threadIdx.y];
   }
}


void verifyResults(const int* A, const int* B, int N) {
   for (int i = 0; i < N; i++) {
       for (int j = 0; j < N; j++) {
           if (B[j * N + i] != A[i * N + j]) {
               printf("Verification failed at element (%d, %d)\n", i, j);
               return;
           }
       }
   }
   printf("OK\n");
}

int main(int argc, char **argv) {
#ifdef SIZE
    int N = SIZE;
#else
    int N = atoi(argv[1]);
#endif

    int* h_A, * h_B, * h_C;  
    int* d_A, * d_B;        

    h_A = (int*)malloc(N * N * sizeof(int));
    h_B = (int*)malloc(N * N * sizeof(int)); 
    h_C = (int*)malloc(N * N * sizeof(int)); 


    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
       h_A[i] = rand() % 100 + 1;  
    }

    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));

    cudaEvent_t start, stop;
    float elapsedTime;


    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float time;
    if (OP == 'c') {
        clock_t cpuStart = clock();
        transpose_cpu(h_A, h_C, N);
        clock_t cpuEnd = clock();
        time = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        
        if (VERIFY == 1) {
            verifyResults(h_A, h_C, N);
        }
    } else {
        if (OP == 'g') {
            transpose_global<<<grid, block>>>(d_A, d_B, N);
        } else if (OP == 's') {
            transpose_shared<<<grid, block>>> (d_A, d_B, N);
        }
        check(cudaGetLastError(), "kernel run fail");

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        time = elapsedTime;

        if (VERIFY == 1 && VERIFY == 1) {
            cudaMemcpy(h_C, d_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);
            verifyResults(h_A, h_C, N);
        }
    }
    
    std::cout << "size: " << N << ", time: " << time << " ms\n";

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
