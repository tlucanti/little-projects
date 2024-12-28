#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_image.h"
#include "common.hpp"
#include <iostream>

/* texture */
texture<unsigned char, 2, cudaReadModeElementType> tex;

#ifdef BLUR
/* Box Blur Kernel */
__global__ void boxBlurKernel(unsigned char* output, int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int sum = 0;
    int count = 0;

    /* scan window around each pixel */
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            /* check image boundaries */
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += tex2D(tex, nx + 0.5f, ny + 0.5f);
                count++;
            }
        }
    }

    /* average values from scanned window */
    output[y * width + x] = sum / count;
}
#else

/* Gaussian Blur Kernel */
__global__ void GaussianBlurKernel(unsigned char* pDst, int radius, float sigma_sq, int w, int h)
{ 
     int tidx = threadIdx.x + blockIdx.x * blockDim.x; 
     int tidy = threadIdx.y + blockIdx.y * blockDim.y; 

     /* check image boundaries */
     if (tidx < w && tidy < h) { 
          float r = 0;
          float  weight_sum = 0.0f; 
          float  weight = 0.0f;
         
          for (int ic = -radius; ic <= radius; ic++) { 
               weight = exp(-(ic * ic)/ sigma_sq); 
               r += tex2D(tex, tidx + 0.5f+ic, tidy + 0.5f) * weight; 
               weight_sum += weight; 
           } 
           
           /* normalize result */
           r /= weight_sum; 
           pDst[tidx + tidy * w] = (int)r;
     }
}
#endif
    
#ifdef BLUR
# define outputPath "./lab4/gpu_blur.pgm"
#else
# define outputPath "./lab4/gpu_gauss.pgm"
#endif

int main()
{
    const char* inputPath = "./lab4/lena.pgm";
    const int radius = 5;

    unsigned char* h_input = nullptr;
    unsigned char* h_output = nullptr;
    unsigned char* d_output = nullptr;
    unsigned int width, height;

    /* load image from file */
    if (!sdkLoadPGM<unsigned char>(inputPath, &h_input, &width, &height)) {
        throw std::runtime_error("image load failed");
    }

    h_output = new unsigned char[width * height];
    check(cudaMalloc(&d_output, width * height * sizeof(unsigned char)), "malloc");

    /* create texture */
    cudaArray* d_array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    check(cudaMallocArray(&d_array, &channelDesc, width, height), "malloc");
    check(cudaMemcpyToArray(d_array, 0, 0,
                            h_input, width * height * sizeof(unsigned char),
                            cudaMemcpyHostToDevice),
          "memcpy");

    /* bind array to texture */
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;
    check(cudaBindTextureToArray(tex, d_array, channelDesc), "bind");

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

#ifdef BLUR
    boxBlurKernel<<<gridDim, blockDim >>>(d_output, width, height, radius);
#else
    GaussianBlurKernel<<<gridDim, blockDim>>>(d_output, 10, 40, width, height);
#endif
    check(cudaGetLastError(), "kernel run fail");

    check(cudaMemcpy(h_output, d_output,
                     width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost),
          "memcpy");

    /* save image to file */
    sdkSavePGM(outputPath, h_output, width, height);
    std::cout << "Blurred image saved to: " << outputPath << std::endl;

    /* cleanup */
    delete[] h_output;
    cudaFreeArray(d_array);
    cudaFree(d_output);

    return 0;
}
