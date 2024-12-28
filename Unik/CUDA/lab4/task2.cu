#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_image.h"
#include "common.hpp"

/* texture */
texture<unsigned char, 2, cudaReadModeElementType> g_Texture;

#define lerp(a, b, t) ((a) + (t) * ((b) - (a)))

/* bilinear filter kernel */
__global__ void Bilinear(unsigned char* dest, float factor, unsigned int w, unsigned int h)
{
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;

   if (tidx < w && tidy < h) {
       float center = tidx / factor;
       unsigned int start = (unsigned int)center;
       unsigned int stop = start + 1;
       float t = center - start;

       unsigned char a = tex2D(g_Texture, tidy + 0.5f, start + 0.5f);
       unsigned char b = tex2D(g_Texture, tidy + 0.5f, stop + 0.5f);

       float linear = lerp(a, b, t);
       dest[tidx + tidy * w] = (int)(linear);
   }
}

/* load image from file */
void loadImage(const char* file, unsigned char** pixels, unsigned int* width, unsigned int* height)
{
   if (!sdkLoadPGM<unsigned char>(file, pixels, width, height)) {
       throw std::runtime_error("load image failed");
   }
}

/* save image to file */
void saveImage(const char* file, unsigned char* pixels, unsigned int width, unsigned int height)
{
   if (!sdkSavePGM(file, pixels, width, height)) {
       printf("Failed to save image: %s\n", file);
       exit(EXIT_FAILURE);
   }
}

int main()
{
   const char* input_file = "./lab4/lena.pgm";
   const char* output_file = "./lab4/gpu_scaled.pgm";

   unsigned char* h_pixels = NULL;
   unsigned int src_width, src_height;
   loadImage(input_file, &h_pixels, &src_width, &src_height);

   /* image scale 2 times */
   float scale_factor = 0.5f;

   unsigned int dst_width = (unsigned int)(src_width * scale_factor);
   unsigned int dst_height = (unsigned int)(src_height * scale_factor);

   unsigned char* h_output_pixels = new unsigned char[dst_width * dst_height];
   unsigned char* d_output_pixels;
   check(cudaMalloc(&d_output_pixels, dst_width * dst_height * sizeof(unsigned char)), "malloc");

   /* create texture array */
   cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();
   cudaArray* cu_array;
   check(cudaMallocArray(&cu_array, &channel_desc, src_width, src_height), "malloc");

   check(cudaMemcpyToArray(cu_array, 0, 0,
                           h_pixels, src_width * src_height * sizeof(unsigned char),
                           cudaMemcpyHostToDevice),
         "memcpy");

   /* bind texture array to gpu memory */
   g_Texture.addressMode[0] = cudaAddressModeClamp;
   g_Texture.addressMode[1] = cudaAddressModeClamp;
   g_Texture.filterMode = cudaFilterModePoint;
   g_Texture.normalized = false;
   check(cudaBindTextureToArray(g_Texture, cu_array, channel_desc), "bind");

   /* run kernel */
   dim3 block(16, 16);
   dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
   Bilinear<<<grid, block>>>(d_output_pixels, scale_factor, dst_width, dst_height);
   check(cudaGetLastError(), "kernel run fail");

   check(cudaMemcpy(h_output_pixels, d_output_pixels,
                    dst_width * dst_height * sizeof(unsigned char), cudaMemcpyDeviceToHost),
         "memcpy");

   /* save image */
   saveImage(output_file, h_output_pixels, dst_width, dst_height);

   /* cleanup */
   cudaUnbindTexture(g_Texture);
   cudaFreeArray(cu_array);
   cudaFree(d_output_pixels);
   delete h_output_pixels;

   std::cout << "Image saved to " << output_file << std::endl;
   return 0;
}
