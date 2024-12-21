#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <algorithm>
#include "util.h"

#define THREADS_PER_BLOCK 1024

__global__ void convert_rgb_to_gray(unsigned char* in_buf, unsigned char* out_buf, int tot_pixels) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < tot_pixels) {
        int rgb_idx = tid_x * 3;
        unsigned char r = in_buf[rgb_idx];
        unsigned char g = in_buf[rgb_idx + 1];
        unsigned char b = in_buf[rgb_idx + 2];
        out_buf[tid_x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input-image-path> <output-image-path>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    Img* img = read_jpeg_img(filename);
    if (!img) {
        return 1;
    }

    int out_pixels = img->tot_pixels / 3;
    int req_blocks = (out_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    unsigned char *out_buf = (unsigned char*)malloc(out_pixels * sizeof(unsigned char));
    unsigned char *out_buf_cu, *in_buf_cu;

    cudaMalloc(&out_buf_cu, out_pixels * sizeof(unsigned char));
    cudaMalloc(&in_buf_cu, img->tot_pixels * sizeof(unsigned char));

    cudaMemcpy(in_buf_cu, img->pixel_buf, img->tot_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threads(THREADS_PER_BLOCK, 1);
    dim3 blocks(req_blocks, 1);

    convert_rgb_to_gray<<<blocks, threads>>>(in_buf_cu, out_buf_cu, out_pixels);
    cudaMemcpy(out_buf, out_buf_cu, out_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_jpepg_img(argv[2], img->height, img->width, 1, out_buf);
    free(img->pixel_buf);
    free(img);
    free(out_buf);
    cudaFree(in_buf_cu);
    cudaFree(out_buf_cu);

    printf("saved image to %s", argv[2]);

    return 0;
}
