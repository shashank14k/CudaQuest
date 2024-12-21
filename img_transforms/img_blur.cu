#include <stdio.h>
#include <jpeglib.h>
#include "util.h"

#define THREADS_PER_BLOCK 32

__constant__ int mask[225]; // Constant memory is limited to 64KB, here I have defined it to hold a max kernel size of 15x15

__global__ void blur_image_channel(unsigned char *in_buf, unsigned char *out_buf, int ht, int wd, int kx, int ky, int channel, int ksum)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int pad_x = kx / 2;
    int pad_y = ky / 2;
    int start_x = tidx - pad_x;
    int start_y = tidy - pad_y;
    int tmp = 0;
    for (int i = 0; i < ky; i++)
    {
        for (int j = 0; j < kx; j++)
        {
            int px = start_x + j;
            int py = start_y + i;
            if (px >= 0 && py >= 0 && px < wd && py < ht)
            {
                int channel_idx = (py * wd * 3) + (px * 3) + channel;
                tmp += mask[i * kx + j] * in_buf[channel_idx];
            }
        }
    }
    // if (tidx < wd && tidy < ht){
    tmp = max(0, min(255, tmp / ksum));
    int out_idx = (tidy * wd * 3) + (tidx * 3) + channel;
    out_buf[out_idx] = static_cast<unsigned char>(tmp);
    //}
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <input-image-path> <output-image-path> <kernel-size>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    Img *img = read_jpeg_img(filename);
    if (!img)
    {
        return 1;
    }
    int kernel_size = 3;
    if (argc > 3)
    {
        kernel_size = atoi(argv[3]);
    }
    int tot_ker_size = kernel_size * kernel_size;
    int mask_cpu[225] = {0};
    int ksum = tot_ker_size;

    for (int i = 0; i < tot_ker_size; i++)
    {
        mask_cpu[i] = 1;
    }

    unsigned char *out_buf = (unsigned char *)malloc(img->tot_pixels * sizeof(unsigned char));
    unsigned char *out_buf_cu, *in_buf_cu;

    cudaMalloc(&out_buf_cu, img->tot_pixels * sizeof(unsigned char));
    cudaMalloc(&in_buf_cu, img->tot_pixels * sizeof(unsigned char));

    cudaMemcpy(in_buf_cu, img->pixel_buf, img->tot_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, mask_cpu, tot_ker_size * sizeof(int));

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((img->width + threads.x - 1) / threads.x, (img->height + threads.y - 1) / threads.y);

    blur_image_channel<<<blocks, threads>>>(in_buf_cu, out_buf_cu, img->height, img->width, kernel_size, kernel_size, 0, ksum);
    cudaDeviceSynchronize();
    blur_image_channel<<<blocks, threads>>>(in_buf_cu, out_buf_cu, img->height, img->width, kernel_size, kernel_size, 1, ksum);
    cudaDeviceSynchronize();
    blur_image_channel<<<blocks, threads>>>(in_buf_cu, out_buf_cu, img->height, img->width, kernel_size, kernel_size, 2, ksum);
    cudaDeviceSynchronize();
    cudaMemcpy(out_buf, out_buf_cu, img->tot_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_jpepg_img(argv[2], img->height, img->width, 3, out_buf);
    free(img->pixel_buf);
    free(img);
    free(out_buf);
    cudaFree(in_buf_cu);
    cudaFree(out_buf_cu);

    printf("saved image to %s\n", argv[2]);

    return 0;
}
