#include <float.h>
#include <math.h>
#include <stdio.h>
#include "transforms.cuh"
#include "../include/util.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void rgb_to_yuv(float *in_buf, float *out_buf, int tot_pixels)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rgb_idx = tid * 3;
    if (rgb_idx < tot_pixels)
    {
        // in_buf is rgb
        float Y = 0.299 * in_buf[rgb_idx] + 0.587 * in_buf[rgb_idx + 1] + 0.114 * in_buf[rgb_idx + 2];
        float U = 0.492 * (in_buf[rgb_idx + 2] - Y);
        float V = 0.877 * (in_buf[rgb_idx] - Y);
        out_buf[rgb_idx] = Y;
        out_buf[rgb_idx + 1] = U;
        out_buf[rgb_idx + 2] = V;
    }
}

// Min-Reduce algorithm.
__global__ void find_minimum(float *buf, float *buf_r, int stride, int buf_size)
{
    /*
        A simple way would be for each thread to load data from global memory at each of the log(n) step,
        and write back the data. Instead, threads can load data to the shared memory once, and continue with subsequent
        computation without having to read/write to global mem.
    */
    // extern __shared__ double reduced_min[];
    __shared__ float reduced_min[MAX_THREADS_PER_BLOCK];
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx * stride < buf_size)
    {
        reduced_min[threadIdx.x] = buf[tidx * stride];
    }
    else
    {
        reduced_min[threadIdx.x] = FLT_MAX;
    }
    __syncthreads();

    for (int r = 1; r < blockDim.x; r *= 2)
    {
        int buf_idx = 2 * r * threadIdx.x;
        if (tidx < buf_size && buf_idx < blockDim.x)
        {
            reduced_min[buf_idx] = fmin(reduced_min[buf_idx], reduced_min[buf_idx + r]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        buf_r[blockIdx.x] = reduced_min[0];
    }
}

// Max-Reduce algorithm.
__global__ void find_maximum(float *buf, float *buf_r, int stride, int buf_size)
{
    /*
        A simple way would be for each thread to load data from global memory at each of the log(n) step,
        and write back the data. Instead, threads can load data to the shared memory once, and continue with subsequent
        computation without having to read/write to global mem.
    */
    // extern __shared__ float reduced_min[];
    __shared__ float reduced_max[MAX_THREADS_PER_BLOCK];
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx * stride < buf_size)
    {
        reduced_max[threadIdx.x] = buf[tidx * stride];
    }
    else
    {
        reduced_max[threadIdx.x] = FLT_MIN;
    }
    __syncthreads();

    for (int r = 1; r < blockDim.x; r *= 2)
    {
        int buf_idx = 2 * r * threadIdx.x;
        if (buf_idx < blockDim.x - r)
        {
            reduced_max[buf_idx] = fmax(reduced_max[buf_idx], reduced_max[buf_idx + r]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        buf_r[blockIdx.x] = reduced_max[0];
    }
}

__global__ void compute_image_histogram_naive(const float *img, const int img_size, const int stride, float min_bin_val, float max_bin_val, int *d_bins, const int numBins)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int pix_idx = tid * stride;
    if (pix_idx < img_size)
    {
        float normalized_val = (img[pix_idx] - min_bin_val) / (max_bin_val - min_bin_val);
        int bin = static_cast<int>(normalized_val * numBins);
        bin = max(0, min(bin, numBins - 1));
        atomicAdd(&(d_bins[bin]), 1); // Number of global atomic adds scale to size of img buffer
    }
}

__global__ void compute_image_histogram_faster(const float *img, const int img_size, const int stride, float min_bin_val, float max_bin_val, int *d_bins, const int numBins)
{
    extern __shared__ int s_res[];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIdx.x < numBins)
    {
        s_res[threadIdx.x] = 0;
    }
    __syncthreads();

    int pix_idx = tid * stride;
    if (pix_idx < img_size)
    {
        float normalized_val = (img[pix_idx] - min_bin_val) / (max_bin_val - min_bin_val);
        int bin = static_cast<int>(normalized_val * numBins);
        bin = max(0, min(bin, numBins - 1));
        atomicAdd(&(s_res[bin]), 1);
    }

    __syncthreads();

    // Number of global atomic adds equals number of bins times the number of threadblocks
    if (threadIdx.x < numBins)
    {
        atomicAdd(&d_bins[threadIdx.x], s_res[threadIdx.x]);
    }
}

__global__ void compute_histogram_distribution(const int *dbins, float *cdf, int numBins, int normalize_const)
{
    /*
       A simple way would be for each thread to load data from global memory at each of the log(n) step,
       and write back the data. Instead, threads can load data to the shared memory once, and continue with subsequent
       computation without having to read/write to global mem.
   */
    // extern __shared__ double reduced_min[];
    extern __shared__ int s_dbins[];
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < numBins)
    {
        s_dbins[threadIdx.x] = dbins[tidx];
    }
    else
    {
        s_dbins[threadIdx.x] = 0;
    }
    __syncthreads();
    // Should r be limited to numBins?
    for (int r = 1; r < blockDim.x; r *= 2)
    {
        int prev_val = 0;
        if (tidx >= r)
        {
            prev_val = s_dbins[tidx - r];
        }
        __syncthreads();
        s_dbins[threadIdx.x] += prev_val;
        __syncthreads();
    }
    if (tidx < numBins)
    {
        cdf[tidx] = s_dbins[threadIdx.x] * (1.f / normalize_const);
    }
}

__global__ void equalize_hist(unsigned char *out_buf, float *in_buf, float *cdf, float min_bin_val, float max_bin_val, const int numBins, int tot_pixels)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lum_idx = tid * 3;
    if (lum_idx < tot_pixels)
    {
        float normalized_val = (in_buf[lum_idx] - min_bin_val) / (max_bin_val - min_bin_val);
        int bin = static_cast<int>(normalized_val * numBins);
        bin = max(0, min(bin, numBins - 1));
        float lum_val = cdf[bin];
        float scale_factor = lum_val / (in_buf[lum_idx] + 1e-6f);
        float U = in_buf[lum_idx + 1] * scale_factor;
        float V = in_buf[lum_idx + 2] * scale_factor;
        float R = fmaxf(0, fminf(lum_val + 1.402f * V, 1));
        float G = fmaxf(0, fminf(lum_val - 0.344136f * U - 0.714136f * V, 1));
        float B = fmaxf(0, fminf(lum_val + 1.772f * U, 1));

        out_buf[tid * 3] = static_cast<unsigned char>(R * 255);
        out_buf[tid * 3 + 1] = static_cast<unsigned char>(G * 255);
        out_buf[tid * 3 + 2] = static_cast<unsigned char>(B * 255);
    }
}

extern "C" float *convert_rgb_to_yuv(float *buf, int rows, int cols, int channels)
{
    size_t tot_pixels = rows * cols * channels;
    float *out_buf = (float *)malloc(sizeof(float) * tot_pixels);
    float *cu_inbuf, *cu_outbuf;
    cudaMalloc(&cu_inbuf, tot_pixels * sizeof(float));
    cudaMalloc(&cu_outbuf, tot_pixels * sizeof(float));
    cudaMemcpy(cu_inbuf, buf, tot_pixels * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = max(1, (tot_pixels + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
    rgb_to_yuv<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(cu_inbuf, cu_outbuf, tot_pixels);

    cudaMemcpy(out_buf, cu_outbuf, tot_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cu_inbuf);
    cudaFree(cu_outbuf);
    return out_buf;
}

extern "C" unsigned char *equalize_histogram(float *buf, int rows, int cols, int channels, int n_bins)
{
    size_t tot_pixels = rows * cols * channels;
    float *cu_inbuf, *cu_outbuf;
    cudaMalloc(&cu_inbuf, tot_pixels * sizeof(float));
    cudaMalloc(&cu_outbuf, tot_pixels * sizeof(float));
    cudaMemcpy(cu_inbuf, buf, tot_pixels * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = max(1, (tot_pixels + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
    rgb_to_yuv<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(cu_inbuf, cu_outbuf, tot_pixels);

    // Compute minimum and maxiumum of luminance channel
    float *min_cu_buf, *max_cu_buf; // buffers to hold min-max values for each thread.
    float min_lum, max_lum;
    cudaMalloc(&min_cu_buf, num_blocks * sizeof(float));
    cudaMalloc(&max_cu_buf, num_blocks * sizeof(float));
    find_maximum<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(cu_outbuf, max_cu_buf, 3, tot_pixels);
    find_maximum<<<1, MAX_THREADS_PER_BLOCK>>>(max_cu_buf, max_cu_buf, 1, num_blocks);
    find_minimum<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(cu_outbuf, min_cu_buf, 3, tot_pixels);
    find_minimum<<<1, MAX_THREADS_PER_BLOCK>>>(min_cu_buf, min_cu_buf, 1, num_blocks);
    cudaMemcpy(&min_lum, min_cu_buf, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_lum, max_cu_buf, sizeof(float), cudaMemcpyDeviceToHost);
    // Histogram bin
    int *dbins;
    float *cdf;

    cudaMalloc(&dbins, n_bins * sizeof(int));
    cudaMemset(dbins, 0, n_bins * sizeof(int));
    cudaMalloc(&cdf, n_bins * sizeof(float));
    cudaMemset(cdf, 0, n_bins * sizeof(float));
    compute_image_histogram_faster<<<num_blocks, MAX_THREADS_PER_BLOCK, n_bins>>>(cu_outbuf, tot_pixels, 3, min_lum, max_lum, dbins, n_bins);
    compute_histogram_distribution<<<1, n_bins, n_bins>>>(dbins, cdf, n_bins, tot_pixels / 3);

    // Out RGB BUF
    unsigned char *rgb_buf, *rgb_buf_cu;
    rgb_buf = (unsigned char *)malloc(sizeof(unsigned char) * tot_pixels);
    cudaMalloc(&rgb_buf_cu, sizeof(int) * tot_pixels);
    equalize_hist<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(rgb_buf_cu, cu_outbuf, cdf, min_lum, max_lum, n_bins, tot_pixels);
    cudaMemcpy(rgb_buf, rgb_buf_cu, sizeof(unsigned char) * tot_pixels, cudaMemcpyDeviceToHost);

    // Free CUDA bufs
    cudaFree(cu_inbuf);
    cudaFree(cu_outbuf);
    cudaFree(dbins);
    cudaFree(cdf);
    cudaFree(max_cu_buf);
    cudaFree(min_cu_buf);

    // Free CPU bufs
    return rgb_buf;
}