#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.h"

#define N_THREADS 256

__global__ void conv1d(const double *vec, const double *mask, double *res, int vec_len, int mask_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double cache[];
    int radius = mask_len / 2;
    int extra = threadIdx.x + blockDim.x;
    int n_padded = 2 * radius + blockDim.x;

    // Load normal elements
    if (tid < vec_len)
    {
        cache[threadIdx.x] = vec[tid];
    }
    else
    {
        cache[threadIdx.x] = 0.0;
    }

    // Load extra elements for convolution
    if (extra < n_padded && (blockDim.x * blockIdx.x + extra) < vec_len)
    {
        cache[extra] = vec[blockDim.x * blockIdx.x + extra];
    }
    __syncthreads();
    double tmp = 0;
    for (int j = 0; j < mask_len; j++)
    {
        tmp += cache[threadIdx.x + j] * mask[j];
    }
    if (tid < vec_len - radius)
    {
        res[tid] = tmp;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Error: Required vector and mask sizes for convolution. Usage: ./conv1_n <vector-size> <mask-size>\n");
        exit(EXIT_FAILURE); // Exit the program with a failure status
    }
    int size = atoi(argv[1]);
    int mask_size = atoi(argv[2]);

    // Generate random matrices
    int pad = mask_size / 2;
    int padded_size = 2 * pad + size;

    double *vec = generate_random_vector(padded_size);
    double *mask = generate_random_vector(mask_size);

    for (int k = 0; k < padded_size; k++)
    {
        if (k < pad || k >= (size + pad))
        {
            vec[k] = 0.0;
        }
    }
    double *cu_vec1, *cu_vec2, *cu_mask;
    cudaMalloc(&cu_vec1, padded_size * sizeof(double));
    cudaMalloc(&cu_vec2, size * sizeof(double));
    cudaMalloc(&cu_mask, mask_size * sizeof(double));

    cudaMemcpy(cu_vec1, vec, padded_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_mask, mask, mask_size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threads(N_THREADS, 1);
    dim3 blocks((size + threads.x - 1) / threads.x, 1);
    int shmem_size = (N_THREADS + pad * 2) * sizeof(double);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv1d<<<blocks, threads, shmem_size>>>(cu_vec1, cu_mask, cu_vec2, padded_size, mask_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_taken = 0;
    cudaEventElapsedTime(&time_taken, start, stop);
    printf("Convolution completed in %f mili-seconds\n", time_taken);

    // Verify results with cpu
    double *vec2 = (double *)(malloc(sizeof(double) * size));
    cudaMemcpy(vec2, cu_vec2, size * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Verifying results\n");
    verify_cuda_1d_conv(vec + pad, mask, vec2, size, mask_size);
    printf("MATRIX MULTIPLICATION RAN SUCCESSFULLY AND VERIFIED ON CPU\n");
    free(vec2);

    // Free memory on device
    cudaFree(cu_vec1);
    cudaFree(cu_vec2);
    cudaFree(cu_mask);

    free(vec);
    free(mask);
}