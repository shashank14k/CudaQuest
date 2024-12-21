#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.h"

#define N_THREADS 256

__global__ void conv1d(const double *vec, const double *mask, double *res, int vec_len, int mask_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = mask_len / 2;
    int start = tid - radius;
    double tmp = 0;

    for (int j = 0; j < mask_len; j++)
    {
        if (start + j >= 0 && start + j < vec_len)
        {
            tmp += vec[start + j] * mask[j];
        }
    }
    if (tid < vec_len)
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
    double *vec = generate_random_vector(size);
    double *mask = generate_random_vector(mask_size);

    double *cu_vec1, *cu_vec2, *cu_mask;
    cudaMalloc(&cu_vec1, size * sizeof(double));
    cudaMalloc(&cu_vec2, size * sizeof(double));
    cudaMalloc(&cu_mask, mask_size * sizeof(double));

    cudaMemcpy(cu_vec1, vec, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_mask, mask, mask_size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threads(N_THREADS, 1);
    dim3 blocks((size + threads.x - 1) / threads.x, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv1d<<<blocks, threads>>>(cu_vec1, cu_mask, cu_vec2, size, mask_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_taken = 0;
    cudaEventElapsedTime(&time_taken, start, stop);
    printf("Convolution completed in %f mili-seconds\n", time_taken);
    double *vec2 = (double *)(malloc(sizeof(double) * size));
    cudaMemcpy(vec2, cu_vec2, size * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Verifying results\n");
    verify_cuda_1d_conv(vec, mask, vec2, size, mask_size);
    printf("MATRIX MULTIPLICATION RAN SUCCESSFULLY AND VERIFIED ON CPU\n");
    free(vec2);
    // Free memory on device
    cudaFree(cu_vec1);
    cudaFree(cu_vec2);
    cudaFree(cu_mask);

    free(vec);
    free(mask);
    return 0;
}