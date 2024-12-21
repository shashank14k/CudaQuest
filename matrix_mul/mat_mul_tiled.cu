#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

#define N_THREADS 32

// Using shared_memory of size threads * threads to load global elements. Then running multiplication on the shared_memory chunk
__global__ void matrixMul(const double *mat1, const double *mat2, double *mat3, int row1, int col1, int row2, int col2)
{

    __shared__ double A[N_THREADS * N_THREADS];
    __shared__ double B[N_THREADS * N_THREADS];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Add to shared memory
    double sum = 0;
    for (int t = 0; t < (col1 - 1) / N_THREADS + 1; t++)
    {
        if (row < row1 && t * N_THREADS + threadIdx.x < col1)
        {
            A[(threadIdx.y * N_THREADS) + threadIdx.x] = mat1[row * col1 + (t * N_THREADS + threadIdx.x)];
        }
        else
        {
            A[(threadIdx.y * N_THREADS) + threadIdx.x] = 0.0;
        }
        if (col < col2 && t * N_THREADS + threadIdx.y < row2)
        {
            B[(threadIdx.y * N_THREADS) + threadIdx.x] = mat2[(t * N_THREADS + threadIdx.y) * col2 + col];
        }
        else
        {
            B[(threadIdx.y * N_THREADS) + threadIdx.x] = 0.0;
        }
        __syncthreads();
        for (int i = 0; i < N_THREADS; i++)
        {
            sum += A[(threadIdx.y * blockDim.x) + i] * B[(i * blockDim.x) + threadIdx.x];
        }
        __syncthreads();
    }
    if (row < row1 && col < col2)
    {
        mat3[row * col2 + col] = sum;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Error: Required matrix dimensions row1 col1 col2. Usage: ./mat_mul_tiled <arg1> <arg2> <arg3>\n");
        exit(EXIT_FAILURE);
    }
    int row1 = atoi(argv[1]);
    int col1 = atoi(argv[2]);
    int row2 = atoi(argv[2]);
    int col2 = atoi(argv[3]);
    // Generate random matrices
    double *mat1 = generate_random_double_matrix(row1, col1, 1, 0);
    double *mat2 = generate_random_double_matrix(row2, col2, 1, 0);
    double *mat3 = generate_random_double_matrix(row1, col2, 1, 0);

    double *cu_mat1, *cu_mat2, *cu_mat3;
    cudaMalloc(&cu_mat1, row1 * col1 * sizeof(double));
    cudaMalloc(&cu_mat2, row2 * col2 * sizeof(double));
    cudaMalloc(&cu_mat3, row1 * col2 * sizeof(double));

    cudaMemcpy(cu_mat1, mat1, row1 * col1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_mat2, mat2, row2 * col2 * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threads(N_THREADS, N_THREADS);
    dim3 blocks((col2 + threads.x - 1) / threads.x,
                (row1 + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMul<<<blocks, threads>>>(cu_mat1, cu_mat2, cu_mat3, row1, col1, row2, col2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_taken = 0;
    cudaEventElapsedTime(&time_taken, start, stop);
    printf("Matrix multiplication completed in %f mili-seconds\n", time_taken);
    cudaMemcpy(mat3, cu_mat3, row1 * col2 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Verifying results\n");
    verify_cuda_mat_mul(mat1, mat2, mat3, row1, col1, row2, col2);
    printf("MATRIX MULTIPLICATION RAN SUCCESSFULLY AND VERIFIED ON CPU\n");
    // Free memory on device
    cudaFree(cu_mat1);
    cudaFree(cu_mat2);
    cudaFree(cu_mat3);

    free(mat1);
    free(mat2);
    free(mat3);
    return 0;
}