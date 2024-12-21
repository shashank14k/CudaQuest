#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const double EPSILON = 1e-9;
void fillrandomMatrix(double *matrix, int size, double max, double min)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = (double)rand() / (double)(RAND_MAX / (max - min)) + min;
    }
}

double *generate_random_double_matrix(int row, int col, double max, double min)
{
    double *mat = (double *)malloc(sizeof(double) * row * col);
    fillrandomMatrix(mat, row * col, max, min);
    return mat;
}

void verify_cuda_mat_mul(double *mat1, double *mat2, double *cuda_mat, int row1, int col1, int row2, int col2)
{
    for (int i = 0; i < row1; i++)
    {
        for (int j = 0; j < col2; j++)
        {
            double sum = 0;
            for (int k = 0; k < col1; k++)
            {
                sum += mat1[i * col1 + k] * mat2[k * col2 + j];
            }
            if (fabs(cuda_mat[i * col2 + j] - sum) > EPSILON)
            {
                fprintf(stderr, "Verification failed at index (%d, %d): GPU result = %f, CPU result = %f\n", i, j, cuda_mat[i * col2 + j], sum);
            }
        }
    }
}

int *generate_ones_vector(int size)
{
    int *vec = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        vec[i] = 1;
    }
    return vec;
}

#endif