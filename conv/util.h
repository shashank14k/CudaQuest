#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <jpeglib.h>

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

int *generate_ones_vector(int size)
{
    int *vec = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        vec[i] = 1;
    }
    return vec;
}

double *generate_random_vector(int size)
{
    double *vec = (double *)malloc(size * sizeof(double));
    fillrandomMatrix(vec, size, 1, 0);
    return vec;
}

void verify_cuda_1d_conv(double *mat1, double *mask, double *cuda_mat, int vec_len, int mask_len)
{
    int radius = mask_len / 2;
    double temp;
    int start;

    for (int i = 0; i < vec_len; i++)
    {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < mask_len; j++)
        {
            if (start + j >= 0 && start + j < vec_len)
            {
                temp += mask[j] * mat1[start + j];
            }
        }
        if (fabs(cuda_mat[i] - temp) > EPSILON)
        {
            fprintf(stderr, "Verification failed at index (%d): GPU result = %f, CPU result = %f\n", i, cuda_mat[i], temp);
        }
    }
}

#endif