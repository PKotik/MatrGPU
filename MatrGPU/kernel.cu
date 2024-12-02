
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define EPS 0.000001 

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

double** CreateMatrix(unsigned int n)
{
    srand(time(NULL));
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(2 * n * sizeof(double));
    }

    return matrix;
}


void freeMatrix(unsigned int n, double** matrix)
{
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void Inic(double** matrix, unsigned int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 2 * n; j++) {
            if (j < n) matrix[i][j] = rand() % 9999999;
            else
            {
                if (i + n == j) matrix[i][j] = 1;
                else matrix[i][j] = 0;
            }
        }
    }
}

__global__ void SRowMatrix(unsigned int row, double* matrix, unsigned int n)
{
    //int j = blockIdx.x * blockDim.x + threadIdx.x;  // Индекс столбца
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Индекс строки
    if (i != row)
    {
        double k = matrix[i* (2 * n) +row] / matrix[row* (2 * n) +row];
        for (int j = 0; j < 2 * n; j++)
        {
            matrix[i* (2 * n) +j] -= k * matrix[row* (2 * n) +j];
        }
    }
}

//void SAllMatrix(double** matrix, unsigned int n)
//{
//    for (int row = 0; row < n; row++)
//    {
//        SRowMatrix(row, matrix, n);
//    }
//}



void PrintMatrixRev(double** matrix, unsigned int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            printf("%.6f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void PrintMatrixOrig(double** matrix, unsigned int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", (int)matrix[i][j]);
        }
        printf("\n");
    }
}

__global__ void GoToOneRows(double* matrix, unsigned int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double k = matrix[i* (2 * n) +i];
    matrix[i* (2 * n) +i] /= k;
    for (int j = n; j < 2 * n; j++)
    {
        matrix[i*(2*n)+j] /= k;
    }
}

int main(void)
{



    int n = 1000;
    size_t matrixSize = sizeof(double) * n * 2 * n;
    double** M = CreateMatrix(n);
    Inic(M, n);

    double *N = (double*)malloc(matrixSize);
    for (int i = 0; i < n * 2 * n; i++)
    {
        int l = i / (2*n);
        int k = i % (2*n);
        N[i] = M[l][k];
    }

    clock_t start, end;
    //PrintMatrixOrig(M, n);
    start = clock();

    double *A;
    cudaMalloc((void**)&A, matrixSize);
    cudaMemcpy(A, N, matrixSize, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    for (int row = 0; row < n; row++)
    {
        SRowMatrix << <gridDim, blockDim >> > (row, A, n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }
    GoToOneRows << <gridDim, blockDim >> > (A, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(N, A, matrixSize, cudaMemcpyDeviceToHost);
    cudaFree(A);

    end = clock();
    
    for (int i = 0; i < n * 2 * n; i++)
    {
        int l = i / (2 * n);
        int k = i % (2 * n);
        M[l][k]=N[i];
    }
    
    double time = (((double)(end - start)) * 1000) / CLOCKS_PER_SEC;

    //PrintMatrixOrig(M, n);
    //PrintMatrixRev(M, n);
    freeMatrix(n, M);
    free(N);
    /*system("cls");*/
    printf("Ready!!! Process took: ");
    printf("%d ms", (int)time);

}

