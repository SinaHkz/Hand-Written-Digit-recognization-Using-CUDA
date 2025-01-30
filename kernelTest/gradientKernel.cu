#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixSubtractKernel(float *A, float *B, float *C, float *biases, float lr, int m, int n)//bugggggggggggggggggggggggggggggggggggggggggggggg
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int index = row * n + col;
        C[index] = A[index] - B[index];
        biases[index] -= lr * C[index];
    }
}

__global__ void matrixOperationKernel(float *images, float *gradients, float *weights, float lr, int m, int n)
{
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sum[];

    // Ensure thread is within bounds of matrix lossDerivives
    if (row < m && col < n)
    {
        // Initialize local sum for lossDerivives[row, col]
        float localSum = 0.0f;

        // Loop over all rows of matrix images and perform the operation
        for (int i = 0; i < m; i++) // --------------------------------------------> can be optimized by adding atomic sum and the other rows of images get calculated in parallel
        {
            localSum += gradients[row * n + col] * images[i * n + col];
        }

        // imagesccumulate the result in lossDerivives
        sum[row * n + col] += localSum;
        // lossDerivives[row * n + col] += localSum;
        sum[row * n + col] /= m;
        // lossDerivives[row * n + col] = sum[row * n + col];
        weights[row * n + col] -= lr * sum[row * n + col];
    }
}

void matrix_operation(float *A, float *B, float *C, float *weights, float lr, int m, int n)
{
    // Loop through each row of A
    for (int k = 0; k < m; k++)
    { // k goes from 0 to N-1 (rows of A)
        // For each row of A, perform operation on B and add to C
        for (int i = 0; i < m; i++)
        { // i goes from 0 to M-1 (rows of B)
            for (int j = 0; j < n; j++)
            { // j goes from 0 to M-1 (columns of B and C)
                C[i * n + j] += B[i * n + j] * A[k * n + j];
            }
        }
    }
    for (int i = 0; i < m; i++)
    { // i goes from 0 to M-1 (rows of B)
        for (int j = 0; j < n; j++)
        { // j goes from 0 to M-1 (columns of B and C)
            C[i * n + j] /= m;
            weights[i * n + j] -= lr * C[i * n + j];
        }
    }
}

int main()
{
    // Define matrix dimensions
    int m = 128; // Number of rows in matrix B (and matrix A)
    int n = 10;  // Number of columns in matrix A and B

    // Allocate memory on the host for matrices A, B, and C
    float *A = (float *)malloc(m * n * sizeof(float));  // Full matrix A (not just first row)
    float *B = (float *)malloc(m * n * sizeof(float));  // Full matrix B
    float *C = (float *)malloc(m * n * sizeof(float));  // Full matrix C, initialized to 0
    float *C2 = (float *)malloc(m * n * sizeof(float)); // Full matrix C, initialized to 0
    float *C3 = (float *)malloc(m * n * sizeof(float)); // Full matrix C, initialized to 0

    // Initialize A with example values (full matrix A in this case)
    for (int i = 0; i < m * n; i++)
    {
        A[i] = (float)i;
    }

    // Initialize B with example values
    for (int i = 0; i < m * n; i++)
    {
        B[i] = (float)i; // Just an example initialization
    }

    // Initialize C to zero
    memset(C, 0, m * n * sizeof(float));
    memset(C2, 0, m * n * sizeof(float));
    memset(C3, 0, m * n * sizeof(float));

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float)); // Full matrix A
    cudaMalloc(&d_B, m * n * sizeof(float)); // Full matrix B
    cudaMalloc(&d_C, m * n * sizeof(float)); // Full matrix C

    // Copy data from host to device
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Handle width of matrix
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Handle height of matrix

    // Launch the kernel
    matrixOperationKernel<<<blocksPerGrid, threadsPerBlock, n * m * sizeof(float)>>>(d_A, d_B, d_C, 0.1, m, n);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result back to the host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    matrix_operation(A, B, C2, C3, 0.1, m, n);

    // Print a part of the result matrix C for verification (optional)
    printf("Matrix C (result):\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // printf("%f ", C3[i * n + j]);
            if (C[i * n + j] != C3[i * n + j])
            {
                printf("%d %d\n", i, j);
                exit(1);
            }
        }
        // printf("\n");
    }
    printf("Horaaaa\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
