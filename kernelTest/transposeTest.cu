#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NX 10  // Number of columns (width)
#define NY 128   // Number of rows (height)

// CUDA Kernel for Matrix Transpose
__global__ void transpose(float *in, float *out, int ny, int nx) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];  // Corrected indexing for NxM matrices
    }
}

// Sequential Transpose function for verification
void transpose_sequential(float *in, float *out, int ny, int nx) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            out[j * ny + i] = in[i * nx + j];  // Correct transpose for non-square matrices
        }
    }
}

// Initialize matrix with values
void init_matrix(float *mat, int ny, int nx) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            mat[i * nx + j] = (float)(i * nx + j + 1);
        }
    }
}

// Print matrix
void print_matrix(float *mat, int ny, int nx) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6.2f ", mat[i * nx + j]);
        }
        printf("\n");
    }
}

// Main function
int main() {
    int nx = NX, ny = NY;
    int size = nx * ny * sizeof(float);
    
    float *h_in, *h_out_cpu, *h_out_gpu;
    float *d_in, *d_out;

    // Allocate host memory
    h_in = (float *)malloc(size);
    h_out_cpu = (float *)malloc(size);
    h_out_gpu = (float *)malloc(size);

    // Initialize input matrix
    init_matrix(h_in, ny, nx);
    
    printf("Original Matrix (%dx%d):\n", ny, nx);
    print_matrix(h_in, ny, nx);

    // Compute the transpose using sequential function
    transpose_sequential(h_in, h_out_cpu, ny, nx);

    // Allocate device memory
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy data from host to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 blockSize(16, 16);  // 16x16 threads per block (tuned for performance)
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    transpose<<<gridSize, blockSize>>>(d_in, d_out, ny, nx);


    dim3 blockSize1(32, 16);  // 16x16 threads per block (tuned for performance)
    dim3 gridSize1((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    transpose<<<gridSize1, blockSize1>>>(d_in, d_out, ny, nx);




    dim3 blockSize2(8, 8);  // 16x16 threads per block (tuned for performance)
    dim3 gridSize2((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    transpose<<<gridSize2, blockSize2>>>(d_in, d_out, ny, nx);



    dim3 blockSize3(8, 16);  // 16x16 threads per block (tuned for performance)
    dim3 gridSize3((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    transpose<<<gridSize3, blockSize3>>>(d_in, d_out, ny, nx);
    cudaDeviceSynchronize();  // Ensure all threads are done

    // Copy result from device to host
    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    printf("\nSequential Transpose (%dx%d):\n", nx, ny);
    print_matrix(h_out_cpu, nx, ny);

    printf("\nGPU Transpose (%dx%d):\n", nx, ny);
    print_matrix(h_out_gpu, nx, ny);

    // Compare results
    int errors = 0;
    for (int i = 0; i < nx * ny; i++) {
        if (h_out_cpu[i] != h_out_gpu[i]) {
            errors++;
        }
    }

    if (errors == 0)
        printf("\nCUDA Transpose is CORRECT!\n");
    else
        printf("\nCUDA Transpose is INCORRECT! (%d mismatches)\n", errors);

    // Free memory
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
