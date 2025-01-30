#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NUM_CLASSES 128  // Number of rows
#define NUM_IMG 1024     // Number of columns

// Sequential reduction sum (CPU version)
void row_reduction_sum_sequential(float *matrix, float *result, int num_classes, int num_img) {
    for (int i = 0; i < num_classes; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_img; j++) {
            sum += matrix[i * num_img + j];  // Sum elements of row i
        }
        result[i] = sum;  // Store result for row i
    }
}

// CUDA Kernel for Row-Wise Reduction Sum
__global__ void row_reduction_sum(float *matrix, float *result, int num_classes, int num_img) {
    __shared__ float shared_data[NUM_IMG];  // Ensure NUM_IMG matches maximum expected row size

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory with matrix row data
    shared_data[tid] = (tid < num_img) ? matrix[row * num_img + tid] : 0.0f;
    __syncthreads();

    // Unrolled parallel reduction with power-of-two assumption
    if (num_img >= 1024) { if (tid < 512) { shared_data[tid] += shared_data[tid + 512]; } __syncthreads(); }
    if (num_img >= 512) { if (tid < 256) { shared_data[tid] += shared_data[tid + 256]; } __syncthreads(); }
    if (num_img >= 256) { if (tid < 128) { shared_data[tid] += shared_data[tid + 128]; } __syncthreads(); }
    if (num_img >= 128) { if (tid < 64) { shared_data[tid] += shared_data[tid + 64]; } __syncthreads(); }

    // Warp-level unrolling (no synchronization needed)
    if (tid < 32) {
        volatile float *vsmem = shared_data;
        if (num_img >= 64) vsmem[tid] += vsmem[tid + 32];
        if (num_img >= 32) vsmem[tid] += vsmem[tid + 16];
        if (num_img >= 16) vsmem[tid] += vsmem[tid + 8];
        if (num_img >= 8) vsmem[tid] += vsmem[tid + 4];
        if (num_img >= 4) vsmem[tid] += vsmem[tid + 2];
        if (num_img >= 2) vsmem[tid] += vsmem[tid + 1];
    }

    // Write final result
    if (tid == 0) {
        result[row] = shared_data[0];
    }
}

int main() {
    int num_classes = NUM_CLASSES;
    int num_img = NUM_IMG;
    int matrix_size = num_classes * num_img * sizeof(float);
    int result_size = num_classes * sizeof(float);

    // Allocate host memory
    float *h_matrix = (float *)malloc(matrix_size);
    float *h_result_cpu = (float *)malloc(result_size);
    float *h_result_gpu = (float *)malloc(result_size);

    // Initialize matrix with random values (for testing)
    for (int i = 0; i < num_classes * num_img; i++) {
        h_matrix[i] = (float)(rand() % 10);  // Example: random values between 0-9
    }

    // Perform row-wise reduction on the CPU (sequential)
    row_reduction_sum_sequential(h_matrix, h_result_cpu, num_classes, num_img);

    // Allocate device memory
    float *d_matrix, *d_result;
    cudaMalloc((void **)&d_matrix, matrix_size);
    cudaMalloc((void **)&d_result, result_size);

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // Launch kernel (one block per row, threads per row = num_img)
    row_reduction_sum<<<num_classes, num_img>>>(d_matrix, d_result, num_classes, num_img);

    // Copy result back to host
    cudaMemcpy(h_result_gpu, d_result, result_size, cudaMemcpyDeviceToHost);

    // Print some results
    printf("Row-wise reduction sum results:\n");
    for (int i = 0; i < num_classes; i++) {
        printf("Row %d - CPU: %.2f, GPU: %.2f\n", i, h_result_cpu[i], h_result_gpu[i]);
    }

    // Compare results
    int errors = 0;
    for (int i = 0; i < num_classes; i++) {
        if (h_result_cpu[i] != h_result_gpu[i]) {
            errors++;
        }
    }

    if (errors == 0)
        printf("\nCUDA row reduction is CORRECT!\n");
    else
        printf("\nCUDA row reduction is INCORRECT! (%d mismatches)\n", errors);

    // Free memory
    free(h_matrix);
    free(h_result_cpu);
    free(h_result_gpu);
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
