// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda.h>
// #include <math.h>

// #define NUM_CLASSES 3
// #define INPUT_SIZE 4
// #define BLOCK_SIZE 4
// #define NUM_IMAGES 2

// __global__ void compute_logits(float *weights, float *biases, float *images, float *logits, int input_size, int num_classes, int num_images)
// {
//     int j = blockIdx.x; // Class index (row of weights)
//     int img_idx = blockIdx.y; // Image index (row of images)
//     int i = threadIdx.x; // Input index

//     extern __shared__ float shared_partial[];

//     // Each threadblock handles one row of weights (one class) and one image
//     if (i < input_size)
//     {
//         shared_partial[i] = weights[j * input_size + i] * images[img_idx * input_size + i]; // Load and multiply weights with image into shared memory
//     }
//     else
//     {
//         shared_partial[i] = 0.0f;
//     }
//     __syncthreads();

//     // Reduction within the block
//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (i < stride)
//         {
//             shared_partial[i] += shared_partial[i + stride];
//         }
//         __syncthreads();
//     }

//     // Write the result to logits (add bias)
//     if (i == 0)
//     {
//         logits[img_idx * num_classes + j] = shared_partial[0] + biases[j];
//     }
// }

// void compute_logits_cpu(float *weights, float *biases, float *images, float *logits, int input_size, int num_classes, int num_images)
// {
//     for (int img_idx = 0; img_idx < num_images; img_idx++)
//     {
//         for (int j = 0; j < num_classes; j++)
//         {
//             logits[img_idx * num_classes + j] = biases[j];
//             for (int i = 0; i < input_size; i++)
//             {
//                 logits[img_idx * num_classes + j] += weights[j * input_size + i] * images[img_idx * input_size + i];
//             }
//         }
//     }
// }

// int main()
// {
//     float weights[NUM_CLASSES * INPUT_SIZE] = {
//         0.1, 0.4, 0.7, 1.0,  // Weights for class 0
//         0.2, 0.5, 0.8, 1.1,  // Weights for class 1
//         0.3, 0.6, 0.9, 1.2   // Weights for class 2
//     };
//     float biases[NUM_CLASSES] = {0.1, 0.2, 0.3};
//     float images[NUM_IMAGES * INPUT_SIZE] = {
//         1.0, 2.0, 3.0, 4.0,  // Image 1
//         2.0, 3.0, 4.0, 5.0   // Image 2
//     };
//     float expected_logits[NUM_IMAGES * NUM_CLASSES] = {0};
//     float device_logits[NUM_IMAGES * NUM_CLASSES] = {0};

//     // Compute logits on CPU for verification
//     compute_logits_cpu(weights, biases, images, expected_logits, INPUT_SIZE, NUM_CLASSES, NUM_IMAGES);

//     // Allocate memory on GPU
//     float *d_weights, *d_biases, *d_images, *d_logits;
//     cudaMalloc(&d_weights, NUM_CLASSES * INPUT_SIZE * sizeof(float));
//     cudaMalloc(&d_biases, NUM_CLASSES * sizeof(float));
//     cudaMalloc(&d_images, NUM_IMAGES * INPUT_SIZE * sizeof(float));
//     cudaMalloc(&d_logits, NUM_IMAGES * NUM_CLASSES * sizeof(float));

//     // Copy data to GPU
//     cudaMemcpy(d_weights, weights, NUM_CLASSES * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_biases, biases, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_images, images, NUM_IMAGES * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemset(d_logits, 0, NUM_IMAGES * NUM_CLASSES * sizeof(float));

//     // Launch kernel
//     dim3 blockDim(BLOCK_SIZE);
//     dim3 gridDim(NUM_CLASSES, NUM_IMAGES);
//     compute_logits<<<gridDim, blockDim, INPUT_SIZE * sizeof(float)>>>(d_weights, d_biases, d_images, d_logits, INPUT_SIZE, NUM_CLASSES, NUM_IMAGES);

//     // Copy results back to CPU
//     cudaMemcpy(device_logits, d_logits, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

//     // Compare results
//     printf("Expected logits:\n");
//     for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
//     {
//         for (int j = 0; j < NUM_CLASSES; j++)
//         {
//             printf("%f ", expected_logits[img_idx * NUM_CLASSES + j]);
//         }
//         printf("\n");
//     }
//     printf("\nDevice logits:\n");
//     for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
//     {
//         for (int j = 0; j < NUM_CLASSES; j++)
//         {
//             printf("%f ", device_logits[img_idx * NUM_CLASSES + j]);
//         }
//         printf("\n");
//     }
//     printf("\n\n");

//     // Free GPU memory
//     cudaFree(d_weights);
//     cudaFree(d_biases);
//     cudaFree(d_images);
//     cudaFree(d_logits);

//     return 0;
// }

#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>


__global__ void matrixSubtractKernel(float *A, float *B, float *C, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int index = row * n + col;
        C[index] = A[index] - B[index];
    }
}

__global__ void matrixOperationKernel(float *A, float *B, float *C, int m, int n)
{
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sum[];

    // Ensure thread is within bounds of matrix C
    if (row < m && col < n)
    {
        // Initialize local sum for C[row, col]
        float localSum = 0.0f;

        // Loop over all rows of matrix A and perform the operation
        for (int i = 0; i < m; i++)
        {
            localSum += B[row * n + col] * A[i * n + col];
        }

        // Accumulate the result in C
        sum[row * n + col] += localSum;
    }
}

void matrix_operation(float *A, float *B, float *C, int m, int n)
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
}

int main()
{
    // Define matrix dimensions
    int m = 3; // Number of rows in matrix B (and matrix A)
    int n = 2; // Number of columns in matrix A and B

    // Allocate memory on the host for matrices A, B, and C
    float *A = (float *)malloc(m * n * sizeof(float));  // Full matrix A (not just first row)
    float *B = (float *)malloc(m * n * sizeof(float));  // Full matrix B
    float *C = (float *)malloc(m * n * sizeof(float));  // Full matrix C, initialized to 0
    float *C2 = (float *)malloc(m * n * sizeof(float)); // Full matrix C, initialized to 0

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
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Handle width of matrix
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Handle height of matrix

    // Launch the kernel
    matrixOperationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n);

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

    matrix_operation(A, B, C2, m, n);

    // Print a part of the result matrix C for verification (optional)
    printf("Matrix C (result):\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // printf("%f ", C[i * n + j]);
            if (C[i * n + j] != C2[i * n + j])
            {
                printf("%d %d\n", i, j);
                exit(1);
            }
        }
        // printf("\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}


#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixSubtractKernel(float *A, float *B, float *C, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int index = row * n + col;
        C[index] = A[index] - B[index];
    }
}

__global__ void matrixOperationKernel(float *A, float *B, float *C, int m, int n)
{
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread is within bounds of matrix C
    if (row < m && col < n)
    {
        // Initialize local sum for C[row, col]
        float localSum = 0.0f;

        // Loop over all rows of matrix A and perform the operation
        for (int i = 0; i < m; i++)
        {
            localSum += B[row * n + col] * A[i * n + col];
        }

        // Accumulate the result in C
        C[row * n + col] += localSum;
    }
}

int main()
{
    // Define matrix dimensions
    int m = 3; // Number of rows in matrix B (and matrix A)
    int n = 2; // Number of columns in matrix A and B

    // Allocate memory on the host for matrices A, B, and C
    float *A = (float *)malloc(m * n * sizeof(float));     // Full matrix A (not just first row)
    float *B = (float *)malloc(m * n * sizeof(float));     // Full matrix B
    float *C = (float *)malloc(m * n * sizeof(float));     // Full matrix C, initialized to 0

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
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Handle width of matrix
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Handle height of matrix

    // Launch the kernel
    matrixOperationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n);

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

    // Print a part of the result matrix C for verification (optional)
    printf("Matrix C (result):\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}



__global__ void softmax_kernel(const float *logits, float *probs, int size, int num_elements) {
    extern __shared__ float shared_data[]; // Shared memory for intermediate values
    float *max_logit_shared = shared_data;  // Shared memory for max_logit reduction
    float *logits_shared = max_logit_shared + blockDim.x * blockDim.y;
    float *probs_shared = logits_shared + blockDim.x * blockDim.y;


    // 2D thread index calculation
    int tid_x = threadIdx.x;  // thread index in x direction
    int tid_y = threadIdx.y;  // thread index in y direction
    int idx = tid_y * blockDim.x + tid_x;  // 1D index for shared memory

    int global_idx = idx + blockIdx.x * blockDim.x * blockDim.y;  // Global index for logits

    // Step 1: Load data into shared memory
    if (global_idx < num_elements) {
        logits_shared[idx] = logits[global_idx];
        max_logit_shared[idx] = logits[global_idx];
    } else {
        logits_shared[idx] = 0.0f;
        max_logit_shared[idx] = -FLT_MAX; // Use a large negative value for out-of-bounds threads
    }
    __syncthreads();

    

    if(tid_x<5){
    max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 5]);
    }
    __syncthreads();

    if(tid_x < 2){
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 2]);
    }
    __syncthreads();
    if(tid_x == 0){
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 4]);
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 1]);
    }


    // Step 3: Compute the exponential values
    if (global_idx < num_elements) {
        logits_shared[idx] = expf(logits_shared[idx] - max_logit_shared[tid_y * blockDim.x]);
        probs_shared[idx] = logits_shared[idx];
        
    }
    __syncthreads();

    if(tid_x<5){
    logits_shared[idx] += logits_shared[idx + 5];
    }
    __syncthreads();
    if(tid_x < 2){
        logits_shared[idx] += logits_shared[idx + 2];
    }
    __syncthreads();
    if(tid_x == 0){
        logits_shared[idx] += logits_shared[idx + 4];
        logits_shared[idx] += logits_shared[idx + 1];
    }

    // Step 4: Compute probabilities
   
    probs_shared[idx] /= logits_shared[tid_y*blockDim.x];
    probs[global_idx] = probs_shared[idx];
}