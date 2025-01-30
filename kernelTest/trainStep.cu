#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define SHARED_MEMORY_SIZE 1024
#define NUM_CLASSES 10
#define BLOCK_SIZE 1024
#define NUM_IMAGES 128
#define SMAX_BLOCK_SIZE 40

typedef struct
{
    float *weights;
    float *biases;
} Model;

__global__ void compute_logits(float *weights, float *biases, unsigned char *images, float *logits, int input_size, int num_images)
{
    int j = blockIdx.x;       // Class index (row of weights)
    int img_idx = blockIdx.y; // Image index (row of images)
    int i = threadIdx.x;      // Input index

    // Define shared memory
    extern __shared__ float shared_mem[];
    float *swieghts = shared_mem;                                         // Start at 0
    float *simages = &shared_mem[(input_size + 31) / 32 * 32];            // Align to next multiple of 32
    float *shared_partial = &shared_mem[(2 * input_size + 31) / 32 * 32]; // Align to next multiple of 32

    // Load weights and images into shared memory
    if (i < input_size)
    {
        swieghts[i] = __ldg(&weights[j * input_size + i]);
        simages[i] = __ldg(&images[img_idx * input_size + i]);
    }
    // Initialize padding range if required
    if (i >= input_size && i < (2 * input_size))
    {
        shared_partial[i] = 0.0f;
    }
    __syncthreads();
    // Perform element-wise multiplication
    if (i < input_size)
    {
        shared_partial[i] = swieghts[i] * simages[i];
    }
    __syncthreads();
    // add unrolling the first 8 itertation ------------------------------------------------------------------------------------------------

    // Assuming shared_partial is already defined with SHARED_MEMORY_SIZE 1024
    if (i < 512)
        shared_partial[i] += shared_partial[i + 512];
    __syncthreads();

    if (i < 256)
        shared_partial[i] += shared_partial[i + 256];
    __syncthreads();

    if (i < 128)
        shared_partial[i] += shared_partial[i + 128];
    __syncthreads();

    if (i < 64)
        shared_partial[i] += shared_partial[i + 64];
    __syncthreads();

    if (i < 32)
    {
        volatile float *vmem = shared_mem;
        vmem[i] += vmem[i + 32];
        vmem[i] += vmem[i + 16];
        vmem[i] += vmem[i + 8];
        vmem[i] += vmem[i + 4];
        vmem[i] += vmem[i + 2];
        vmem[i] += vmem[i + 1];
    }
    // Write the result to logits (add bias)
    if (i == 0)
        logits[img_idx * NUM_CLASSES + j] = shared_partial[0] + __ldg(&biases[j]);
}

__global__ void softmax_kernel(const float *logits, float *probs, int num_elements)
{
    extern __shared__ float shared_data[]; // Shared memory for intermediate values
    float *max_logit_shared = shared_data; // Shared memory for max_logit reduction
    float *logits_shared = max_logit_shared + blockDim.x * blockDim.y;
    float *probs_shared = logits_shared + blockDim.x * blockDim.y;

    // 2D thread index calculation
    int tid_x = threadIdx.x;                                     // thread index in x direction
    int tid_y = threadIdx.y;                                     // thread index in y direction
    int idx = tid_y * blockDim.x + tid_x;                        // 1D index for shared memory
    int global_idx = idx + blockIdx.x * blockDim.x * blockDim.y; // Global index for logits

    // Step 1: Load data into shared memory
    if (global_idx < num_elements)
    {
        logits_shared[idx] = logits[global_idx];
        max_logit_shared[idx] = logits[global_idx];
    }
    else
    {
        logits_shared[idx] = 0.0f;
        max_logit_shared[idx] = -FLT_MAX; // Use a large negative value for out-of-bounds threads
    }
    __syncthreads();

    if (tid_x < 5)
    {
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 5]);
    }
    __syncthreads();

    if (tid_x < 2)
    {
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 2]);
    }
    __syncthreads();
    if (tid_x == 0)
    {
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 4]);
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 1]);
    }
    // Step 3: Compute the exponential values
    if (global_idx < num_elements)
    {
        logits_shared[idx] = expf(logits_shared[idx] - max_logit_shared[tid_y * blockDim.x]);
        probs_shared[idx] = logits_shared[idx];
    }
    __syncthreads();

    if (tid_x < 5)
    {
        logits_shared[idx] += logits_shared[idx + 5];
    }
    __syncthreads();
    if (tid_x < 2)
    {
        logits_shared[idx] += logits_shared[idx + 2];
    }
    __syncthreads();
    if (tid_x == 0)
    {
        logits_shared[idx] += logits_shared[idx + 4];
        logits_shared[idx] += logits_shared[idx + 1];
    }
    // Step 4: Compute probabilities
    probs_shared[idx] /= logits_shared[tid_y * blockDim.x];
    probs[global_idx] = probs_shared[idx];
}

__global__ void compute_delta(float *A, float *B, float *delta, float lr, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d = A[idx] - B[idx % NUM_CLASSES];
    
    if (idx < input_size)
        delta[idx] = d;
}

__global__ void update_weights(unsigned char *images, float *delta, float *weights, float lr, int m, int n)
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
            localSum += delta[row * n + col] * images[i * n + col];
        }

        // imagesccumulate the result in lossDerivives
        sum[row * n + col] += localSum;
        sum[row * n + col] /= m;
        weights[row * n + col] -= lr * sum[row * n + col];
    }
}

__global__ void traiStepKernel(Model model, unsigned char *images, int *label, int batch_size, int row, int col, float lr)
{
    int input_size = row * col;
    float *logits, *probs;
    const float lr = 0.1;
    cudaMalloc((void **)logits, input_size * sizeof(float));
    cudaMalloc((void **)probs, input_size * sizeof(float));

    dim3 threadPerBlock(BLOCK_SIZE);
    dim3 blockPerGrid(NUM_CLASSES, NUM_IMAGES);
    compute_logits<<<threadPerBlock, blockPerGrid, SHARED_MEMORY_SIZE>>>(model.weights, model.biases, images, logits, input_size, batch_size);

    dim3 threadPerBlock(SMAX_BLOCK_SIZE);
    dim3 blockPerGrid(NUM_CLASSES, SMAX_BLOCK_SIZE / NUM_CLASSES);
    softmax_kernel<<<threadPerBlock, blockPerGrid,  3 * SMAX_BLOCK_SIZE * sizeof(float)>>>(logits, probs, input_size);
    dim3 threadPerBlock();
    compute_delta<<<threadPerBlock, blockPerGrid>>>(probs, model.biases, probs, lr, row * col);
    update_weights<<<threadPerBlock, blockPerGrid, input_size * NUM_CLASSES * sizeof(float)>>>(images, probs, model.weights, lr, row, col);

    cudaFree(probs);
    cudaFree(logits);
}