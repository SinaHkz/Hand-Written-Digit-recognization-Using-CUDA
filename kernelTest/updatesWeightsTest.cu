#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_CLASSES 10
#define IMG_SIZE 784
#define NUM_IMG 128
#define THREADS_PER_BLOCK 1024
#define SUM_ROW_SIZE 1024

// Function to print matrices
void print_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%0.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void checkResults(float *seq_weights, float *cuda_weights, int size)
{
    float tolerance = 1e-5;

    for (int i = 0; i < size; ++i)
    {
        if (fabs(seq_weights[i] - cuda_weights[i]) > tolerance)
        {
            printf("NO\nFirst difference at index %d: CPU = %f, GPU = %f\n", i, seq_weights[i], cuda_weights[i]);
            return;
        }
    }

    printf("YES\n");
}

// Sequential function to update weights
void update_weights_sequential(float *images, float *deltas, float *weights, float lr)
{
    for (int xRow = 0; xRow < IMG_SIZE; ++xRow)
    {
        for (int wRow = 0; wRow < NUM_CLASSES; ++wRow)
        {
            float sum = 0.0f;

            // Calculate the summation part
            for (int tid = 0; tid < NUM_IMG; ++tid)
            {
                sum += images[xRow * NUM_IMG + tid] * deltas[wRow * NUM_IMG + tid];
            }

            // Update the weight
            weights[wRow * IMG_SIZE + xRow] -= sum / NUM_IMG * lr;
        }
    }
}

// CUDA kernel to update weights
__global__ void update_wieghts(float *images, float *deltas, float *weights, 
                              float lr, int num_img, int num_classes, int img_size)
{
    extern __shared__ float sdata[];
    const int xRow = blockIdx.x;
    const int wRow = blockIdx.y;
    const int tid = threadIdx.x;

    // Load data into shared memory (coalesced access)
    sdata[tid] = (tid < num_img) ? 
        images[xRow * num_img + tid] * deltas[wRow * num_img + tid] : 
        0.0f;

    __syncthreads();

    // Optimized unrolled reduction
    if (blockDim.x >= 1024 && tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    // Warp-level unrolled reduction (no synchronization needed)
    if (tid < 32) {
        volatile float *vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }

    // Final update by thread 0
    if (tid == 0) {
        weights[wRow * img_size + xRow] -= sdata[0] / num_img * lr;
    }
}

int main()
{
    float lr = 0.1f; // Learning rate

    // Allocate memory for images, deltas, and weights
    float *images = (float *)malloc(IMG_SIZE * NUM_IMG * sizeof(float));
    float *deltas = (float *)malloc(NUM_CLASSES * NUM_IMG * sizeof(float));
    float *weights_seq = (float *)malloc(NUM_CLASSES * IMG_SIZE * sizeof(float));
    float *weights_cuda = (float *)malloc(NUM_CLASSES * IMG_SIZE * sizeof(float));

    // Initialize weights
    for (int i = 0; i < NUM_CLASSES * IMG_SIZE; i++)
    {
        weights_seq[i] = 0.5f;  // Initial value
        weights_cuda[i] = 0.5f; // Same initial value for CUDA comparison
    }

    // Initialize images and deltas with random values between 0 and 1
    for (int i = 0; i < IMG_SIZE * NUM_IMG; i++)
    {
        images[i] = rand() % 1000 / 1000.0f;
    }

    for (int i = 0; i < NUM_CLASSES * NUM_IMG; i++)
    {
        deltas[i] = rand() % 1000 / 1000.0f;
    }

    // Run the sequential function
    update_weights_sequential(images, deltas, weights_seq, lr);

    // Allocate device memory for CUDA
    float *d_images, *d_deltas, *d_weights;
    cudaMalloc((void **)&d_images, IMG_SIZE * NUM_IMG * sizeof(float));
    cudaMalloc((void **)&d_deltas, NUM_CLASSES * NUM_IMG * sizeof(float));
    cudaMalloc((void **)&d_weights, NUM_CLASSES * IMG_SIZE * sizeof(float));

    // Copy input data to the device
    cudaMemcpy(d_images, images, IMG_SIZE * NUM_IMG * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deltas, deltas, NUM_CLASSES * NUM_IMG * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_cuda, NUM_CLASSES * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(IMG_SIZE, NUM_CLASSES);

    // Run CUDA kernel
    update_wieghts<<<gridDim, blockDim, SUM_ROW_SIZE * sizeof(float)>>>(d_images, d_deltas, d_weights, lr, NUM_IMG, NUM_CLASSES, IMG_SIZE);

    // Copy updated weights from device to host
    cudaMemcpy(weights_cuda, d_weights, NUM_CLASSES * IMG_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Check if results match
    checkResults(weights_seq, weights_cuda, NUM_CLASSES * IMG_SIZE);

    // Free device memory
    cudaFree(d_images);
    cudaFree(d_deltas);
    cudaFree(d_weights);

    // Free host memory
    free(images);
    free(deltas);
    free(weights_seq);
    free(weights_cuda);

    return 0;
}