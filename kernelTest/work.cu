#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define NUM_CLASSES 3
#define INPUT_SIZE 4
#define BLOCK_SIZE 4
#define NUM_IMAGES 2

__global__ void compute_logits(float *weights, float *biases, float *images, float *logits, int input_size, int num_classes, int num_images)
{
    int j = blockIdx.x; // Class index (row of weights)
    int img_idx = blockIdx.y; // Image index (row of images)
    int i = threadIdx.x; // Input index

    extern __shared__ float shared_partial[];

    // Each threadblock handles one row of weights (one class) and one image
    if (i < input_size)
    {
        shared_partial[i] = weights[j * input_size + i] * images[img_idx * input_size + i]; // Load and multiply weights with image into shared memory
    }
    else
    {
        shared_partial[i] = 0.0f;
    }
    __syncthreads();

    // Reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (i < stride)
        {
            shared_partial[i] += shared_partial[i + stride];
        }
        __syncthreads();
    }

    // Write the result to logits (add bias)
    if (i == 0)
    {
        logits[img_idx * num_classes + j] = shared_partial[0] + biases[j];
    }
}

void compute_logits_cpu(float *weights, float *biases, float *images, float *logits, int input_size, int num_classes, int num_images)
{
    for (int img_idx = 0; img_idx < num_images; img_idx++)
    {
        for (int j = 0; j < num_classes; j++)
        {
            logits[img_idx * num_classes + j] = biases[j];
            for (int i = 0; i < input_size; i++)
            {
                logits[img_idx * num_classes + j] += weights[j * input_size + i] * images[img_idx * input_size + i];
            }
        }
    }
}

int main()
{
    float weights[NUM_CLASSES * INPUT_SIZE] = {
        0.1, 0.4, 0.7, 1.0,  // Weights for class 0
        0.2, 0.5, 0.8, 1.1,  // Weights for class 1
        0.3, 0.6, 0.9, 1.2   // Weights for class 2
    };
    float biases[NUM_CLASSES] = {0.1, 0.2, 0.3};
    float images[NUM_IMAGES * INPUT_SIZE] = {
        1.0, 2.0, 3.0, 4.0,  // Image 1
        2.0, 3.0, 4.0, 5.0   // Image 2
    };
    float expected_logits[NUM_IMAGES * NUM_CLASSES] = {0};
    float device_logits[NUM_IMAGES * NUM_CLASSES] = {0};

    // Compute logits on CPU for verification
    compute_logits_cpu(weights, biases, images, expected_logits, INPUT_SIZE, NUM_CLASSES, NUM_IMAGES);

    // Allocate memory on GPU
    float *d_weights, *d_biases, *d_images, *d_logits;
    cudaMalloc(&d_weights, NUM_CLASSES * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_biases, NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_images, NUM_IMAGES * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_logits, NUM_IMAGES * NUM_CLASSES * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_weights, weights, NUM_CLASSES * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_images, images, NUM_IMAGES * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_logits, 0, NUM_IMAGES * NUM_CLASSES * sizeof(float));

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(NUM_CLASSES, NUM_IMAGES);
    compute_logits<<<gridDim, blockDim, INPUT_SIZE * sizeof(float)>>>(d_weights, d_biases, d_images, d_logits, INPUT_SIZE, NUM_CLASSES, NUM_IMAGES);

    // Copy results back to CPU
    cudaMemcpy(device_logits, d_logits, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    printf("Expected logits:\n");
    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            printf("%f ", expected_logits[img_idx * NUM_CLASSES + j]);
        }
        printf("\n");
    }
    printf("\nDevice logits:\n");
    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            printf("%f ", device_logits[img_idx * NUM_CLASSES + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    // Free GPU memory
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_images);
    cudaFree(d_logits);

    return 0;
}
