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