#include "kernel.h"

__device__ float cross_entropy_loss(float *probs, int true_label)
{
    return -logf(fmaxf(probs[true_label], 1e-8));
}

__global__ void init_weights_kernel(float *weights, int input_size, int num_classes, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = input_size * num_classes;

    if (idx < total_elements)
    {
        curandState state;
        curand_init(1234, idx, 0, &state); // Seed for reproducibility
        weights[idx] = scale * (curand_uniform(&state) - 0.5f);
    }
}

__global__ void compute_logits(float *weights, float *biases, float *image, float *logits, int input_size, int num_classes)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ float partial_sums[];

    if (i < input_size && j < num_classes)
    {
        partial_sums[threadIdx.y * num_classes + j] = weights[i * num_classes + j] * image[i];
    }
    else
    {
        partial_sums[threadIdx.y * num_classes + j] = 0.0f;
    }
    __syncthreads();

    // Perform reduction within the block for each j
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.y < stride)
        {
            partial_sums[threadIdx.y * num_classes + j] += partial_sums[(threadIdx.y + stride) * num_classes + j];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && j < num_classes)
    {
        logits[j] += partial_sums[j];
    }

    // Ensure logits[j] is initialized with biases
    if (threadIdx.y == 0 && i == 0 && j < num_classes)
    {
        logits[j] += biases[j];
    }
    __syncthreads();

    if (i < input_size && j < num_classes)
    {
        weights[i * num_classes + j] = partial_sums[threadIdx.y * num_classes];
    }
}
