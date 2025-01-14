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