#include "math.h"
#include <curand_kernel.h>


__device__ float cross_entropy_loss(float *probs, int true_label);

__global__ void init_weights_kernel(float *weights, int input_size, int num_classes, float scale);