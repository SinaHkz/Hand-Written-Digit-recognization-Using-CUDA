#include "math.h"
#include <curand_kernel.h>
#include "../includes/common.h"


__device__ float cross_entropy_loss(float *probs, int true_label);

__global__ void init_weights_kernel(float *weights, int input_size, int num_classes, float scale);

__global__ void compute_z(float *weights, float *biases, float *images, float *z, int img_size, int num_classes, int num_img);

__global__ void compute_softmax(const float *logits, float *probs, int num_elements);

__global__ void matrixSubtractKernel(float *A, bool *B, float *C, int m, int n);

__global__ void transpose(float *in, float *out, int rows, int cols);

__global__ void update_biases(float *matrix, float *result, float lr, int num_classes, int num_img);

__global__ void update_wieghts(float *images, float *deltas, float *weights, float lr, int num_img, int num_classes, int img_size);

__global__ void softmax_kernel(const float *logits, float *probs, int num_elements);

__global__ void matrixNormalizeKernel(unsigned char *A, float *B,int m, int n);
