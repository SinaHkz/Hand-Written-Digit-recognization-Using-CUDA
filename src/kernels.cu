#include "kernel.h"
#include <cfloat>
#include <stdio.h>

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

__global__ void compute_z(float *weights, float *biases, float *images, float *z, int img_size, int num_classes, int num_img)
{
    /*
    images -> num_img * img_size
    weights -> num_classes * img_size
    biases -> num_classes
    z -> num_img * num_classes
    */
    extern __shared__ float sdata[];
    int img_index = blockIdx.x / num_classes;
    int weight_index = blockIdx.x % num_classes;
    int tid = threadIdx.x;

    if (tid < img_size && img_index < num_img && weight_index < num_classes)
    {
        sdata[tid] = images[img_index * img_size + tid] * weights[weight_index * img_size + tid];
    }
    if (tid < (SHARED_MEMORY_SIZE - COMPUTE_Z_BLOCK_SIZE)) // i.e. tid < 224
    {
        sdata[tid + blockDim.x] = 0.0f;
    }
    __syncthreads();

    if (tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();

    // Warp-level reduction (no __syncthreads needed within a warp)
    if (tid < 32)
    {
        volatile float *vmem = sdata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
    {
        z[img_index * num_classes + weight_index] = sdata[0] + biases[weight_index];
    }
}

__global__ void softmax_kernel(const float *logits, float *probs, int num_elements)
{
    extern __shared__ float shared_data[];
    float *max_logit_shared = shared_data;
    float *logits_shared = max_logit_shared + blockDim.x * blockDim.y;
    float *probs_shared = logits_shared + blockDim.x * blockDim.y;

    // 2D thread index calculation
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx = tid_y * blockDim.x + tid_x;
    int global_idx = idx + blockIdx.x * blockDim.x * blockDim.y;

    // Step 1: Load data into shared memory
    if (global_idx < num_elements)
    {
        logits_shared[idx] = logits[global_idx];
        max_logit_shared[idx] = logits_shared[idx];
    }
    else
    {
        logits_shared[idx] = 0.0f;
        max_logit_shared[idx] = -FLT_MAX; // Use a large negative value for out-of-bounds threads
    }
    __syncthreads();

    if (tid_x < 5)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 5]);
    __syncthreads();

    if (tid_x < 2)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 2]);
    __syncthreads();

    if (tid_x == 0)
    {
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 4]);
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 1]);
    }
    __syncthreads();

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
    __syncthreads();
    // Step 4: Compute probabilities
    probs_shared[idx] /= logits_shared[tid_y * blockDim.x];
    probs[global_idx] = probs_shared[idx];
}

__global__ void compute_softmax(const float *logits, float *probs, int num_elements)
{
    extern __shared__ float shared_data[];
    float *max_logit_shared = shared_data;
    float *logits_shared = max_logit_shared + blockDim.x * blockDim.y;
    float *probs_shared = logits_shared + blockDim.x * blockDim.y;

    // 2D thread index calculation
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx = tid_y * blockDim.x + tid_x;
    int global_idx = idx + blockIdx.x * blockDim.x * blockDim.y;

    // Step 1: Load data into shared memory
    if (global_idx < num_elements)
    {
        logits_shared[idx] = logits[global_idx];
        max_logit_shared[idx] = logits_shared[idx];
    }
    else
    {
        logits_shared[idx] = 0.0f;
        max_logit_shared[idx] = -FLT_MAX; // Use a large negative value for out-of-bounds threads
    }
    __syncthreads();

    if (tid_x < 5)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 5]);
    __syncthreads();

    if (tid_x < 2)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 2]);
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

__global__ void matrixSubtractKernel(float *A, bool *B, float *C, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int index = row * n + col;
        C[index] = A[index] - (float)B[index];
        // printf("A: %f B: %d C: %f\n", A[index], B[index], C[index]);
    }
}

__global__ void transpose(float *in, float *out, int ny, int nx)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
        // printf("ix: %d, iy: %d\n", ix, iy);
    }
}

__global__ void update_biases(float *matrix, float *result, float lr, int num_classes, int num_img)
{
    extern __shared__ float shared_data[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Grid-stride loop for coalesced memory access and handling arbitrary num_img
    float sum = 0.0f;
    for(int i = tid; i < num_img; i += blockDim.x) {
        sum += matrix[row * num_img + i];
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Sequential reduction with power-of-two assumption
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle instructions (more efficient)
    if(tid < 32) {
        float val = shared_data[tid];
        for(int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        shared_data[tid] = val;
    }
    __syncthreads();

    // Write final result
    if(tid == 0) {
        result[row] = shared_data[0] / num_img * lr;
    }
}

__global__ void update_wieghts(float *images, float *deltas, float *weights, float lr, int num_img, int num_classes, int img_size)
{
    extern __shared__ float sdata[];
    const int xRow = blockIdx.x;
    const int wRow = blockIdx.y;
    const int tid = threadIdx.x;

    // Load data into shared memory (coalesced access)
    sdata[tid] = (tid < num_img) ? images[xRow * num_img + tid] * deltas[wRow * num_img + tid] : 0.0f;
    __syncthreads();

    // Optimized unrolled reduction
    // Sequential addressing reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final update by thread 0
    if (tid == 0)
    {
        weights[wRow * img_size + xRow] -= sdata[0] / num_img * lr;
    }
}

__global__ void matrixNormalizeKernel(unsigned char *A, float *B, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int index = row * n + col;
        B[index] = A[index] / 255.0f;
    }
}

__global__ void infer_softmax(const float *logits, int num_elements, int *predicted_class)
{
    extern __shared__ float shared_data[];
    float *max_logit_shared = shared_data;
    float *logits_shared = max_logit_shared + blockDim.x * blockDim.y;
    float *probs_shared = logits_shared + blockDim.x * blockDim.y;

    // 2D thread index calculation
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx = tid_y * blockDim.x + tid_x;
    int global_idx = idx + blockIdx.x * blockDim.x * blockDim.y;

    // Step 1: Load data into shared memory
    if (global_idx < num_elements)
    {
        logits_shared[idx] = logits[global_idx];
        max_logit_shared[idx] = logits_shared[idx];
    }
    else
    {
        logits_shared[idx] = 0.0f;
        max_logit_shared[idx] = -FLT_MAX; // Use a large negative value for out-of-bounds threads
    }
    __syncthreads();

    // Step 2: Compute the max logit for numerical stability
    if (tid_x < 5)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 5]);
    __syncthreads();

    if (tid_x < 2)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 2]);
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

    // Step 4: Sum up the exponentials to normalize probabilities
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

    // Step 5: Normalize probabilities (Divide each exp value by sum of exp values)
    probs_shared[idx] /= logits_shared[tid_y * blockDim.x];

    __syncthreads();

    // Step 6: Find the class with the maximum probability (for classification)
    if (tid_x == 0)
    {
        float max_prob = -FLT_MAX;
        int class_idx = -1;

        // Iterate through all probs to find max probability
        for (int i = 0; i < blockDim.x * blockDim.y; i++) // Change this to the number of classes if needed
        {
            if (probs_shared[i] > max_prob)
            {
                max_prob = probs_shared[i];
                class_idx = i;
            }
        }

        // Store the class index with the maximum probability
        if (class_idx != -1)
        {
            predicted_class[blockIdx.x] = class_idx; // store predicted class index
        }
    }
}
