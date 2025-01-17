
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdint.h>

#define INPUT_SIZE 32  // Adjust to be larger for 8 elements per thread
#define NUM_CLASSES 10 // Number of classes
#define BLOCK_SIZE 8   // Threads per block dimension

unsigned char *read_idx3_file(const char *filename, int *image_count, int *rows, int *cols)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    uint32_t magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(image_count, sizeof(uint32_t), 1, file);
    fread(rows, sizeof(uint32_t), 1, file);
    fread(cols, sizeof(uint32_t), 1, file);

    *image_count = __builtin_bswap32(*image_count);
    *rows = __builtin_bswap32(*rows);
    *cols = __builtin_bswap32(*cols);

    size_t image_size = (*rows) * (*cols);
    unsigned char *images = (unsigned char *)malloc((*image_count) * image_size * sizeof(unsigned char));
    fread(images, sizeof(unsigned char), (*image_count) * image_size, file);
    fclose(file);

    return images;
}

unsigned char *read_idx1_file(const char *filename, int *label_count)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    uint32_t magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(label_count, sizeof(uint32_t), 1, file);

    *label_count = __builtin_bswap32(*label_count);

    unsigned char *labels = (unsigned char *)malloc((*label_count) * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), *label_count, file);
    fclose(file);

    return labels;
}

__global__ void compute_logits(float *weights, float *biases, float *image, float *logits, int input_size, int num_classes)
{
    // Thread and block indices
    int threadId_x = threadIdx.x;
    int threadId_y = threadIdx.y;
    int blockId_x = blockIdx.x;
    int blockId_y = blockIdx.y;

    // Global thread indices
    int j = blockId_x * blockDim.x + threadId_x; // Column (class index)
    int base_i = blockId_y * blockDim.y * 8;     // Starting row for this block
    int local_row = threadId_y * 8;              // Local thread row start

    // Shared memory for partial sums
    extern __shared__ float partial_sums[];

    // Initialize thread-local sums
    float thread_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Perform 8 multiplications per thread
    for (int k = 0; k < 8; ++k)
    {
        int i = base_i + local_row + k; // Actual row index
        if (i < input_size && j < num_classes)
        {
            thread_sum[k / 2] += weights[i * num_classes + j] * image[i];
        }
    }

    // Store results into shared memory for reduction
    for (int k = 0; k < 4; ++k)
    {
        partial_sums[threadId_y * blockDim.x * 4 + threadId_x * 4 + k] = thread_sum[k];
    }
    __syncthreads();

    // Perform reductions across threads
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (threadId_y < stride)
        {
            for (int k = 0; k < 4; ++k)
            {
                partial_sums[threadId_y * blockDim.x * 4 + threadId_x * 4 + k] +=
                    partial_sums[(threadId_y + stride) * blockDim.x * 4 + threadId_x * 4 + k];
            }
        }
        __syncthreads();
    }

    // Accumulate final sums into global logits array and add bias
    if (threadId_y == 0 && j < num_classes)
    {
        for (int k = 0; k < 4; ++k)
        {
            logits[j] += partial_sums[threadId_x * 4 + k];
        }
        logits[j] += biases[j];  // Add bias to the final logits
    }
}

void compute_logits_cpu(float *weights, float *biases, float *image, float *logits, int input_size, int num_classes)
{
    for (int j = 0; j < num_classes; j++)
    {
        logits[j] = biases[j]; // Initialize with bias
        for (int i = 0; i < input_size; i++)
        {
            logits[j] += weights[i * num_classes + j] * image[i];
        }
    }
}

int main()
{
    const char *image_file = "../dataSet/train-images.idx3-ubyte";
    const char *label_file = "../dataSet/train-labels.idx1-ubyte";

    int image_count, label_count, rows, cols;
    unsigned char *images = read_idx3_file(image_file, &image_count, &rows, &cols);
    unsigned char *labels = read_idx1_file(label_file, &label_count);

    if (image_count == 0 || label_count == 0)
    {
        printf("Failed to read images or labels.\n");
        return 1;
    }

    printf("Loaded %d images with dimensions %d x %d\n", image_count, rows, cols);

    float weights[INPUT_SIZE * NUM_CLASSES];
    float biases[NUM_CLASSES];
    float expected_logits[NUM_CLASSES] = {0};
    float device_logits[NUM_CLASSES] = {0};

    // for (int i = 0; i < INPUT_SIZE; i++)
    // {
    //     for (int j = 0; j < NUM_CLASSES; j++)
    //     {
    //         weights[i * NUM_CLASSES + j] = (float)(i + 1) * (j + 1);
    //     }
    // }
    // for (int j = 0; j < NUM_CLASSES; j++)
    // {
    //     biases[j] = (float)(j + 1);
    // }

       for (int j = 0; j < NUM_CLASSES; j++)
    {
        biases[j] = 0.1f * (j + 1);
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            weights[j * INPUT_SIZE + i] = 0.001f * (i + j + 1);
        }
    }

    printf("Testing with the first image:\n");
    for (int i = 0; i < rows * cols; i++)
    {
        printf("%d ", images[i]);
    }
    printf("\n");

    float image[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE && i < rows * cols; i++)
    {
        image[i] = (float)images[i];
    }

    compute_logits_cpu(weights, biases, image, expected_logits, INPUT_SIZE, NUM_CLASSES);

    float *d_weights, *d_biases, *d_image, *d_logits;
    cudaMalloc(&d_weights, INPUT_SIZE * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_biases, NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_image, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_logits, NUM_CLASSES * sizeof(float));

    cudaMemcpy(d_weights, weights, INPUT_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_logits, 0, NUM_CLASSES * sizeof(float));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((NUM_CLASSES + BLOCK_SIZE - 1) / BLOCK_SIZE, (INPUT_SIZE / 8 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    compute_logits<<<gridDim, blockDim, BLOCK_SIZE * BLOCK_SIZE * NUM_CLASSES * sizeof(float)>>>(d_weights, d_biases, d_image, d_logits, INPUT_SIZE, NUM_CLASSES);

    cudaMemcpy(device_logits, d_logits, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nExpected logits:\n");
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        printf("%f ", expected_logits[i]);
    }
    printf("\n\nDevice logits:\n");
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        printf("%f ", device_logits[i]);
    }
    printf("\n\n");

    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_image);
    cudaFree(d_logits);
    free(images);
    free(labels);

    return 0;
}
