#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define NUM_CLASSES 10
#define INPUT_SIZE 784 // MNIST image size: 28x28
#define BLOCK_SIZE 1024
#define SHARED_MEMORY_SIZE 1024
#define NUM_IMAGES 128

__global__ void compute_logits(float *weights, float *biases, float *images, float *logits,
                               int input_size, int num_classes, int num_images)
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

    // Reduction within the block
    for (int stride = SHARED_MEMORY_SIZE / 2; stride > 0; stride >>= 1)
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
        logits[img_idx * num_classes + j] = shared_partial[0] + __ldg(&biases[j]);
        // Uncomment for debugging
        // printf("blockIdx.x: %d blockIdx.y: %d shared_partial: %f\n", blockIdx.x, blockIdx.y, shared_partial[0]);
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

unsigned char *read_idx3_file(const char *filename, int *count, int *rows, int *cols)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2051)
    { // 0x00000803 in decimal
        fprintf(stderr, "Invalid magic number for idx3 file: %d\n", magic_number);
        exit(EXIT_FAILURE);
    }

    fread(count, sizeof(int), 1, file);
    *count = __builtin_bswap32(*count);

    fread(rows, sizeof(int), 1, file);
    *rows = __builtin_bswap32(*rows);

    fread(cols, sizeof(int), 1, file);
    *cols = __builtin_bswap32(*cols);

    int image_size = (*rows) * (*cols);
    unsigned char *data = (unsigned char *)malloc((*count) * image_size);
    fread(data, sizeof(unsigned char), (*count) * image_size, file);
    fclose(file);
    return data;
}

int main()
{
    // Load MNIST images
    const char *filename = "../dataSet/train-images.idx3-ubyte";
    int total_images, rows, cols;
    unsigned char *image_data = read_idx3_file(filename, &total_images, &rows, &cols);

    if (total_images < NUM_IMAGES)
    {
        fprintf(stderr, "Not enough images in the file.\n");
        exit(EXIT_FAILURE);
    }

    // Prepare input data
    float *images = (float *)malloc(NUM_IMAGES * INPUT_SIZE * sizeof(float));
    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            images[img_idx * INPUT_SIZE + i] = image_data[img_idx * INPUT_SIZE + i] / 255.0f; // Normalize to [0, 1]
        }
    }
    free(image_data);

    // for (int i = 0; i < 28; i++)
    // {
    //     for (int j = 0; j < 28; j++)
    //         printf("%.2f ", images[i * 28 + j]);
    //     printf("\n");
    // }

    float weights[NUM_CLASSES * INPUT_SIZE];
    float biases[NUM_CLASSES];
    float expected_logits[NUM_IMAGES * NUM_CLASSES] = {0};
    float device_logits[NUM_IMAGES * NUM_CLASSES] = {0};

    // Initialize weights and biases
    for (int j = 0; j < NUM_CLASSES; j++)
    {
        biases[j] = 0.1f * (j + 1);
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            weights[j * INPUT_SIZE + i] = 0.001f * (i + j + 1);
        }
    }

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
    compute_logits<<<gridDim, blockDim, (3 * SHARED_MEMORY_SIZE + 3) * sizeof(float)>>>(d_weights, d_biases, d_images, d_logits, INPUT_SIZE, NUM_CLASSES, NUM_IMAGES);

    cudaDeviceSynchronize();

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

    // Free GPU memory
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_images);
    cudaFree(d_logits);
    free(images);

    return 0;
}