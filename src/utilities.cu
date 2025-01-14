#include "utilities.h"

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

// Function to read idx1 file (labels)
unsigned char *read_idx1_file(const char *filename, int *count)
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
    if (magic_number != 2049)
    { // 0x00000801 in decimal
        fprintf(stderr, "Invalid magic number for idx1 file: %d\n", magic_number);
        exit(EXIT_FAILURE);
    }

    fread(count, sizeof(int), 1, file);
    *count = __builtin_bswap32(*count);

    unsigned char *data = (unsigned char *)malloc(*count);
    fread(data, sizeof(unsigned char), *count, file);
    fclose(file);
    return data;
}

Model init_model(int input_size, int num_classes)
{
    Model model;

    // Allocate memory on the GPU
    cudaMalloc((void **)&model.weights, input_size * num_classes * sizeof(float));
    cudaMalloc((void **)&model.biases, num_classes * sizeof(float));

    // Calculate scale factor
    float scale = sqrtf(2.0f / input_size);

    // Initialize weights using a CUDA kernel
    int total_weights = input_size * num_classes;
    dim3 block(BLOCKSIZE);
    dim3 grid((total_weights + BLOCKSIZE - 1) / BLOCKSIZE);
    init_weights_kernel<<<grid, block>>>(model.weights, input_size, num_classes, scale);

    // Use cudaMemset to set biases to zero
    cudaMemset(model.biases, 0, num_classes * sizeof(float));

    // Synchronize to ensure initialization is complete
    cudaDeviceSynchronize();

    return model;
}