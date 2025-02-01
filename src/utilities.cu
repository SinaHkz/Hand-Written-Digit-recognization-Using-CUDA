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
    unsigned char *data;
    cudaMallocManaged(&data, (*count) * image_size);
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

    unsigned char *data;
    cudaMallocManaged(&data, *count);
    fread(data, sizeof(unsigned char), *count, file);
    fclose(file);
    return data;
}

void init_weights_cpu(float *weights, int input_size, int num_classes, float scale)
{
    srand(1234); // Set the seed for reproducibility

    int total_elements = input_size * num_classes;
    for (int i = 0; i < total_elements; i++)
    {
        // Generate random numbers in the range [-0.5, 0.5)
        weights[i] = scale * ((rand() / (float)RAND_MAX) - 0.5f);
    }
}

void init_model(Model model, int input_size, int num_classes)
{
    // Allocate unified memory (accessible by both host and device)


    float scale = sqrtf(2.0f / input_size);

    // Initialize weights using the CPU-based function (CPU function is still valid with unified memory)
    init_weights_cpu(model.weights, input_size, num_classes, scale);

    // Use memset to set biases to zero (this works on managed memory as well)
    memset(model.biases, 0, num_classes * sizeof(float));
}
