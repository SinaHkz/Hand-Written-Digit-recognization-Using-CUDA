#include "utilities.h"
#include <cfloat>

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


void print_model(Model h_model, int input_size, int num_classes)
{
    FILE *file = fopen("./matrixTests/test.txt", "w");
    if (file == NULL)
    {
        printf("Error opening file for writing.\n");
        return;
    }

    // Write the dimensions (rows and columns) at the beginning of the file
    fprintf(file, "Rows: %d, Columns: %d\n\n", input_size, num_classes);

    // Write the weights
    fprintf(file, "Model Weights:\n");
    for (int i = 0; i < num_classes; i++)
    {
        fprintf(file, "Class %d: ", i);
        for (int j = 0; j < input_size; j++)
        {
            fprintf(file, "%.6f ", h_model.weights[i * input_size + j]);
        }
        fprintf(file, "\n");
    }

    // Write the biases
    fprintf(file, "\nModel Biases:\n");
    for (int i = 0; i < num_classes; i++)
    {
        fprintf(file, "Bias for Class %d: %.6f\n", i, h_model.biases[i]);
    }

    // Close the file
    fclose(file);
}

void print_matrix(float *matrix, int num_rows, int num_cols)
{
    for (int i = 0; i < num_rows; i++)
    {
        printf("[");
        for (int j = 0; j < num_cols; j++)
        {
            printf("%.2f", matrix[i * num_cols + j]); // Access element in row-major order
            if (j < num_cols - 1)
                printf(", "); // Add comma between elements
        }
        printf("]\n"); // New line for next row
    }
    printf("\n"); // Extra newline for better readability
}

void compare_matrices(float *matrix1, float *matrix2, int num_rows, int num_cols)
{
    bool are_equal = true;

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            int index = i * num_cols + j; // Row-major order index
            float diff = fabs(matrix1[index] - matrix2[index]);

            if (diff > EPSILON)
            { // If difference is greater than tolerance
                printf("Difference at (%d, %d): Matrix1 = %.6f, Matrix2 = %.6f, Diff = %.6f\n",
                       i, j, matrix1[index], matrix2[index], diff);
                are_equal = false;
            }
        }
    }

    if (are_equal)
    {
        printf("The matrices are equal (within tolerance of %.6f).\n", EPSILON);
    }
    else
    {
        printf("The matrices are NOT equal.\n");
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

void compute_full_softmax(const float *matrix, float *softmax_result, int num_images, int num_classes)
{
    for (int i = 0; i < num_images; i++)
    {
        float max_value = -FLT_MAX;

        // Step 1: Find the maximum logit for numerical stability
        for (int j = 0; j < num_classes; j++)
        {
            float value = *(matrix + i * num_classes + j); // Access matrix[i][j] using pointer arithmetic
            if (value > max_value)
            {
                max_value = value;
            }
        }

        // Step 2: Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++)
        {
            float exp_value = expf(*(matrix + i * num_classes + j) - max_value);
            *(softmax_result + i * num_classes + j) = exp_value; // Store intermediate exponential value
            sum_exp += exp_value;
        }

        // Step 3: Compute probabilities
        for (int j = 0; j < num_classes; j++)
        {
            *(softmax_result + i * num_classes + j) /= sum_exp; // Normalize by the sum of exponentials
        }
    }
}



void update_biases_sequential(float *matrix, float *result, int num_classes, int num_img, float lr)
{
    for (int i = 0; i < num_classes; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < num_img; j++)
        {
            sum += matrix[i * num_img + j]; // Sum elements of row i
        }
        result[i] = sum / num_img * lr; // Store result for row i
    }
}


void update_weights_sequential(float *images, float *deltas, float *weights, float lr, int num_img, int img_size, int num_classes)
{
    for (int xRow = 0; xRow < img_size; ++xRow)
    {
        for (int wRow = 0; wRow < num_classes; ++wRow)
        {
            float sum = 0.0f;

            // Calculate the summation part
            for (int tid = 0; tid < num_img; ++tid)
            {
                sum += images[xRow * num_img + tid] * deltas[wRow * num_img + tid];
            }

            // Update the weight
            weights[wRow * img_size + xRow] -= sum / num_img * lr;
        }
    }
}

int infer_digit(Model *model, unsigned char *image, int input_size)
{
    float logits[NUM_CLASSES] = {0};        // Array to store the logits for each class
    float softmax_probs[NUM_CLASSES] = {0}; // Array to store the softmax probabilities

    // Step 1: Compute logits
    for (int j = 0; j < NUM_CLASSES; j++)
    {
        logits[j] = model->biases[j]; // Start with the bias
        for (int i = 0; i < input_size; i++)
        {
            logits[j] += model->weights[j * input_size + i] * image[i]; // Weighted sum of inputs
        }
    }

    // Step 2: Compute softmax probabilities for numerical stability
    compute_full_softmax(logits, softmax_probs, 1, NUM_CLASSES); // We only infer for 1 image at a time

    // Step 3: Find the predicted label (class with the highest probability)
    int predicted_label = 0;
    float max_prob = softmax_probs[0];
    for (int j = 1; j < NUM_CLASSES; j++)
    {
        if (softmax_probs[j] > max_prob)
        {
            max_prob = softmax_probs[j];
            predicted_label = j;
        }
    }

    // printf("Predicted Label: %d (Confidence: %.2f%%)\n", predicted_label, max_prob * 100);
    return predicted_label;
}