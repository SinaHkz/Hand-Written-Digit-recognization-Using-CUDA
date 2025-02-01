#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// Define constants
#define NUM_CLASSES 10
#define NUM_IMAGES 60000 // Number of MNIST training images
#define IMG_SIZE 784     // 28x28 images flattened into 784
#define BATCH_SIZE 1     // Size of each batch
#define LEARNING_RATE 0.01f
#define EPSILON 1e-6 // Tolerance for comparing floating point numbers

// Model structure
typedef struct
{
    float *weights;
    float *biases;
} Model;

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

    // Initialize random number generator
    srand(time(NULL));

    // Allocate memory for weights and biases
    model.weights = malloc(input_size * num_classes * sizeof(float));
    if (model.weights == NULL)
    {
        fprintf(stderr, "Memory allocation for weights failed!\n");
        exit(1);
    }

    model.biases = malloc(num_classes * sizeof(float));
    if (model.biases == NULL)
    {
        fprintf(stderr, "Memory allocation for biases failed!\n");
        free(model.weights); // Free previously allocated memory
        exit(1);
    }

    // Xavier/Glorot initialization for weights (ReLU case)
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * num_classes; i++)
    {
        model.weights[i] = scale * ((float)rand() / RAND_MAX - 0.5f);
    }

    // Initialize biases to 0
    for (int i = 0; i < num_classes; i++)
    {
        model.biases[i] = 0.0f;
    }

    return model;
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

void print_model(Model h_model, int input_size, int num_classes)
{
    FILE *file = fopen("../matrixTests/test.txt", "w");
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

// Implement the other functions (print_matrix, init_weights_cpu, compare_matrices, etc.)

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

void subtract_matrices(float *A, bool *B, float *result, int rows, int cols)
{
    // Perform the matrix subtraction using pointers
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            *(result + i * cols + j) = *(A + i * cols + j) - *(B + i * cols + j); // Subtract corresponding elements
        }
    }
}

void transpose_sequential(float *in, float *out, int ny, int nx)
{
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            out[j * ny + i] = in[i * nx + j];
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
        result[i] -= sum / num_img * lr; // Store result for row i
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

// Optimized function to infer the predicted digit from the model
int infer_digit(Model *model, float *image, int input_size)
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

    printf("Predicted Label: %d (Confidence: %.2f%%)\n", predicted_label, max_prob * 100);
    return predicted_label;
}

void write_logits_to_file(const char *filename, float *logits, int num_images, int num_classes)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Error opening file");
        return;
    }

    for (int img_idx = 0; img_idx < num_images; img_idx++)
    {
        fprintf(file, "Image %d:\n", img_idx + 1); // Print image index
        for (int j = 0; j < num_classes; j++)
        {
            fprintf(file, "Class %d: %.6f\n", j, logits[img_idx * num_classes + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void write_batch_to_file(const char *filename, float *batch_matrix, int num_images, int num_features)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Error opening file");
        return;
    }

    // Write the batch matrix to the file
    for (int img_idx = 0; img_idx < num_images; img_idx++)
    {
        for (int feature_idx = 0; feature_idx < num_features; feature_idx++)
        {
            fprintf(file, "%d ", (int)batch_matrix[img_idx * num_features + feature_idx]);
        }
        fprintf(file, "\n"); // Add a newline after each image
    }

    fclose(file);
}

#include <stdio.h>

void print_matrix_weights_to_file(float *matrix, int rows, int cols, const char *filename)
{
    FILE *file = fopen(filename, "w"); // Open file in write mode
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fprintf(file, "Weight at [%d][%d]: %.6f\n", i, j, matrix[i * cols + j]);
        }
    }

    fclose(file); // Close the file after writing
}

Model init_model1(int input_size, int num_classes)
{
    Model model;
    model.weights = malloc(input_size * num_classes * sizeof(float));
    model.biases = malloc(num_classes * sizeof(float));
    for (int i = 0; i < input_size * num_classes; i++)
    {
        float scale = sqrtf(2.0f / input_size);
        model.weights[i] = scale * ((float)rand() / RAND_MAX - 0.5f);
    }
    for (int i = 0; i < num_classes; i++)
    {
        model.biases[i] = 0.0;
    }
    return model;
}

int main()
{
    const char *image_file = "../dataSet/train-images.idx3-ubyte";
    const char *label_file = "../dataSet/train-labels.idx1-ubyte";

    // Load dataset
    int image_count, label_count, rows, cols;
    unsigned char *image_data = read_idx3_file(image_file, &image_count, &rows, &cols);
    unsigned char *labels = read_idx1_file(label_file, &label_count);

    if (image_count != label_count)
    {
        fprintf(stderr, "Image and label counts do not match!\n");
        return EXIT_FAILURE;
    }

    // Prepare input data for all images
    float *images = (float *)malloc(NUM_IMAGES * IMG_SIZE * sizeof(float));
    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        for (int i = 0; i < IMG_SIZE; i++)
        {
            images[img_idx * IMG_SIZE + i] = image_data[img_idx * IMG_SIZE + i] / 255.0f; // Normalize to [0, 1]
        }
    }

    Model h_model;
    int input_size = rows * cols;
    // h_model.weights = (float *)malloc(input_size * NUM_CLASSES * sizeof(float));
    // h_model.biases = (float *)malloc(NUM_CLASSES * sizeof(float));
    // float scale = sqrtf(2.0f / input_size);
    h_model = init_model1(input_size, NUM_CLASSES);
    memset(h_model.biases, 0, NUM_CLASSES * sizeof(float));

    bool *h_onehot_labels = (bool *)calloc(image_count * NUM_CLASSES, sizeof(bool));
    for (int i = 0; i < image_count; i++)
    {
        int label = labels[i];
        h_onehot_labels[i * NUM_CLASSES + label] = true;
    }

    // Allocate memory for logits, deltas, and transposed images and deltas
    float *h_logits = (float *)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
    float *h_delta = (float *)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
    float *th_images = (float *)malloc(BATCH_SIZE * IMG_SIZE * sizeof(float));
    float *th_deltas = (float *)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));

    // Training loop - process all batches and update the model
    for (int epoch = 0; epoch < 10; epoch++)
    { // Run for a fixed number of epochs
        printf("Epoch %d\n", epoch + 1);

        for (int batch_start = 0; batch_start < image_count; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > image_count)
            {
                batch_end = image_count;
            }
            memset(h_logits, 0, BATCH_SIZE * NUM_CLASSES * sizeof(float));

            // write_batch_to_file("text.txt",images,28,28);
            // exit(1);

            int batch_size = batch_end - batch_start;

            // Compute logits for the current batch
            compute_logits_cpu(h_model.weights, h_model.biases, &images[batch_start * IMG_SIZE], h_logits, IMG_SIZE, NUM_CLASSES, batch_size);
            // write_logits_to_file("test.txt",h_logits, batch_size, NUM_CLASSES);
            // exit(1);
            // Apply softmax
            compute_full_softmax(h_logits, h_logits, batch_size, NUM_CLASSES);
            // write_logits_to_file("test.txt", h_logits, batch_size, NUM_CLASSES);
            // exit(1);
            // Calculate delta (error between predictions and true labels)
            subtract_matrices(h_logits, &h_onehot_labels[batch_start * NUM_CLASSES], h_delta, batch_size, NUM_CLASSES);

            // Transpose images and deltas for efficient memory access
            transpose_sequential(&images[batch_start * IMG_SIZE], th_images, batch_size, IMG_SIZE);
            transpose_sequential(h_delta, th_deltas, batch_size, NUM_CLASSES);

            // Update biases and weights using gradient descent
            update_biases_sequential(th_deltas, h_model.biases, NUM_CLASSES, batch_size, LEARNING_RATE);

            update_weights_sequential(th_images, th_deltas, h_model.weights, LEARNING_RATE, batch_size, IMG_SIZE, NUM_CLASSES);
        }
    }

    print_matrix_weights_to_file(h_model.weights, NUM_CLASSES, IMG_SIZE, "test.txt");

    int result = infer_digit(&h_model, (images + 0 * input_size), rows * cols);
    result = infer_digit(&h_model, (images + 1 * input_size), rows * cols);
    result = infer_digit(&h_model, (images + 2 * input_size), rows * cols);
    result = infer_digit(&h_model, (images + 3 * input_size), rows * cols);

    // Print final model weights and biases
    print_model(h_model, IMG_SIZE, NUM_CLASSES);

    // Free memory
    free(images);
    free(h_model.weights);
    free(h_model.biases);
    free(h_onehot_labels);
    free(h_logits);
    free(h_delta);
    free(th_images);
    free(th_deltas);

    return EXIT_SUCCESS;
}
