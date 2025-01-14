#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NUM_CLASSES 10
#define MINI_BATCH_SIZE 64 // Define the mini-batch size

typedef struct
{
    float *weights;
    float *biases;
} Model;

// Function to read idx3 file (images)
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
    unsigned char *data = malloc((*count) * image_size);
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

    unsigned char *data = malloc(*count);
    fread(data, sizeof(unsigned char), *count, file);
    fclose(file);
    return data;
}

// Softmax function
void softmax(float *logits, float *probs, int size)
{
    float max_logit = logits[0];
    for (int i = 1; i < size; i++)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }

    for (int i = 0; i < size; i++)
    {
        probs[i] /= sum;
    }
}

// Cross-entropy loss function
float cross_entropy_loss(float *probs, int true_label)
{
    return -logf(fmaxf(probs[true_label], 1e-8)); // Use fmaxf to clamp values
}

// Initialize model weights and biases
Model init_model(int input_size, int num_classes)
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

// Training step for mini-batch gradient descent
void train_step(Model *model, unsigned char **images, int *labels, int batch_size, int input_size, float lr)
{
    // Initialize gradient accumulators
    float gradient_weights[input_size * NUM_CLASSES];
    float gradient_biases[NUM_CLASSES] = {0};

    for (int i = 0; i < input_size * NUM_CLASSES; i++)
    {
        gradient_weights[i] = 0;
    }

    // Process each image in the mini-batch
    for (int b = 0; b < batch_size; b++)
    {
        unsigned char *image = images[b];
        int true_label = labels[b];

        float logits[NUM_CLASSES] = {0};
        float probs[NUM_CLASSES] = {0};

        // Compute logits
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            logits[j] = model->biases[j];
            for (int i = 0; i < input_size; i++)
            {
                logits[j] += model->weights[i * NUM_CLASSES + j] * image[i];
            }
        }

        // Compute softmax probabilities
        softmax(logits, probs, NUM_CLASSES);

        // Compute loss and gradient
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            float delta = probs[j] - (j == true_label ? 1.0 : 0.0);
            gradient_biases[j] += delta;
            for (int i = 0; i < input_size; i++)
            {
                gradient_weights[i * NUM_CLASSES + j] += delta * image[i];
            }
        }
    }

    // Average the gradients
    for (int j = 0; j < NUM_CLASSES; j++)
    {
        gradient_biases[j] /= batch_size;
    }
    for (int i = 0; i < input_size * NUM_CLASSES; i++)
    {
        gradient_weights[i] /= batch_size;
    }

    // Update model weights and biases
    for (int j = 0; j < NUM_CLASSES; j++)
    {
        model->biases[j] -= lr * gradient_biases[j];
        for (int i = 0; i < input_size; i++)
        {
            model->weights[i * NUM_CLASSES + j] -= lr * gradient_weights[i * NUM_CLASSES + j];
        }
    }
}

// Free the model
void free_model(Model *model)
{
    free(model->weights);
    free(model->biases);
}

// Function to infer the digit from a single image
int infer_digit(Model *model, unsigned char *image, int input_size)
{
    float logits[NUM_CLASSES] = {0};
    float probs[NUM_CLASSES] = {0};

    // Compute logits
    for (int j = 0; j < NUM_CLASSES; j++)
    {
        logits[j] = model->biases[j];
        for (int i = 0; i < input_size; i++)
        {
            logits[j] += model->weights[i * NUM_CLASSES + j] * image[i];
        }
    }

    // Compute softmax probabilities
    softmax(logits, probs, NUM_CLASSES);

    // Find the class with the highest probability
    int predicted_label = 0;
    float max_prob = probs[0];
    for (int j = 1; j < NUM_CLASSES; j++)
    {
        if (probs[j] > max_prob)
        {
            max_prob = probs[j];
            predicted_label = j;
        }
    }

    printf("Predicted Label: %d (Confidence: %.2f%%)\n", predicted_label, max_prob * 100);
    return predicted_label;
}

int main()
{
    const char *image_file = "/home/sinahz/Desktop/softmaxProject/dataSet/train-images.idx3-ubyte";
    const char *label_file = "/home/sinahz/Desktop/softmaxProject/dataSet/train-labels.idx1-ubyte";

    int image_count, label_count, rows, cols;
    unsigned char *images = read_idx3_file(image_file, &image_count, &rows, &cols);

    unsigned char *labels = read_idx1_file(label_file, &label_count);

    if (image_count != label_count)
    {
        fprintf(stderr, "Image and label counts do not match!\n");
        return EXIT_FAILURE;
    }

    int input_size = rows * cols;
    Model model = init_model(input_size, NUM_CLASSES);
    float learning_rate = 0.1; // Reduced from 0.1

    // Mini-batch training loop
    for (int epoch = 0; epoch < 1; epoch++)
    {
        printf("Epoch %d\n", epoch + 1);
        for (int i = 0; i < image_count; i += MINI_BATCH_SIZE)
        {
            // Prepare mini-batch
            unsigned char *mini_batch_images[MINI_BATCH_SIZE];
            int mini_batch_labels[MINI_BATCH_SIZE];
            int batch_size = MINI_BATCH_SIZE;

            for (int j = 0; j < batch_size && (i + j) < image_count; j++)
            {
                mini_batch_images[j] = images + (i + j) * input_size;
                mini_batch_labels[j] = labels[i + j];
            }

            // Perform the training step
            train_step(&model, mini_batch_images, mini_batch_labels, batch_size, input_size, learning_rate);
        }
    }

    // Normalize the image for inference
    unsigned char *test_image = images; // Select a test image
    int result = infer_digit(&model, (images + 14521 * input_size), rows * cols);
    int result1 = infer_digit(&model, (images + 14525 * input_size), rows * cols);
    int result2 = infer_digit(&model, (images + 14526 * input_size), rows * cols);
    int result3 = infer_digit(&model, (images + 14527 * input_size), rows * cols);

    // printf("Inference result0: %d\nInference result1: %d\nInference result2: %d\nInference result3: %d\n", result, result1, result2, result3);

    free_model(&model);
    free(images);
    free(labels);

    return 0;
}
