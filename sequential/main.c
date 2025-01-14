#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NUM_CLASSES 10

typedef struct {
    float *weights;
    float *biases;
} Model;

// Function to read idx3 file (images)
unsigned char *read_idx3_file(const char *filename, int *count, int *rows, int *cols) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2051) { // 0x00000803 in decimal
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
unsigned char *read_idx1_file(const char *filename, int *count) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number);
    if (magic_number != 2049) { // 0x00000801 in decimal
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

// Rest of the functions remain unchanged
void softmax(float *logits, float *probs, int size) {
    float max_logit = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }

    for (int i = 0; i < size; i++) {
        probs[i] /= sum;
    }
}


float cross_entropy_loss(float *probs, int true_label) {
    return -logf(fmaxf(probs[true_label], 1e-8)); // Use fmaxf to clamp values
}


Model init_model(int input_size, int num_classes) {
    Model model;
    model.weights = malloc(input_size * num_classes * sizeof(float));
    model.biases = malloc(num_classes * sizeof(float));
    for (int i = 0; i < input_size * num_classes; i++) {
//        model.weights[i] = ((float) rand() / RAND_MAX) * 0.001; // Smaller values
        float scale = sqrtf(2.0f / input_size);
        model.weights[i] = scale * ((float) rand() / RAND_MAX - 0.5f);
    }
    for (int i = 0; i < num_classes; i++) {
        model.biases[i] = 0.0;
    }
    return model;
}

void train_step(Model *model, unsigned char *image, int true_label, int input_size, float lr) {
    float logits[NUM_CLASSES] = {0};
    float probs[NUM_CLASSES] = {0};
    float gradient_weights[input_size * NUM_CLASSES];
    float gradient_biases[NUM_CLASSES] = {0};

    for (int m = 0; m < input_size * NUM_CLASSES; m++) {
        gradient_weights[m] = 0;
    }

    for (int j = 0; j < NUM_CLASSES; j++) {
        logits[j] = model->biases[j];
        for (int i = 0; i < input_size; i++) {
            logits[j] += model->weights[i * NUM_CLASSES + j] * image[i];
        }
    }

    softmax(logits, probs, NUM_CLASSES);
    float loss = cross_entropy_loss(probs, true_label);

    for (int j = 0; j < NUM_CLASSES; j++) {
        //true label is the number that the input label had
        float delta = probs[j] - (j == true_label ? 1.0 : 0.0);
        gradient_biases[j] = delta;
        for (int i = 0; i < input_size; i++) {
            gradient_weights[i * NUM_CLASSES + j] += delta * image[i];
        }
    }

    for (int j = 0; j < NUM_CLASSES; j++) {
        //lr is learning rate
        model->biases[j] -= lr * gradient_biases[j];
        for (int i = 0; i < input_size; i++) {
            model->weights[i * NUM_CLASSES + j] -= lr * gradient_weights[i * NUM_CLASSES + j];
        }
    }

    //printf("Loss: %.4f\n", loss);
}

void free_model(Model *model) {
    free(model->weights);
    free(model->biases);
}

// Function to infer the digit from a single image
int infer_digit(Model *model, unsigned char *image, int input_size) {
    float logits[NUM_CLASSES] = {0};
    float probs[NUM_CLASSES] = {0};

    // Compute logits
    for (int j = 0; j < NUM_CLASSES; j++) {
        logits[j] = model->biases[j];
        for (int i = 0; i < input_size; i++) {
            logits[j] += model->weights[i * NUM_CLASSES + j] * image[i];
        }
    }

    // Compute softmax probabilities
    softmax(logits, probs, NUM_CLASSES);

    // Find the class with the highest probability
    int predicted_label = 0;
    float max_prob = probs[0];
    for (int j = 1; j < NUM_CLASSES; j++) {
        if (probs[j] > max_prob) {
            max_prob = probs[j];
            predicted_label = j;
        }
    }

    printf("Predicted Label: %d (Confidence: %.2f%%)\n", predicted_label, max_prob * 100);
    return predicted_label;
}

int main() {
    const char *image_file = "/home/sinahz/Desktop/softmaxProject/dataSet/train-images.idx3-ubyte";
    const char *label_file = "/home/sinahz/Desktop/softmaxProject/dataSet/train-labels.idx1-ubyte";

    int image_count, label_count, rows, cols;
    unsigned char *images = read_idx3_file(image_file, &image_count, &rows, &cols);
    float *normalized_images = malloc(image_count * rows * cols * sizeof(float));
    for (int i = 0; i < image_count * rows * cols; i++) {
        normalized_images[i] = images[i] / 255.0f;
    }

    unsigned char *labels = read_idx1_file(label_file, &label_count);

    if (image_count != label_count) {
        fprintf(stderr, "Image and label counts do not match!\n");
        return EXIT_FAILURE;
    }

    int input_size = rows * cols;
    Model model = init_model(input_size, NUM_CLASSES);
    float learning_rate = 0.1; // Reduced from 0.1


    for (int epoch = 0; epoch < 1; epoch++) {
        printf("Epoch %d\n", epoch + 1);
        for (int i = 0; i < image_count; i++) {
            unsigned char *image = images + i * input_size;
            int true_label = labels[i];
            train_step(&model, image, true_label, input_size, learning_rate);

        }
    }
    int result = infer_digit(&model, (images + 14507 * input_size), rows * cols);
    printf("result : %d\n", result);

    
    free(images);
    free(labels);
    free_model(&model);
    return 0;
}
