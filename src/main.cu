#include "utilities.h"


void print_model(Model hostModel, int input_size, int num_classes)
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
            fprintf(file, "%.6f ", hostModel.weights[i * input_size + j]);
        }
        fprintf(file, "\n");
    }

    // Write the biases
    fprintf(file, "\nModel Biases:\n");
    for (int i = 0; i < num_classes; i++)
    {
        fprintf(file, "Bias for Class %d: %.6f\n", i, hostModel.biases[i]);
    }

    // Close the file
    fclose(file);
}

int main()
{
    const char *image_file = "./dataSet/train-images.idx3-ubyte";
    const char *label_file = "./dataSet/train-labels.idx1-ubyte";

    int image_count, label_count, rows, cols;
    unsigned char *images = read_idx3_file(image_file, &image_count, &rows, &cols);

    unsigned char *labels = read_idx1_file(label_file, &label_count);
    if (image_count != label_count)
    {
        fprintf(stderr, "Image and label counts do not match!\n");
        return EXIT_FAILURE;
    }

    Model hostModel;
    int input_size = rows * cols;
    hostModel.weights = (float *)malloc(input_size * NUM_CLASSES * sizeof(float));
    hostModel.biases = (float *)malloc(NUM_CLASSES * sizeof(float));

    Model model = init_model(input_size, NUM_CLASSES);
    float learning_rate = 0.1; // Reduced from 0.1

    cudaMemcpy(hostModel.weights, model.weights, input_size * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    // Copy biases from device to host
    cudaMemcpy(hostModel.biases, model.biases, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    print_model(hostModel, input_size, NUM_CLASSES);
}