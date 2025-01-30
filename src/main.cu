#include "utilities.h"

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

    Model h_model;
    int input_size = rows * cols;
    h_model.weights = (float *)malloc(input_size * NUM_CLASSES * sizeof(float));
    h_model.biases = (float *)malloc(NUM_CLASSES * sizeof(float));

    Model d_model = init_model(input_size, NUM_CLASSES);
    float learning_rate = 0.1;

    float *d_images, *d_logits, *d_delta, *dt_images, *dt_deltas;
    bool *d_label;

    cudaMalloc(&d_images, NUM_IMAGES * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_logits, NUM_IMAGES * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_delta, NUM_IMAGES * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_label, NUM_IMAGES * NUM_CLASSES * sizeof(bool));
    cudaMalloc(&dt_images, NUM_IMAGES * IMG_SIZE * sizeof(float));
    cudaMalloc(&dt_deltas, NUM_IMAGES * NUM_CLASSES * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_images, images, NUM_IMAGES * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, labels, NUM_CLASSES * NUM_IMAGES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(d_logits, 0, NUM_IMAGES * NUM_CLASSES * sizeof(float));

    float *d_prob = d_logits;

    // Updated thread and block names
    dim3 threadsPerBlock_z(COMPUTE_Z_BLOCK_SIZE);
    dim3 blocksPerGrid_z(NUM_CLASSES * NUM_IMAGES);
    compute_z<<<blocksPerGrid_z, threadsPerBlock_z>>>(h_model.weights, h_model.biases, d_images, d_logits, IMG_SIZE, NUM_CLASSES, NUM_IMAGES);

    dim3 threadsPerBlock_softmax(NUM_CLASSES, SMAX_BLOCK_SIZE / NUM_CLASSES);
    dim3 blocksPerGrid_softmax(NUM_IMAGES * NUM_CLASSES / SMAX_BLOCK_SIZE);
    compute_softmax<<<blocksPerGrid_softmax, threadsPerBlock_softmax>>>(d_logits, d_prob, NUM_CLASSES * NUM_IMAGES);

    dim3 threadsPerBlock_subtract(32, 16);
    dim3 blocksPerGrid_subtract((NUM_IMAGES + threadsPerBlock_subtract.x - 1) / threadsPerBlock_subtract.x,
                                (NUM_CLASSES + threadsPerBlock_subtract.y - 1) / threadsPerBlock_subtract.y);
    matrixSubtractKernel<<<blocksPerGrid_subtract, threadsPerBlock_subtract>>>(d_prob, d_label, d_delta, NUM_IMAGES, NUM_CLASSES);

    dim3 threadsPerBlock_transpose(8, 16);
    dim3 blocksPerGrid_transpose((NUM_IMAGES + 8 - 1) / 8, (IMG_SIZE + 16 - 1) / 16);
    transpose<<<blocksPerGrid_transpose, threadsPerBlock_transpose>>>(d_images, dt_images, NUM_IMAGES, NUM_CLASSES);

    dim3 threadsPerBlock_transpose_prob(8, 16);
    dim3 blocksPerGrid_transpose_prob((NUM_IMAGES + 8 - 1) / 8, (NUM_CLASSES + 16 - 1) / 16);
    transpose<<<blocksPerGrid_transpose_prob, threadsPerBlock_transpose_prob>>>(d_prob, dt_deltas, NUM_IMAGES, NUM_CLASSES);

    dim3 threadsPerBlock_update_biases(NUM_IMAGES);
    dim3 blocksPerGrid_update_biases(NUM_CLASSES);
    update_biases<<<blocksPerGrid_update_biases, threadsPerBlock_update_biases, IMG_SIZE * sizeof(float)>>>(dt_deltas, d_model.biases, learning_rate, NUM_CLASSES, IMG_SIZE, NUM_IMAGES);

    dim3 threadsPerBlock_update_weights(UPDATE_WEIGHT_BLOCK_SIZE);
    dim3 blocksPerGrid_update_weights(IMG_SIZE, NUM_CLASSES);
    update_wieghts<<<blocksPerGrid_update_weights, threadsPerBlock_update_weights>>>(dt_images, dt_deltas, d_model.weights, learning_rate, NUM_IMAGES, NUM_CLASSES, IMG_SIZE);

    cudaMemcpy(h_model.weights, d_model.weights, NUM_CLASSES * IMG_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_model.biases, d_model.biases, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    print_model(h_model, input_size, NUM_CLASSES);
}