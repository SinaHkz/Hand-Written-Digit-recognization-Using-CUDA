#include "utilities.h"
#include <cfloat>

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

    printf("Predicted Label: %d (Confidence: %.2f%%)\n", predicted_label, max_prob * 100);
    return predicted_label;
}

int main()
{
    const char *image_file = "./dataSet/train-images.idx3-ubyte";
    const char *label_file = "./dataSet/train-labels.idx1-ubyte";

    int image_count, label_count, rows, cols;
    unsigned char *image_data = read_idx3_file(image_file, &image_count, &rows, &cols);

    unsigned char *labels = read_idx1_file(label_file, &label_count);
    if (image_count != label_count)
    {
        fprintf(stderr, "Image and label counts do not match!\n");
        return EXIT_FAILURE;
    }

    // Normilize data and save them in float array
    float *images;
    cudaMalloc(&images, image_count * IMG_SIZE * sizeof(float));
    dim3 block(32, 16);
    dim3 grid((image_count + block.x - 1) / block.x, (IMG_SIZE + block.y - 1) / block.y);
    matrixNormalizeKernel<<<grid, block>>>(image_data, images, image_count, IMG_SIZE);

    int input_size = rows * cols;

    Model d_model;
    cudaMallocManaged((void **)&d_model.weights, input_size * NUM_CLASSES * sizeof(float));
    cudaMallocManaged((void **)&d_model.biases, NUM_CLASSES * sizeof(float));

    init_model(d_model, IMG_SIZE, NUM_CLASSES);
    bool *h_onehot_labels = (bool *)calloc(image_count * NUM_CLASSES, sizeof(bool));

    for (int i = 0; i < image_count; i++)
    {
        int label = labels[i];
        h_onehot_labels[i * NUM_CLASSES + label] = true;
    }

    float learning_rate = 0.1f;

    // Allocate memory on GPU
    float *d_images, *d_logits, *dt_images, *dt_deltas, *d_delta;
    bool *d_label;
    cudaMalloc(&d_images, BATCH_SIZE * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_logits, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    cudaMalloc(&dt_images, BATCH_SIZE * IMG_SIZE * sizeof(float));
    cudaMalloc(&dt_deltas, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_label, BATCH_SIZE * NUM_CLASSES * sizeof(bool));
    cudaMallocManaged(&d_delta, BATCH_SIZE * NUM_CLASSES * sizeof(float));

    // Train in batches
    for (int epoch = 0; epoch < 10; epoch++)
    {
        for (int batch_start = 0; batch_start < image_count; batch_start += BATCH_SIZE)
        {
            int batch_end = min(batch_start + BATCH_SIZE, image_count);
            int batch_size = batch_end - batch_start;

            // Copy current batch to GPU
            cudaMemcpy(d_images, &images[batch_start * IMG_SIZE], batch_size * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_label, &h_onehot_labels[batch_start * NUM_CLASSES], batch_size * NUM_CLASSES * sizeof(bool), cudaMemcpyHostToDevice);
            cudaMemset(d_logits, 0, batch_size * NUM_CLASSES * sizeof(float));

            // Compute z
            dim3 threadsPerBlock_z(COMPUTE_Z_BLOCK_SIZE);
            dim3 blocksPerGrid_z(NUM_CLASSES * batch_size);
            compute_z<<<blocksPerGrid_z, threadsPerBlock_z, SHARED_MEMORY_SIZE * sizeof(float)>>>(d_model.weights, d_model.biases, d_images, d_logits, IMG_SIZE, NUM_CLASSES, batch_size);

            float *d_prob = d_logits;
            dim3 threadsPerBlock_softmax(NUM_CLASSES, SMAX_BLOCK_SIZE / NUM_CLASSES);
            dim3 blocksPerGrid_softmax(batch_size * NUM_CLASSES / SMAX_BLOCK_SIZE);
            compute_softmax<<<blocksPerGrid_softmax, threadsPerBlock_softmax, 3 * SMAX_BLOCK_SIZE * sizeof(float)>>>(d_logits, d_prob, NUM_CLASSES * batch_size);

            dim3 threadsPerBlock_subtract(8, 4);
            dim3 blocksPerGrid_subtract((NUM_CLASSES + threadsPerBlock_subtract.x - 1) / threadsPerBlock_subtract.x, (batch_size + threadsPerBlock_subtract.y - 1) / threadsPerBlock_subtract.y);
            matrixSubtractKernel<<<blocksPerGrid_subtract, threadsPerBlock_subtract>>>(d_prob, d_label, d_delta, batch_size, NUM_CLASSES);

            // Transpose images and deltas for backpropagation
            dim3 threadsPerBlock_transpose(16, 8);
            dim3 blocksPerGrid_transpose(
                (IMG_SIZE + threadsPerBlock_transpose.x - 1) / threadsPerBlock_transpose.x,
                (batch_size + threadsPerBlock_transpose.y - 1) / threadsPerBlock_transpose.y);
            transpose<<<blocksPerGrid_transpose, threadsPerBlock_transpose>>>(d_images, dt_images, batch_size, IMG_SIZE);

            dim3 threadsPerBlock_transpose_prob(16, 8);
            dim3 blocksPerGrid_transpose_prob((NUM_CLASSES + threadsPerBlock_transpose_prob.x - 1) / threadsPerBlock_transpose_prob.x, (batch_size + threadsPerBlock_transpose_prob.y - 1) / threadsPerBlock_transpose_prob.y);
            transpose<<<blocksPerGrid_transpose_prob, threadsPerBlock_transpose_prob>>>(d_delta, dt_deltas, batch_size, NUM_CLASSES);

            // Update biases
            dim3 threadsPerBlock_update_biases(batch_size);
            dim3 blocksPerGrid_update_biases(NUM_CLASSES);
            update_biases<<<blocksPerGrid_update_biases, threadsPerBlock_update_biases, batch_size * sizeof(float)>>>(dt_deltas, d_model.biases, learning_rate, NUM_CLASSES, batch_size);

            // Update weights
            dim3 threadsPerBlock_update_weights(UPDATE_WEIGHT_BLOCK_SIZE);
            dim3 blocksPerGrid_update_weights(IMG_SIZE, NUM_CLASSES);
            update_wieghts<<<blocksPerGrid_update_weights, threadsPerBlock_update_weights, 1024 * sizeof(float)>>>(dt_images, dt_deltas, d_model.weights, learning_rate, batch_size, NUM_CLASSES, IMG_SIZE);
        }
    }
    cudaDeviceSynchronize();

    const char *image_file_test = "./dataSet/t10k-images.idx3-ubyte";
    const char *label_file_test = "./dataSet/t10k-labels.idx1-ubyte";

    image_data = read_idx3_file(image_file_test, &image_count, &rows, &cols);

    labels = read_idx1_file(label_file_test, &label_count);
    if (image_count != label_count)
    {
        fprintf(stderr, "Image and label counts do not match!\n");
        return EXIT_FAILURE;
    }

    // Normilize data and save them in float array
    matrixNormalizeKernel<<<grid, block>>>(image_data, images, image_count, IMG_SIZE);

    int result = infer_digit(&d_model, (image_data + 0 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 1 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 2 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 3 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 4 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 5 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 6 * input_size), rows * cols);
    infer_digit(&d_model, (image_data + 7 * input_size), rows * cols);

    // print_model(d_model, IMG_SIZE, NUM_CLASSES);
    // compare_matrices(d_model.weights, h_model.weights, 1, NUM_CLASSES);

    // Free GPU memory
    cudaFree(d_images);
    cudaFree(d_logits);
    cudaFree(dt_images);
    cudaFree(dt_deltas);
    cudaFree(d_label);
    cudaFree(d_delta);
    cudaFree(images);

    // Free host memory
    free(h_onehot_labels);
    // free(h_model.weights);
    // free(h_model.biases);

    return 0;
}
