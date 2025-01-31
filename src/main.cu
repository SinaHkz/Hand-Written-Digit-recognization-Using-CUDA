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
            printf("%.6f", matrix[i * num_cols + j]); // Access element in row-major order
            if (j < num_cols - 1)
                printf(", "); // Add comma between elements
        }
        printf("]\n"); // New line for next row
    }
    printf("\n"); // Extra newline for better readability
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
            out[j * ny + i] = in[i * nx + j]; // Correct transpose for non-square matrices
        }
    }
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

    // Prepare input data
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
    h_model.weights = (float *)malloc(input_size * NUM_CLASSES * sizeof(float));
    h_model.biases = (float *)malloc(NUM_CLASSES * sizeof(float));
    float scale = sqrtf(2.0f / input_size);
    init_weights_cpu(h_model.weights, IMG_SIZE, NUM_CLASSES, scale);
    memset(h_model.biases, 0, NUM_CLASSES * sizeof(float));

    Model d_model = init_model(h_model, IMG_SIZE, NUM_CLASSES);
    bool *h_onehot_labels = (bool *)calloc(NUM_IMAGES * NUM_CLASSES, sizeof(bool));
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        int label = labels[i];
        h_onehot_labels[i * NUM_CLASSES + label] = true;
    }
    float *ph_logits, *ph_delta;
    float *h_images, *h_logits, *th_images, *th_deltas, *h_prob, *h_delta;

    h_logits = (float *)malloc(NUM_IMAGES * IMG_SIZE * sizeof(float));
    ph_logits = (float *)malloc(NUM_IMAGES * IMG_SIZE * sizeof(float));
    h_prob = (float *)malloc(NUM_IMAGES * NUM_CLASSES * sizeof(float));
    h_delta = (float *)malloc(NUM_IMAGES * NUM_CLASSES * sizeof(float));
    ph_delta = (float *)malloc(NUM_IMAGES * NUM_CLASSES * sizeof(float));

    th_images = (float *)malloc(NUM_IMAGES * NUM_CLASSES * sizeof(float));
    th_deltas = (float *)malloc(NUM_IMAGES * NUM_CLASSES * sizeof(float));

    h_images = images;

    memset(h_logits, 0, NUM_IMAGES * IMG_SIZE * sizeof(float));

    // Allocate memory on GPU
    float *d_images, *d_logits, *dt_images, *dt_deltas, *d_delta;
    bool *d_label;
    cudaMalloc(&d_images, NUM_IMAGES * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_logits, NUM_IMAGES * NUM_CLASSES * sizeof(float));
    cudaMalloc(&dt_images, NUM_IMAGES * IMG_SIZE * sizeof(float));
    cudaMalloc(&dt_deltas, NUM_IMAGES * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_label, NUM_IMAGES * NUM_CLASSES * sizeof(bool));
    cudaMallocManaged(&d_delta, NUM_IMAGES * NUM_CLASSES * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_images, images, NUM_IMAGES * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, h_onehot_labels, NUM_CLASSES * NUM_IMAGES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(d_logits, 0, NUM_IMAGES * NUM_CLASSES * sizeof(float));

    dim3 threadsPerBlock_z(COMPUTE_Z_BLOCK_SIZE);
    dim3 blocksPerGrid_z(NUM_CLASSES * NUM_IMAGES);
    compute_z<<<blocksPerGrid_z, threadsPerBlock_z, SHARED_MEMORY_SIZE * sizeof(float)>>>(d_model.weights, d_model.biases, d_images, d_logits, IMG_SIZE, NUM_CLASSES, NUM_IMAGES);
    compute_logits_cpu(h_model.weights, h_model.biases, h_images, h_logits, IMG_SIZE, NUM_CLASSES, NUM_IMAGES);

    float *d_prob = d_logits;
    dim3 threadsPerBlock_softmax(NUM_CLASSES, SMAX_BLOCK_SIZE / NUM_CLASSES);
    dim3 blocksPerGrid_softmax(NUM_IMAGES * NUM_CLASSES / SMAX_BLOCK_SIZE);
    compute_softmax<<<blocksPerGrid_softmax, threadsPerBlock_softmax, 3 * SMAX_BLOCK_SIZE * sizeof(float)>>>(d_logits, d_prob, NUM_CLASSES * NUM_IMAGES);
    compute_full_softmax(h_logits, h_logits, NUM_IMAGES, NUM_CLASSES);

    dim3 threadsPerBlock_subtract(32, 16);
    dim3 blocksPerGrid_subtract((NUM_CLASSES + threadsPerBlock_subtract.x - 1) / threadsPerBlock_subtract.x, (NUM_IMAGES + threadsPerBlock_subtract.y - 1) / threadsPerBlock_subtract.y);
    matrixSubtractKernel<<<blocksPerGrid_subtract, threadsPerBlock_subtract>>>(d_prob, d_label, d_delta, NUM_IMAGES, NUM_CLASSES);
    subtract_matrices(h_logits, h_onehot_labels, h_delta, NUM_IMAGES, NUM_CLASSES);

    // dim3 threadsPerBlock_transpose(8, 16);
    // dim3 blocksPerGrid_transpose((NUM_IMAGES + 8 - 1) / 8, (IMG_SIZE + 16 - 1) / 16);
    // transpose<<<blocksPerGrid_transpose, threadsPerBlock_transpose>>>(d_images, dt_images, NUM_IMAGES, NUM_CLASSES);

    // dim3 threadsPerBlock_transpose_prob(8, 16);
    // dim3 blocksPerGrid_transpose_prob((NUM_IMAGES + 8 - 1) / 8, (NUM_CLASSES + 16 - 1) / 16);
    // transpose<<<blocksPerGrid_transpose_prob, threadsPerBlock_transpose_prob>>>(d_prob, dt_deltas, NUM_IMAGES, NUM_CLASSES);

    // transpose(h_images, th_images);
    // transpose(pro);

    // print_matrix(h_logits, NUM_IMAGES, NUM_CLASSES);

    cudaDeviceSynchronize();
    cudaError_t err = cudaMemcpy(ph_delta, d_delta, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA memcpy Device to Host failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); // Stop execution if an error occurs
    }
    

    // print_matrix(d_delta, NUM_CLASSES, NUM_IMAGES);
    compare_matrices(ph_delta, h_delta, NUM_IMAGES, NUM_CLASSES);
}
