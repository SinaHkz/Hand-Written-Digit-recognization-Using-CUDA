#include "utilities.h"

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

    float learning_rate = 0.5f;
    int batchSize = BATCH_SIZE;

    // Allocate memory on GPU
    float *d_images, *d_logits, *dt_images, *dt_deltas, *d_delta;
    bool *d_label;
    // Allocate memory for entire dataset on GPU
    cudaMalloc(&d_images, image_count * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_label, image_count * NUM_CLASSES * sizeof(bool));
    // Allocate other buffers with original batchSize (maximum size)
    cudaMalloc(&d_logits, batchSize * NUM_CLASSES * sizeof(float));
    cudaMalloc(&dt_images, batchSize * IMG_SIZE * sizeof(float));
    cudaMalloc(&dt_deltas, batchSize * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_delta, batchSize * NUM_CLASSES * sizeof(float));

    // Copy entire dataset to GPU once
    cudaMemcpy(d_images, images, image_count * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, h_onehot_labels, image_count * NUM_CLASSES * sizeof(bool), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < 10; epoch++)
    {
        for (int batch_start = 0; batch_start < image_count; batch_start += batchSize)
        {
            int batch_end = min(batch_start + batchSize, image_count);
            int current_batch_size = batch_end - batch_start; // Renamed to avoid shadowing

            // Calculate pointers for current batch
            float *current_d_images = d_images + batch_start * IMG_SIZE;
            bool *current_d_label = d_label + batch_start * NUM_CLASSES;

            // Reset logits for current batch
            cudaMemset(d_logits, 0, current_batch_size * NUM_CLASSES * sizeof(float));

            // Compute logits with current batch data
            dim3 threadsPerBlock_z(COMPUTE_Z_BLOCK_SIZE);
            dim3 blocksPerGrid_z(NUM_CLASSES * current_batch_size);
            compute_z<<<blocksPerGrid_z, threadsPerBlock_z, SHARED_MEMORY_SIZE * sizeof(float)>>>(d_model.weights, d_model.biases, current_d_images, d_logits, IMG_SIZE, NUM_CLASSES, current_batch_size);

            // Softmax computation
            float *d_prob = d_logits;
            dim3 threadsPerBlock_softmax(NUM_CLASSES, SMAX_BLOCK_SIZE / NUM_CLASSES);
            dim3 blocksPerGrid_softmax(current_batch_size * NUM_CLASSES / SMAX_BLOCK_SIZE);
            compute_softmax<<<blocksPerGrid_softmax, threadsPerBlock_softmax, 3 * SMAX_BLOCK_SIZE * sizeof(float)>>>(d_logits, d_prob, NUM_CLASSES * current_batch_size);

            // Calculate delta
            dim3 threadsPerBlock_subtract(8, 4);
            dim3 blocksPerGrid_subtract(
                (NUM_CLASSES + threadsPerBlock_subtract.x - 1) / threadsPerBlock_subtract.x,
                (current_batch_size + threadsPerBlock_subtract.y - 1) / threadsPerBlock_subtract.y);
            matrixSubtractKernel<<<blocksPerGrid_subtract, threadsPerBlock_subtract>>>(d_prob, current_d_label, d_delta, current_batch_size, NUM_CLASSES);

            // Transpose current batch images
            dim3 threadsPerBlock_transpose(16, 8);
            dim3 blocksPerGrid_transpose(
                (IMG_SIZE + threadsPerBlock_transpose.x - 1) / threadsPerBlock_transpose.x,
                (current_batch_size + threadsPerBlock_transpose.y - 1) / threadsPerBlock_transpose.y);
            transpose<<<blocksPerGrid_transpose, threadsPerBlock_transpose>>>(current_d_images, dt_images, current_batch_size, IMG_SIZE);

            // Transpose deltas
            dim3 threadsPerBlock_transpose_prob(16, 8);
            dim3 blocksPerGrid_transpose_prob(
                (NUM_CLASSES + threadsPerBlock_transpose_prob.x - 1) / threadsPerBlock_transpose_prob.x,
                (current_batch_size + threadsPerBlock_transpose_prob.y - 1) / threadsPerBlock_transpose_prob.y);
            transpose<<<blocksPerGrid_transpose_prob, threadsPerBlock_transpose_prob>>>(d_delta, dt_deltas, current_batch_size, NUM_CLASSES);

            // Update biases
            dim3 threadsPerBlock_update_biases(current_batch_size);
            dim3 blocksPerGrid_update_biases(NUM_CLASSES);
            update_biases<<<blocksPerGrid_update_biases, threadsPerBlock_update_biases, current_batch_size * sizeof(float)>>>(dt_deltas, d_model.biases, learning_rate, NUM_CLASSES, current_batch_size);

            // Update weights
            dim3 threadsPerBlock_update_weights(UPDATE_WEIGHT_BLOCK_SIZE);
            dim3 blocksPerGrid_update_weights(IMG_SIZE, NUM_CLASSES);
            update_wieghts<<<blocksPerGrid_update_weights, threadsPerBlock_update_weights, 1024 * sizeof(float)>>>(dt_images, dt_deltas, d_model.weights, learning_rate, current_batch_size, NUM_CLASSES, IMG_SIZE);
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

    int result;
    int count = 0;
    for (int i = 0; i < image_count; i++)
    {
        result = infer_digit(&d_model, (image_data + i * input_size), rows * cols);
        if (result == labels[i])
            count++;
    }
    printf("%d\n", count);

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

    return 0;
}
