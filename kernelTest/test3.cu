#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <float.h>

#define NUM_IMAGES 128     // Number of images
#define NUM_CLASSES 10     // Number of classes per image (adjusted for 2 thread blocks)
#define SMAX_BLOCK_SIZE 20 // Total number of threads in a block

#define EPSILON 1e-6 // Tolerance for floating-point comparison

void compare_matrices(float matrix1[NUM_IMAGES][NUM_CLASSES], float matrix2[NUM_IMAGES][NUM_CLASSES])
{
    bool are_equal = true;

    for (int i = 0; i < NUM_IMAGES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            float diff = fabs(matrix1[i][j] - matrix2[i][j]);

            if (diff > EPSILON)
            { // If difference is greater than tolerance
                printf("Difference at (%d, %d): Matrix1 = %.6f, Matrix2 = %.6f, Diff = %.6f\n",
                       i, j, matrix1[i][j], matrix2[i][j], diff);
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

// Kernel declaration
__global__ void softmax_kernel(const float *logits, float *probs, int num_elements)
{
    extern __shared__ float shared_data[];
    float *max_logit_shared = shared_data;
    float *logits_shared = max_logit_shared + blockDim.x * blockDim.y;
    float *probs_shared = logits_shared + blockDim.x * blockDim.y;

    // 2D thread index calculation
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int idx = tid_y * blockDim.x + tid_x;
    int global_idx = idx + blockIdx.x * blockDim.x * blockDim.y;

    // Step 1: Load data into shared memory
    if (global_idx < num_elements)
    {
        logits_shared[idx] = logits[global_idx];
        max_logit_shared[idx] = logits_shared[idx];
    }
    else
    {
        logits_shared[idx] = 0.0f;
        max_logit_shared[idx] = -FLT_MAX; // Use a large negative value for out-of-bounds threads
    }
    __syncthreads();

    if (tid_x < 5)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 5]);
    __syncthreads();

    if (tid_x < 2)
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 2]);
    __syncthreads();

    if (tid_x == 0)
    {
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 4]);
        max_logit_shared[idx] = fmaxf(max_logit_shared[idx], max_logit_shared[idx + 1]);
    }

    // Step 3: Compute the exponential values
    if (global_idx < num_elements)
    {
        logits_shared[idx] = expf(logits_shared[idx] - max_logit_shared[tid_y * blockDim.x]);
        probs_shared[idx] = logits_shared[idx];
    }
    __syncthreads();

    if (tid_x < 5)
    {
        logits_shared[idx] += logits_shared[idx + 5];
    }
    __syncthreads();
    if (tid_x < 2)
    {
        logits_shared[idx] += logits_shared[idx + 2];
    }
    __syncthreads();
    if (tid_x == 0)
    {
        logits_shared[idx] += logits_shared[idx + 4];
        logits_shared[idx] += logits_shared[idx + 1];
    }
    // Step 4: Compute probabilities
    probs_shared[idx] /= logits_shared[tid_y * blockDim.x];
    probs[global_idx] = probs_shared[idx];
}

void compute_exponential_sums(const float matrix[NUM_IMAGES][NUM_CLASSES])
{
    // float sum_exponentials[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        float max_value = -FLT_MAX;
        // Step 1: Find the maximum logit for numerical stability
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            if (matrix[i][j] > max_value)
            {
                max_value = matrix[i][j];
            }
        }

        // Step 2: Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            sum_exp += expf(matrix[i][j] - max_value);
        }

        // Store the result
        // sum_exponentials[i] = sum_exp;

        // Print intermediate results for verification
        std::cout << "Row " << i << ": Sum of exponentials = " << sum_exp << std::endl;
    }
}
// Function to compute expected results (up to the exponential part of the softmax algorithm)
void compute_expected_results(const float *logits, float *expected_exp_logits, int num_images, int num_classes)
{
    for (int i = 0; i < num_images; i++)
    {
        float max_logit = -FLT_MAX;
        // Find the maximum logit for numerical stability
        for (int j = 0; j < num_classes; j++)
        {
            if (logits[i * num_classes + j] > max_logit)
            {
                max_logit = logits[i * num_classes + j];
            }
        }
        // Compute exponentials of logits (shifted by max_logit)
        for (int j = 0; j < num_classes; j++)
        {
            expected_exp_logits[i * num_classes + j] = expf(logits[i * num_classes + j] - max_logit);
        }
    }
}

void compute_full_softmax(const float matrix[NUM_IMAGES][NUM_CLASSES], float softmax_result[NUM_IMAGES][NUM_CLASSES])
{
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        float max_value = -FLT_MAX;
        // Step 1: Find the maximum logit for numerical stability
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            if (matrix[i][j] > max_value)
            {
                max_value = matrix[i][j];
            }
        }

        // Step 2: Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            float exp_value = expf(matrix[i][j] - max_value);
            softmax_result[i][j] = exp_value; // Store intermediate exponential value
            sum_exp += exp_value;
        }

        // printf("image: %d\n",i);
        // Step 3: Compute probabilities
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            // printf("expected sum %f\n",sum_exp);
            // printf("each exponent %f\n",softmax_result[i][j]);
            softmax_result[i][j] /= sum_exp; // Normalize by the sum of exponentials
        }
    }
    std::cout << "Computed softmax results (for validation):" << std::endl;
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            std::cout << softmax_result[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void initialize_random_matrix(float matrix[NUM_IMAGES][NUM_CLASSES])
{
    srand(time(NULL)); // Seed the random number generator
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            matrix[i][j] = ((float)rand() / RAND_MAX); // Generate random float between 0.0 and 1.0
        }
    }
}
void print_matrix(float *logits, int num_elements)
{
    int count = 0;
    for (int i = 0; i < num_elements; i++)
    {
        if (count % 40 == 0)
            printf("\n");
        std::cout << logits[i] << " ";
        if ((i + 1) % NUM_CLASSES == 0)
        {
            std::cout << std::endl; // Print each block in a new line
        }
        count += 1;
    }
}
int areEqual(double a, double b)
{
    double epsilon = 1e-9;
    return fabs(a - b) < epsilon;
}
// bool compare_matrices(float matrix1[NUM_IMAGES][NUM_CLASSES], float matrix2[NUM_IMAGES][NUM_CLASSES], float tolerance) {
//     for (int i = 0; i < NUM_IMAGES; i++) {
//         for (int j = 0; j < NUM_CLASSES; j++) {
//             if(!areEqual(matrix1[i][j],matrix2[i][j])){
//                 printf("i: %d j:%d\n",i,j);
//             printf("%f %f \n",matrix1[i][j],matrix2[i][j]);
//                 }
//         }
//     }
//     return true;
// }
void replaceFirstWithMax(int rows, int cols, float matrix[NUM_IMAGES][NUM_CLASSES])
{
    for (int i = 0; i < rows; i++)
    {
        float max = matrix[i][0];
        for (int j = 1; j < cols; j++)
        {
            if (matrix[i][j] > max)
            {
                max = matrix[i][j];
            }
        }
        matrix[i][0] = max;
    }
}
// bool compare_matrices(float *matrix1, float *matrix2, float tolerance) {
//     for (int i = 0; i < NUM_IMAGES; i++) {
//         for (int j = 0; j < NUM_CLASSES; j++) {
//             // Calculate the index in the 1D array (pointer representation)
//             int index = i * NUM_CLASSES + j;

//             // Compare values at the current index
//             if (fabs(matrix1[index] - matrix2[index]) > tolerance) {
//                 printf("Mismatch found at [%d][%d]: %.2f != %.2f\n", i, j, matrix1[index], matrix2[index]);
//                 return false;
//             }
//         }
//     }
//     return true;
// }

int main()
{
    // Define a small matrix for testing (NUM_IMAGES x NUM_CLASSES)
    //   float h_logits[NUM_IMAGES][NUM_CLASSES] = {
    //         {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
    //         {0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f},
    //         {1.0f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f},
    //         {0.9f, 1.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f}
    //     };
    float h_logits[NUM_IMAGES][NUM_CLASSES];
    float h_logits_softmax_result[NUM_IMAGES][NUM_CLASSES];
    initialize_random_matrix(h_logits);

    // Flatten the matrix
    printf("matrix:");
    float *h_logits_flat = &h_logits[0][0];
    print_matrix(h_logits_flat, NUM_IMAGES * NUM_CLASSES);

    //  replaceFirstWithMax(NUM_IMAGES,NUM_CLASSES,h_logits);

    // Compute expected results (up to the exponential part)
    float expected_exp_logits[NUM_IMAGES][NUM_CLASSES];
    compute_expected_results(h_logits_flat, &expected_exp_logits[0][0], NUM_IMAGES, NUM_CLASSES);

    // Print expected results
    // std::cout << "Expected results (up to exponential part):\n";
    // for (int i = 0; i < NUM_IMAGES; i++) {
    //     for (int j = 0; j < NUM_CLASSES; j++) {
    //         std::cout << expected_exp_logits[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // printf("expected sums");
    // compute_exponential_sums(h_logits);

    printf("expected full result: \n");
    compute_full_softmax(h_logits, h_logits_softmax_result);

    // Allocate memory on the device
    float *d_logits, *d_probs;
    size_t size = NUM_IMAGES * NUM_CLASSES * sizeof(float);
    cudaMalloc(&d_logits, size);
    cudaMalloc(&d_probs, size);

    // Copy the matrix to the device
    cudaMemcpy(d_logits, h_logits_flat, size, cudaMemcpyHostToDevice);

    // Configure the kernel
    dim3 gridDim(NUM_IMAGES * NUM_CLASSES / SMAX_BLOCK_SIZE);
    dim3 blockDim(NUM_CLASSES, SMAX_BLOCK_SIZE / NUM_CLASSES);

    // Launch the kernel
    softmax_kernel<<<gridDim, blockDim, 3 * SMAX_BLOCK_SIZE * sizeof(float)>>>(d_logits, d_probs, NUM_CLASSES * NUM_IMAGES);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    float h_probs[NUM_IMAGES][NUM_CLASSES];
    cudaMemcpy(h_probs, d_probs, size, cudaMemcpyDeviceToHost);

    // // Print the result from the kernel
    // std::cout << "Softmax kernel results:\n";
    // for (int i = 0; i < NUM_IMAGES; i++) {
    //     for (int j = 0; j < NUM_CLASSES; j++) {
    //         std::cout << h_probs[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    compare_matrices(h_probs, h_logits_softmax_result);

    // if(compare_matrices(h_logits_softmax_result,h_probs,0))
    //     printf("yay");
    // else
    // printf("shit");

    // Free device memory
    cudaFree(d_logits);
    cudaFree(d_probs);

    return 0;
}
