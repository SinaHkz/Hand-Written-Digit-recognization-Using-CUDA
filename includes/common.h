#define NUM_CLASSES 10
#define MINI_BATCH_SIZE 1024

#define EPSILON 1e-6 // Tolerance for floating-point comparison

// #define NUM_CLASSES 10
#define IMG_SIZE 784 // MNIST image size: 28x28
#define COMPUTE_Z_BLOCK_SIZE 1024
#define SMAX_BLOCK_SIZE 20
#define SHARED_MEMORY_SIZE 1024
#define UPDATE_WEIGHT_BLOCK_SIZE 1024
#define BATCH_SIZE 64 // Define your batch size


#define SMAX_ROWS_PER_BLOCK 16 //Number of rows in a block

#ifndef COMMON_H
#define COMMON_H

typedef struct {
    float *weights;
    float *biases;
} Model;

#endif // COMMON_H
