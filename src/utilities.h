#include <stdio.h>
#include <stdlib.h>
#include "../includes/common.h"
#include "kernel.h"

#define BLOCKSIZE 16


unsigned char *read_idx3_file(const char *filename, int *count, int *rows, int *cols);

unsigned char *read_idx1_file(const char *filename, int *count);


void init_model(Model model, int input_size, int num_classes);

void print_model(Model h_model, int input_size, int num_classes);

void print_matrix(float *matrix, int num_rows, int num_cols);

void compare_matrices(float *matrix1, float *matrix2, int num_rows, int num_cols);

void compute_logits_cpu(float *weights, float *biases, float *images, float *logits, int input_size, int num_classes, int num_images);

void compute_full_softmax(const float *matrix, float *softmax_result, int num_images, int num_classes);

void update_biases_sequential(float *matrix, float *result, int num_classes, int num_img, float lr);

void update_weights_sequential(float *images, float *deltas, float *weights, float lr, int num_img, int img_size, int num_classes);

int infer_digit(Model *model, unsigned char *image, int input_size);

