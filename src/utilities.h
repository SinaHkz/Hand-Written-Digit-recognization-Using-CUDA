#include <stdio.h>
#include <stdlib.h>
#include "../includes/common.h"
#include "kernel.h"

#define BLOCKSIZE 16


unsigned char *read_idx3_file(const char *filename, int *count, int *rows, int *cols);

unsigned char *read_idx1_file(const char *filename, int *count);


Model init_model(Model h_model,int input_size, int num_classes);
