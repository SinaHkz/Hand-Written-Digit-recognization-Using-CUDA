# Handwritten Digit Recognition Using CUDA

This project implements and optimizes a Softmax-based classifier for handwritten digit recognition on the MNIST dataset using CUDA. By leveraging GPU parallelism, it significantly improves training performance over traditional CPU-based and Python implementations.

## Project Overview

- **Goal**: Accelerate Softmax classification using CUDA for the MNIST dataset.
- **Method**: Implement batch gradient descent with optimized CUDA kernels.
- **Result**: Achieved up to **84.09% accuracy** and **2× speedup** compared to CPU baselines.

## Features

- CUDA-based Softmax classifier from scratch.
- Custom GPU kernels for:
  - Matrix multiplication
  - Softmax computation
  - Gradient updates
- Batch training using gradient descent.
- Optimization techniques:
  - Memory transfer efficiency
  - Thread block tuning
  - Loop unrolling for kernel performance

## Performance Comparison

| Version         | Accuracy (%) | Time (s)    |
|----------------|--------------|-------------|
| C Code         | N/A          | 35.13       |
| CUDA V1        | 81.2         | 24.89       |
| CUDA V2        | 81.2         | 19.79       |
| CUDA V3        | 84.09        | 19.81       |
| CUDA V4        | 84.09        | **10.62**    |



## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- MNIST dataset

