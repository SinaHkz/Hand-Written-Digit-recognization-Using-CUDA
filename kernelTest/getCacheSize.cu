#include <iostream>
#include <cuda_runtime.h>

__global__ void test_reduction(float *data, float *result, int n) {
    extern __shared__ float shared[]; // Shared memory for partial results

    int i = threadIdx.x;
    int lane_id = i % 32;  // Get the lane id within the warp
    int warp_id = i / 32;  // Get the warp id (assuming block size is a multiple of 32)

    // Load data into a register for the reduction
    float val = (i < n) ? data[i] : 0.0f;

    // Perform parallel reduction within the warp using __shfl_down_sync
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // After the warp-level reduction, each thread in a warp will have the partial sum in `val`.
    // Store the result of each warp in shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    // Perform block-level reduction: reduce the partial results from each warp (stored in shared)
    if (warp_id == 0) {
        // Reduce the warp-level results stored in shared memory
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            if (lane_id < offset) {
                shared[warp_id] += shared[warp_id + offset];
            }
            __syncthreads();
        }
    }

    // Write the final reduced result to the global memory
    if (warp_id == 0 && lane_id == 0) {
        *result = shared[0];
    }
}

int main() {
    const int n = 784; // Size of the array
    const int blockSize = 1024; // Threads per block

    // Allocate host memory
    float *h_data = new float[n];
    float h_result = 0.0f;

    // Initialize the host array with ones
    for (int i = 0; i < n; i++) {
        h_data[i] = 1.0f;
    }

    // Allocate device memory
    float *d_data, *d_result;
    cudaMalloc((void **)&d_data, n * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    test_reduction<<<1, blockSize, blockSize * sizeof(float)>>>(d_data, d_result, n);

    // Check for CUDA kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result from device to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Reduction result: " << h_result << std::endl;

    // Free memory
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}
