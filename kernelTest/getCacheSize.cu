__global__ void matrixOperationKernel(float *images, float *gradients, float *lossDeriv, float *weights, float lr, int m, int n)
{
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sum[];

    // Ensure thread is within bounds of matrix lossDeriv
    if (row < m && col < n)
    {
        // Initialize local sum for lossDeriv[row, col]
        float localSum = 0.0f;

        // Loop over all rows of matrix images and perform the operation
        for (int i = 0; i < m; i++) // -------------------------------------------------> can be optimized by adding atomic sum and the other rows of images get calculated in parallel
        {
            localSum += gradients[row * n + col] * images[i * n + col];
        }

        // imagesccumulate the result in lossDeriv
        sum[row * n + col] += localSum;
        // lossDeriv[row * n + col] += localSum;
        sum[row * n + col] /= m;
        lossDeriv[row * n + col] = sum[row * n + col];
    }
}