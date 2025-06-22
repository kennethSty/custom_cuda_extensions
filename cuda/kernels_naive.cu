#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__ void dense_naive(const float* input, const float* weight, float* output, int batch, int n) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x; // feature index
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; // row (batch) index

    if (x < n && y < batch) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += input[y * n + k] * weight[k * n + x];
        }
        output[y * n + x] = sum;
    }
}

__global__ void bias_relu(float* input, const float* bias, int batch, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // feature index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row (batch) index

    if (x < n && y < batch) {
        input[y * n + x] = fmaxf(0.0, input[y * n + x] + bias[x]);
    }
}

__global__ void square(float* input, int batch, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // feature index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row (batch) index

    if (x < n && y < batch) {
        input[y * n + x] = input[y * n + x] * input[y * n + x];
    }
}

__global__ void outer_prod(const float* input, float* output, int batch, int n) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col index
    int z = blockIdx.z; // batch index

    if (x < n && y < n && z < batch) {
        int batch_offset = z * n * n;
        int row_offset = y * n;
        float elem_y = input[z * n + y];
        float elem_x = input[z * n + x];
        output[batch_offset + row_offset + x] = elem_y * elem_x;
    }
}

torch::Tensor custom_forward_naive(torch::Tensor input, torch::Tensor weight1, torch::Tensor bias, torch::Tensor weight2) {
    // Setup kernel execution variables
    const int batch = input.size(0);
    const int n = input.size(1);

    const int threads_per_dim = 16; // Reduced to potentially better fit larger n
    const dim3 blockDim(threads_per_dim, threads_per_dim);
    const dim3 blockDimOuter(threads_per_dim, threads_per_dim);

    const int num_blocks_n = CEIL_DIV(n, threads_per_dim);
    const int num_blocks_batch = CEIL_DIV(batch, threads_per_dim);
    const dim3 gridDimDense(num_blocks_n, num_blocks_batch);
    const dim3 gridDimOut(num_blocks_n, num_blocks_n, batch);

    // Allocate output tensors on the same device as the input tensor
    auto options = input.options();
    auto out1 = torch::empty({batch, n}, options);
    auto out2 = torch::empty({batch, n}, options);
    auto out3 = torch::empty({batch, n, n}, options);

    // Layer 1: Dense (Matrix Multiplication)
    dense_naive<<<gridDimDense, blockDim>>>(
        input.data_ptr<float>(),
        weight1.data_ptr<float>(),
        out1.data_ptr<float>(),
        batch,
        n);
    C10_CUDA_CHECK(cudaGetLastError());

    // Apply Bias and ReLU
    bias_relu<<<gridDimDense, blockDim>>>(
        out1.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch,
        n);
    C10_CUDA_CHECK(cudaGetLastError());

    // Layer 2: Dense (Matrix Multiplication)
    dense_naive<<<gridDimDense, blockDim>>>(
        out1.data_ptr<float>(),
        weight2.data_ptr<float>(),
        out2.data_ptr<float>(),
        batch,
        n);
    C10_CUDA_CHECK(cudaGetLastError());

    // Square Operation
    square<<<gridDimDense, blockDim>>>(
        out2.data_ptr<float>(),
        batch,
        n);
    C10_CUDA_CHECK(cudaGetLastError());

    // Layer 3: Outer Product
    outer_prod<<<gridDimOut, blockDimOuter>>>(
        out2.data_ptr<float>(),
        out3.data_ptr<float>(),
        batch,
        n);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return out3;
}

