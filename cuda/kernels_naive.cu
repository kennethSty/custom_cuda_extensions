#include <torch/extension.h>
#include <cuda_runtime.h>
#include "error_handling.h"

#define CEIL_DIV(a, b) ((a + b - 1)/b)

__device__ void dot_product_k(float& sum, const float* input, const float* weight, 
		int x, int y, int n) {
	/*
	   Computes output element (y,x) in matrix multiplication 
	*/

	for (int k = 0; k < n; k++) {
	    sum += input[y * n + k] * weight[k * n + x];
	}
}

__global__ void dense_relu_k(const float* input, const float* weight, 
		const float* bias, float* output, int batch, int n) {
	/*
	Cuda kernel computing the output of a dense relu layer in place.
	Assumes start of 1D grid with 'batch' number of blocks of size (n).
	A thread computes the j-th out of n elements of the bth out of 'batch' vectors.
	*/
	int y = blockIdx.y * blockDim.y + threadIdx.y; //row index
	int x = blockIdx.x * blockDim.x + threadIdx.x; //col index
        bool is_valid_thread = y < batch && x < n;

	if (is_valid_thread) {

	    float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += input[y * n + k] * weight[k * n + x];
	    }
	    output[y * n + x] = fmaxf(0.0, sum + bias[x]);
	}
}

__global__ void dense_square_k(const float* input, const float* weight,
	       	float* output, int batch, int n) {
	/*
	Cuda kernel computing the output of a dense square layer in place.
	Assumes start of 1D grid with 'batch' number of blocks of size (n).
	A thread computes the j-th out of n elements of the bth out of 'batch' vectors.
	*/
	int y = blockIdx.y * blockDim.y + threadIdx.y; //row index
	int x = blockIdx.x * blockDim.x + threadIdx.x; //col index
        bool is_valid_thread = x < n && y < batch;
	
	if(is_valid_thread) {
	    float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += input[y * n + k] * weight[k * n + x];
            }
            output[y * n + x] = sum * sum;
	}
}	

__global__ void outer_prod_k(const float* input,
		float* output, int batch, int n) {
	/*
	Cuda kernel computing the outer product of the input vector in place.
	Assumes start of 2d grid with 'batch' number of blocks of size (n, n).
	Reason for using 2d: Task is inherently 2d -> one matrix per input vector.
	A thread computes the (i,j) element of the nxn matrix for 
	the b-th vector out of 'batch' input vectors. 
	*/
        int y = blockIdx.y * blockDim.y + threadIdx.y; //row index
	int x = blockIdx.x * blockDim.x + threadIdx.x; //col index
	int z = blockIdx.z; //batch index
	bool is_valid_thread = (x < n) && (y < n) && (z < batch);

	if (is_valid_thread) {
	    int batch_offset = z * n * n;
	    int row_offset = y * n;
	    float elem_y = input[z * n + y];
	    float elem_x = input[z * n + x];
	    output[batch_offset + row_offset + x] = elem_y * elem_x;    
	}
}

torch::Tensor custom_forward_naive(torch::Tensor input, torch::Tensor weight1,
	      torch::Tensor bias, torch::Tensor weight2) {
	/*
	Host function launching the three kernels in forward-pass-like manner.
	*/

	//Setup kernel execution variables
	int batch = input.size(0);
	int n = input.size(1);

	int threads_per_block = 16;
	int num_blocks_n = CEIL_DIV(n, threads_per_block);
        int num_blocks_batch = CEIL_DIV(batch, threads_per_block);

	dim3 blockDim(threads_per_block, threads_per_block);
	dim3 gridDimDense(num_blocks_n, num_blocks_batch);
	dim3 gridDimOut(num_blocks_n, num_blocks_n, batch);


	//Allocate output tensors on same device as input tensor
	auto options = input.options();
	auto out1 = torch::empty({batch, n}, options);
	auto out2 = torch::empty({batch, n}, options);
	auto out3 = torch::empty({batch, n, n}, options);

	//Layer 1
	dense_relu_k<<<gridDimDense, blockDim>>>(
			input.data_ptr<float>(), 
			weight1.data_ptr<float>(),
			bias.data_ptr<float>(),
			out1.data_ptr<float>(),
			batch,
			n);

        HANDLE_ERROR(cudaGetLastError());	
	
	//Layer 2
	dense_square_k<<<gridDimDense, blockDim>>>(
			out1.data_ptr<float>(),
	       	        weight2.data_ptr<float>(),
	                out2.data_ptr<float>(),
	               	batch,
			n);
	
	HANDLE_ERROR(cudaGetLastError());

	//Layer 3
	outer_prod_k<<<gridDimOut, blockDim>>>(
			out2.data_ptr<float>(),
			out3.data_ptr<float>(), 
			batch,
			n);
	
	HANDLE_ERROR(cudaGetLastError());
	
	HANDLE_ERROR(cudaDeviceSynchronize());
	return out3;	
}
