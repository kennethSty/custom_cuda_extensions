#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__ void bias_relu_k(float* input, const float* bias,int batch, int n) {
	/*
	 Performs elementwise bias addition and relu
	 */
	int x = blockIdx.x * blockDim.x + threadIdx.x; //feature index
        int y = blockIdx.y * blockDim.y + threadIdx.y; //row (batch) index

	bool is_valid_thread = x < n && y < batch;
        if (is_valid_thread) {
	    input[y * n + x] = fmaxf(0.0, input[y * n + x] + bias[x]);
	}		
}

__host__ void custom_dense_relu_layer(const float* input,
	        const float* weight,	
		const float* bias,
		float* output,
		dim3 gridDim, dim3 blockDim,
		int batch, int n,
		float alpha, float beta,
		cublasHandle_t handle) {

        //Cublas expects col major, but input and weight are row major order
	//i.e. input and weight are transposed in col major representation
	//=> swap weight and input in Sgemm call. 
	cublasSgemm(
		handle,
		CUBLAS_OP_N, 
		CUBLAS_OP_N,
		n, batch, n,
		&alpha,
		weight, n,
		input, n,
		&beta,
		output, n
	);

	bias_relu_k<<<gridDim, blockDim>>>(output, bias, batch, n);
}

__global__ void square_k(float* input, int batch, int n) {
	/*
	Squares each element of input in-place
	*/
        int x = blockIdx.x * blockDim.x + threadIdx.x; //feature index
	int y = blockIdx.y * blockDim.y + threadIdx.y; //row (batch) index
	bool is_valid_thread = y < batch && x < n; 

	if (is_valid_thread) {
	    //Pytorch stores tensor in row major order
	    input[y * n + x] = input[y * n + x] * input[y * n + x]; 
	}
}

__host__ void custom_dense_square_layer(const float* input,
	        const float* weight,	
		float* output,
		dim3 gridDim, dim3 blockDim,
		int batch, int n,
		float alpha, float beta,
		cublasHandle_t handle) {

	cublasSgemm(
		handle,
		CUBLAS_OP_N, 
		CUBLAS_OP_N,
		n, batch, n,
		&alpha,
		weight, n,
		input, n,
		&beta,
		output, n
	);

	square_k<<<gridDim, blockDim>>>(output, batch, n);
}

__global__ void outer_prod_k_opt(const float* input,
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

torch::Tensor custom_forward_opt(torch::Tensor input, torch::Tensor weight1,
		torch::Tensor bias, torch::Tensor weight2) {
	
	//Setup kernel execution variables
	int batch = input.size(0);
	int n = input.size(1);
	
	const int threads_per_dim = 32;
	const int num_blocks_n = CEIL_DIV(n, threads_per_dim);
        const int num_blocks_batch = CEIL_DIV(batch, threads_per_dim);	

	dim3 blockDim(threads_per_dim, threads_per_dim);
	dim3 gridDimDense(num_blocks_n, num_blocks_batch);
        dim3 gridDimOut(num_blocks_n, num_blocks_n, batch);

        //Allocate output tensors on same device as input tensor	
       	auto options = input.options();
	auto out1 = torch::empty({batch, n}, options);
	auto out2 = torch::empty({batch, n}, options);
	auto out3 = torch::empty({batch, n, n}, options);
        
	//Init variables needed for cuda Sgemm and cuda kernels
	cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
	const float alpha = 1.0f, beta = 0.0f;

	//Layer 1: Dense Relu 
        custom_dense_relu_layer(
		input.data_ptr<float>(),
		weight1.data_ptr<float>(),
		bias.data_ptr<float>(),
		out1.data_ptr<float>(),
		gridDimDense, blockDim,
		batch, n, alpha, beta, handle
	);

	//Layer 2: Dense Square
        custom_dense_square_layer(
		out1.data_ptr<float>(),
		weight2.data_ptr<float>(),
		out2.data_ptr<float>(),
		gridDimDense, blockDim,
		batch, n, alpha, beta, handle
	);

	// Layer 3: Outer Product
        outer_prod_k_opt<<<gridDimOut, blockDim>>>(
		out2.data_ptr<float>(),
		out3.data_ptr<float>(),
		batch,
		n
    	);//automatically linked from kernels_naive.cu 

    	cudaDeviceSynchronize();
    	return out3;
}
