#include <torch/extension.h>

//Forward declaration. Implementation in kernels.cu
torch::Tensor custom_forward_naive(
        torch::Tensor input, 
        torch::Tensor weight1,
        torch::Tensor bias,
        torch::Tensor weight2);

torch::Tensor custom_forward_opt(
        torch::Tensor input, 
        torch::Tensor weight1,
        torch::Tensor bias,
        torch::Tensor weight2);

// Bind both functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_forward_naive", &custom_forward_naive, "Naive forward pass through custom layers");
    m.def("custom_forward_opt", &custom_forward_opt, "Optimized forward pass through custom layers");
}

