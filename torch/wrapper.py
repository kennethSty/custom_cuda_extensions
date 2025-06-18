import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

#Loads the custom_kernels module
module_path = os.path.dirname(__file__)
cuda_path = os.path.join(module_path, "..", "cuda")

custom_kernels = load(
    name="custom_kernels",
    sources=[
        os.path.join(cuda_path, "binding.cpp"),
        os.path.join(cuda_path, "kernels_naive.cu"),
        os.path.join(cuda_path, "kernels_optimized.cu")
    ],
    verbose=True,
)

def custom_forward_naive(input, weight1, bias, weight2, debug = False): 
    """
    Forward pass using naive custom kernels for n <= 32
    """
    return custom_kernels.custom_forward_naive(
        input.contiguous(),
        weight1.contiguous(),
        bias.contiguous(),
        weight2.contiguous()
    )


def custom_forward_opt(input, weight1, bias, weight2):
    """
    Forward pass using optimized custom kernels.
    """
    return custom_kernels.custom_forward_opt(
        input.contiguous(),
        weight1.contiguous(),
        bias.contiguous(),
        weight2.contiguous()
    )

def torch_forward(x, weight1, bias, weight2, batch, n):
    """
    Forward pass using standard PyTorch operations.
    """
    x = x @ weight1
    x = x + bias
    x = F.relu(x)
    x = x @ weight2
    x = x ** 2
    
    rows = x.reshape(batch, n, 1) # each vector is a col vector of dim (n,1)
    cols = x.reshape(batch, 1, n) # each vector is a row vector of dim (1, n)

    # Broadcasting: for each of the 'batch' vectors compute outer product.
    # (batch, n, 1) @ (batch, 1, n) -> (batch, n, n)
    return rows @ cols

def torch_forward_debug(input, weight1, bias, weight2, batch, n):
    x1 = input @ weight1            # (B, N)
    x2 = x1 + bias                  # (B, N)
    x3 = torch.relu(x2)            # (B, N)
    x4 = x3.unsqueeze(2) @ x3.unsqueeze(1)  # (B, N, N)
    return x1, x2, x3, x4

