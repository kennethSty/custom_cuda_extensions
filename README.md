![Header](CudaHeader.png)
# CUDA Custom Layers
This repository implements a custom forward pass for a neural operation using CUDA and integrates it into PyTorch via C++/CUDA extensions. It supports:

- A **naive implementation** with custom CUDA kernels  
- An **optimized version** using cuBLAS  
- A **PyTorch reference** version for correctness and performance comparison  

## Forward Pass Description

The operation consists of the following layers:

1. **Linear + Bias + ReLU**: `x1 = ReLU(input @ weight1 + bias)`  
2. **Linear + Square**:   `x2 = (x1 @ weight2) ** 2` (element-wise)  
3. **Outer Product**:    `out = x2 ⊗ x2` → shape: `(batch, n, n)`

## File Overview

| File                   | Description                                       |
|------------------------|---------------------------------------------------|
| `binding.cpp`          | Pybind11 interface binding CUDA kernels to Python |
| `kernels_naive.cu`     | Naive CUDA implementation of the forward pass     |
| `kernels_optimized.cu` | Optimized CUDA version using cuBLAS               |
| `wrapper.py`           | Python interface to load and run kernels          |
| `test.py`              | Benchmarking and correctness testing              |
| `run_test.sh`          | Shell script to compile and run tests             |
| `requirements.txt`     | Python dependencies                               |

## Setup

### 1. Create Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n kernel_env python=3.11 -y

# Activate the environment
conda activate kernel_env
```

### 2. Install Dependencies 

```bash
# Install additional dependencies from requirements.txt
pip install -r requirements.txt
```

## Usage

### Running Tests Locally

To run the test directly on your local machine or login node:

```bash
python test.py
```

### Running Tests on a Cluster

For cluster environments with SLURM job scheduler:

```bash
sbatch run_test.sh
```

This will submit a job to the cluster and run the tests on a compute node with GPU access.

## Details About Custom Kernels

The project implements three custom CUDA kernels:

1. **Dense ReLU Layer** (`dense_relu_k`) - Performs matrix multiplication followed by bias addition and ReLU activation
2. **Dense Square Layer** (`dense_square_k`) - Performs matrix multiplication and squares the result
3. **Outer Product Layer** (`outer_prod_k`) - Computes the outer product of input vectors

## Troubleshooting

### Common Issues

1. **Ninja not found**: Make sure ninja is installed via pip
2. **CUDA compilation errors**: Ensure CUDA toolkit is properly installed and accessible
3. **PyTorch version mismatch**: Verify PyTorch is compiled with CUDA support

## Requirements

- Python 3.11
- PyTorch >= 2.0.0 with CUDA support
- CUDA Toolkit >= 12.1
- Ninja build system
- C++ compiler with C++17 support

## Notes

- The code uses PyTorch's JIT compilation to build the CUDA extensions at runtime
- Make sure you have appropriate GPU access when running the tests
- The kernels are optimized for educational purposes


