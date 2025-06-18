import torch
import time
from wrapper import custom_forward_naive, custom_forward_opt, torch_forward

batch, n = 10, 2200
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def test_naive_shape():
    """
    Tests the shape of the output from the naive implementation.
    """
    input = torch.randn(batch, n, device='cuda')
    weight1 = torch.randn(n, n, device='cuda')
    bias = torch.randn(n, device='cuda')
    weight2 = torch.randn(n, n, device='cuda')
    out = custom_forward_naive(input, weight1, bias, weight2)
    if out.shape == (batch, n, n):
        print("Naive Impl. Shape Test PASSED")
    else: 
        print("Naive Impl. Shape Test NOT PASSED")
 
def test_opt_shape():
    """
    Tests the shape of the output from the optimized implementation.
    """
    input = torch.randn(batch, n, device='cuda')
    weight1 = torch.randn(n, n, device='cuda')
    bias = torch.randn(n, device='cuda')
    weight2 = torch.randn(n, n, device='cuda')
    out = custom_forward_opt(input, weight1, bias, weight2)
    if out.shape == (batch, n, n):
        print("Optimized Impl. Shape Test PASSED")
    else: 
        print("Optimized Impl. Shape Test NOT PASSED")


def test_naive_values_and_time():
    """
    Compares the output values of the standard torch version
    with the output values of the naive torch version
    """
    torch.manual_seed(42)
    input = torch.randn(batch, n, device='cuda')
    weight1 = torch.randn(n, n, device='cuda')
    bias = torch.randn(n, device='cuda')
    weight2 = torch.randn(n, n, device='cuda')
   
    # Warmup for custom
    for i in range(0, 3):
        _ = custom_forward_naive(input, weight1, bias, weight2)
        torch.cuda.synchronize()

    start_custom = time.perf_counter()
    out_custom = custom_forward_naive(input, weight1, bias, weight2)
    torch.cuda.synchronize()
    end_custom = time.perf_counter()
    print(f"custom_forward_naive time: {(end_custom - start_custom) * 1000:.3f} ms")

    # Warmup for torch
    for i in range(0, 3):
        _ = torch_forward(input, weight1, bias, weight2, batch, n)
        torch.cuda.synchronize()

    start_torch = time.perf_counter()
    out_torch = torch_forward(input, weight1, bias, weight2, batch, n)
    torch.cuda.synchronize()
    end_torch = time.perf_counter()
    print(f"torch_forward time: {(end_torch - start_torch) * 1000:.3f} ms")
    
    if torch.allclose(out_custom, out_torch, rtol=1e-4, atol=1e-5):
        print("Naive Impl. Value Test PASSED")
    else: 
        print("Naive Impl. Value Test NOT PASSED")
        print("-----------out_torch - out_custom ----------")
        print(out_torch - out_custom)

def test_opt_values_and_time():
    """
    Compares the output values of the standard torch version
    with the output values of the naive torch version
    """
    torch.manual_seed(42)
    input = torch.randn(batch, n, device='cuda')
    weight1 = torch.randn(n, n, device='cuda')
    bias = torch.randn(n, device='cuda')
    weight2 = torch.randn(n, n, device='cuda')

    # Warmup for opt
    for i in range(0, 3):
        _ = custom_forward_opt(input, weight1, bias, weight2)
        torch.cuda.synchronize()

    start_opt = time.perf_counter()
    out_opt = custom_forward_opt(input, weight1, bias, weight2)
    torch.cuda.synchronize()
    end_opt = time.perf_counter()
    print(f"custom_forward_opt time: {(end_opt - start_opt) * 1000:.3f} ms")

    # Warmup for torch
    for i in range(0, 3):
        _ = torch_forward(input, weight1, bias, weight2, batch, n)
        torch.cuda.synchronize()

    start_torch = time.perf_counter()
    out_torch = torch_forward(input, weight1, bias, weight2, batch, n)
    torch.cuda.synchronize()
    end_torch = time.perf_counter()
    print(f"torch_forward time: {(end_torch - start_torch) * 1000:.3f} ms")

    if torch.allclose(out_opt, out_torch, rtol=1e-4, atol=1e-5):
        print("Optimized Impl. Value Test PASSED")
    else:
        print("Optimized Impl. Value Test NOT PASSED")
        print("-----------out_torch - out_opt ----------")
        print(out_torch - out_opt)

    end_torch = time.perf_counter()
    print(f"torch_forward time: {(end_torch - start_torch) * 1000:.3f} ms")

    if torch.allclose(out_opt, out_torch, rtol=1e-4, atol=1e-5):
        print("Optimized Impl. Value Test PASSED")
    else:
