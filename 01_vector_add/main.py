# import torch
# import ctypes


# def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
#     torch.add(A,B,out=C)

# #####
# lib = ctypes.cdll.LoadLibrary("./libvector_add.so")

# lib.solve.argtypes = [
#     ctypes.c_void_p,  # A device ptr
#     ctypes.c_void_p,  # B device ptr
#     ctypes.c_void_p,  # C device ptr
#     ctypes.c_int      # N
# ]
# lib.solve.restype = None

# if __name__ == "__main__":
#     A : torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
#     B:torch.Tensor =  torch.tensor([5.0, 6.0, 7.0, 8.0])
#     N = A.shape[0]
#     C:torch.Tensor = torch.empty_like(A)
    
#     solve(A,B,C,N)
    
#     print(C)
    
    
    
import ctypes
import os
import torch

# 1) load .so
lib_path = os.path.abspath("./libvector_add.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

# 2) signature: solve(const float* A, const float* B, float* C, int N)
lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.solve.restype = None


def run_one(N: int, seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.synchronize()

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    # call your solve with device pointers
    lib.solve(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(N),
    )

    # reference
    ref = A + B

    max_err = (C - ref).abs().max().item()
    return max_err


def main():
    # correctness: edge + random
    tests = [1, 2, 3, 4, 7, 31, 32, 33, 255, 256, 257, 1024, 10000]
    for i, N in enumerate(tests):
        err = run_one(N, seed=123 + i)
        print(f"N={N:>8d}  max_err={err:.3e}")
        assert err < 1e-5, f"FAILED at N={N}, err={err}"

    # optional: larger random sanity
    for N in [1_000_000, 5_000_000]:
        err = run_one(N, seed=42)
        print(f"[large] N={N} max_err={err:.3e}")
        assert err < 1e-5

    print("All tests passed ✅")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this environment")
    main()