import os
import torch
import ctypes

from utils.utils import *

# 1) load .so
lib_path = os.path.abspath("./libvector_add.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

# 2) signature: solve(const float* A, const float* B, float* C, int M, int N, int K)
lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.solve.restype = None

@timing
def solve_cuda(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    lib.solve(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )

@timing
def solve_torch(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    
    torch.matmul(A,B,out=C)
    
    
if __name__ == "__main__":
    A : torch.Tensor = torch.tensor([[1.0, 2.0],[3.0,4.0]]).to("cuda")
    B : torch.Tensor =  torch.tensor([[5.0, 6.0],[ 7.0, 8.0]]).to("cuda")
    M,K = A.shape
    _,N = B.shape
    C: torch.Tensor = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    solve_cuda(A,B,C,M,N,K)
    
    print(C)
