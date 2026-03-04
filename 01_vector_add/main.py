import os
import torch
import ctypes

from utils.utils import *

# 1) load .so
lib_path = os.path.abspath("./libvector_add.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

# 2) signature: solve(const float* A, const float* B, float* C, int N)
lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.solve.restype = None

@timing
def solve_cuda(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    # call your solve with device pointers
    lib.solve(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(N),
    )

@timing
def solve_torch(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A,B,out=C)

def main():
    tests = [1, 2, 3, 4, 7, 31, 32, 33, 255, 256, 257, 1024, 10000, 1_000_000, 5_000_000]
    for i, N in enumerate(tests):
        A = create_matrix('float32', N)
        B = create_matrix('float32', N)
        C_1 = torch.empty_like(A).to("cuda")
        C_2 = torch.empty_like(A).to("cuda")

        solve_torch(A,B,C_1,N)
        solve_cuda(A,B,C_2,N)  

        check_result(C_1,C_2)
        print(10*"=")




    print("All tests passed ✅")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this environment")
    main()