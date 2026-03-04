import torch
import ctypes

def torch_solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    
    torch.matmul(A,B,out=C)
    
    
if __name__ == "__main__":
    A : torch.Tensor = torch.tensor([[1.0, 2.0],[3.0,4.0]]).to("cuda")
    B:torch.Tensor =  torch.tensor([[5.0, 6.0],[ 7.0, 8.0]]).to("cuda")
    M,K = A.shape
    _,N = B.shape
    C: torch.Tensor = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    torch_solve(A,B,C,M,N,K)
    
    print(C)