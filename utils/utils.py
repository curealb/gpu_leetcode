import time
import torch
from functools import wraps


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        start_event.record() 
        result = func(*args, **kwargs)
        end_event.record()   
        
        end_event.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"[{func.__name__}] took {elapsed_ms:.6f} ms")
        
        return result

    return wrapper

def create_matrix(dtype_str: str = 'float32', *dims: int) -> torch.Tensor:
    t_dtype = getattr(torch, dtype_str, torch.float32)
    return torch.randn(dims, device="cuda", dtype=t_dtype)

def check_result(my_tensor: torch.Tensor, ref_tensor: torch.Tensor):
    if torch.allclose(my_tensor, ref_tensor, atol=1e-5):
        print("✅ Success!")
    else:
        diff = (my_tensor - ref_tensor).abs().max()
        print(f"❌ Failed! Max diff: {diff}")