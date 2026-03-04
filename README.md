# gpu_leetcode

CUDA practice repo with one exercise per directory.

## Layout

- `01_vector_add/`: vector addition kernel and Python verification script
- `02_Matrix_Multiplication/`: matrix multiplication exercise
- `utils/`: shared Python and C/CUDA helpers

## Run Exercises

Run from the repo root through the shared launcher:

```bash
uv run python run.py 01_vector_add
uv run python run.py 02_Matrix_Multiplication
```

This preserves each exercise's own working directory while keeping root-level modules like `utils.utils` importable.
