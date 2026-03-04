#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

__global__ void matrix_multiplication_kernel(const float* __restrict__ A, const float* __restrict__ B, float*  __restrict__ C, int M, int N,
    int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // [0,M)
    int col = threadIdx.x + blockIdx.x * TILE;  // [0,N)

    float acc = 0.0;

    

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
