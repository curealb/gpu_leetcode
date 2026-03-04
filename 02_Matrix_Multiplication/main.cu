#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

// one by one
// __global__ void  matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K){
//     int col = threadIdx.x + blockIdx.x*blockDim.x;
//     int row = threadIdx.y + blockIdx.y*blockDim.y;
//     int col_stride = gridDim.x * blockDim.x;
//     int row_stride = gridDim.y * blockDim.y;
//     for (int r = row; r < M; r += row_stride) {
//         for (int c = col; c < K; c += col_stride) {
//             float sum = 0;
//             for (int i = 0; i < N; i++) {
//                 sum += A[r * M + i] * B[i * K + c];
//             }
//             C[r * K + c] = sum;
//         }
//     }
// }


// Use shared matrix
__global__ void matrix_multiplication_kernel(const float* __restrict__ A, const float* __restrict__ B, float*  __restrict__ C, int M, int N,
    int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = threadIdx.y + blockIdx.y * TILE;  // [0,M)
    int col = threadIdx.x + blockIdx.x * TILE;  // [0,N)

    float acc = 0.0f;
    
    
    for(int t=0;t<(K+TILE-1)/TILE;++t){
        int a_col = threadIdx.x + t*TILE; // [0.k)
        int b_row = threadIdx.y + t*TILE; // [0,k)

        // Fill tile matrix
        As[threadIdx.y][threadIdx.x] = (row<M&&a_col<K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =(b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Compute tile matrix
        for(int i=0;i<TILE;++i){
            acc+=As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
