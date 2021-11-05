#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__
void matrixTransposeKernel(const int* d_matrix_in,
                           int        N,
                           int*       d_matrix_out) {
    __shared__ int ds_matrix_in[BLOCK_SIZE_X][BLOCK_SIZE_Y];

    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    if(row < N && col < N) {
        ds_matrix_in[threadIdx.y][threadIdx.x] = d_matrix_in[row*N + col];
        __syncthreads();

        row = blockIdx.x * blockDim.x + threadIdx.y;
        col = blockIdx.y * blockDim.y + threadIdx.x;

        d_matrix_out[row*N + col] = ds_matrix_in[threadIdx.x][threadIdx.y];
    }
}

const int N  = 1000;

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_matrix_in  = new int[N * N];
    int* h_matrix_tmp = new int[N * N]; // <-- used for device result
    int* h_matrix_out = new int[N * N];

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N * N; i++)
        h_matrix_in[i] = distribution(generator);

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = 0; i < N ; i++) {
        for (int j = 0; j < N ; j++)
            h_matrix_out[i * N + j] = h_matrix_in[j * N + i];
    }

    TM_host.stop();
    TM_host.print("MatrixTranspose host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_matrix_in, *d_matrix_out;
    SAFE_CALL( cudaMalloc( &d_matrix_in, N*N * sizeof(int) ) );
    SAFE_CALL( cudaMalloc( &d_matrix_out, N*N * sizeof(int) ) );

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_matrix_in, h_matrix_in, N*N * sizeof(int), cudaMemcpyHostToDevice ) );

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    dim3 num_blocks(N/BLOCK_SIZE_X, N/BLOCK_SIZE_Y, 1);
    if (N % BLOCK_SIZE_X) num_blocks.x++;
    if (N % BLOCK_SIZE_Y) num_blocks.y++;
    dim3 dim_block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    matrixTransposeKernel<<< num_blocks, dim_block >>>(d_matrix_in, N, d_matrix_out);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("MatrixTranspose device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( h_matrix_tmp, d_matrix_out, N*N * sizeof(int), cudaMemcpyDeviceToHost ) );


    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N * N; i++) {
        if (h_matrix_out[i] != h_matrix_tmp[i]) {
            std::cerr << "wrong result at: ("
                      << (i / N) << ", " << (i % N) << ")"
                      << "\nhost:   " << h_matrix_out[i]
                      << "\ndevice: " << h_matrix_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_matrix_in;
    delete[] h_matrix_out;
    delete[] h_matrix_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_matrix_in ) );
    SAFE_CALL( cudaFree( d_matrix_out ) );

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
