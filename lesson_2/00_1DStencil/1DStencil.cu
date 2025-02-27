#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int RADIUS = 7;
const int DIM_BLOCK = 256;

__global__
void stencilKernel(const int* d_input, int N, int* d_output) {
    __shared__ int d_sm[DIM_BLOCK + 2 * RADIUS];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    d_sm[threadIdx.x] = d_input[global_id];

    if (threadIdx.x < RADIUS) {
        d_sm[threadIdx.x + DIM_BLOCK] = d_input[blockDim.x + global_id];
        d_sm[threadIdx.x + RADIUS + DIM_BLOCK] = d_input[global_id + RADIUS + blockDim.x];
    }
    __syncthreads();

    if (global_id < N - 2 * RADIUS) {
        int v = 0;
        for (int i = 0; i < RADIUS * 2 + 1; ++i)
            v += d_sm[threadIdx.x + i];
        d_output[global_id + RADIUS] = v;
    }
}

const int N  = 10000000;

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_input      = new int[N];
    int* h_output_tmp = new int[N]; // <-- used for device result
    int* h_output     = new int[N](); // initilization to zero

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++)
        h_input[i] = distribution(generator);

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = RADIUS; i < N - RADIUS; i++) {
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            h_output[i] += h_input[j];
    }

    TM_host.stop();
    TM_host.print("1DStencil host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_input, *d_output;
    SAFE_CALL( cudaMalloc( &d_input, N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_output, N * sizeof(int) ));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // did you miss something?
    dim3 DimGrid(N/256, 1, 1);
    if (N%256) DimGrid.x++;
    dim3 DimBlock(DIM_BLOCK, 1, 1);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    stencilKernel<<<DimGrid, DimBlock >>>(d_input,N,d_output);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("1DStencil device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( h_output_tmp, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << h_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_input ) );
    SAFE_CALL( cudaFree( d_output ) );

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
