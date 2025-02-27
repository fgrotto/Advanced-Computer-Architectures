#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

// Macros
#define DIV(a, b)   (((a) + (b) - 1) / (b))

const int N  = 16777216;
#define BLOCK_SIZE 256

__global__ void ReduceKernel(int* VectorIN, int N) {
    __shared__ int SMem[1024];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    SMem[threadIdx.x] = VectorIN[global_id];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (i * 2) == 0) 
            SMem[threadIdx.x] += SMem[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) 
        VectorIN[blockIdx.x] = SMem[0];
}

__global__ void ReduceKernelNaive(int* VectorIN, int N) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (i * 2) == 0) 
            VectorIN[global_id] += VectorIN[global_id + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) 
        VectorIN[blockIdx.x] = VectorIN[global_id];
}

__global__ void ReduceKernelSharedLessDivergence(int* VectorIN, int N) {
    __shared__ int SMem[1024];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    SMem[threadIdx.x] = VectorIN[global_id];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = threadIdx.x * i * 2;
        if (index < blockDim.x)
            SMem[index] += SMem[index + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) 
        VectorIN[blockIdx.x] = SMem[0];
}

int main() {
    
    // ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    Timer<HOST> host_TM;
    Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------
    
	int* devVectorIN;
	SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
	
	int sum;
	float dev_time;
    SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int),
       cudaMemcpyHostToDevice) );

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dev_TM.start();
    
    ReduceKernelSharedLessDivergence<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>
            (devVectorIN, N);
    ReduceKernelSharedLessDivergence<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE>>>
                (devVectorIN, DIV(N, BLOCK_SIZE));
    ReduceKernelSharedLessDivergence<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>
                (devVectorIN, DIV(N, BLOCK_SIZE * BLOCK_SIZE));

    dev_TM.stop();
    
    SAFE_CALL( cudaMemcpy(&sum, devVectorIN, sizeof(int),
        cudaMemcpyDeviceToHost) );
    
    dev_time = dev_TM.duration();
	CHECK_CUDA_ERROR;
		
	// ------------------- HOST ------------------------------------------------
    host_TM.start();

	int host_sum = std::accumulate(VectorIN, VectorIN + N, 0);

    host_TM.stop();

    std::cout << std::setprecision(3)
              << "KernelTime Divergent: " << dev_time << std::endl
              << "HostTime            : " << host_TM.duration() << std::endl
              << std::endl;

    // ------------------------ VERIFY -----------------------------------------

    if (host_sum != sum) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }

    //-------------------------- SPEEDUP ---------------------------------------

    float speedup = host_TM.duration() / dev_time;

    std::cout << "Correct result" << std::endl
              << "Speedup achieved: " << std::setprecision(3)
              << speedup << " x" << std::endl << std::endl;

    std::cout << host_TM.duration() << ";" << dev_TM.duration() << ";" << host_TM.duration() / dev_TM.duration() << std::endl;

    delete[] VectorIN;
    SAFE_CALL( cudaFree(devVectorIN) );
    cudaDeviceReset();
}
