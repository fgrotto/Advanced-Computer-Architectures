#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

// Macros
#define DIV(a, b)   (((a) + (b) - 1) / (b))

const int N  = 16777216;
const int SegSize = 8388608;
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
    
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
	int* devVectorIN0;
    int* devVectorIN1;
	SAFE_CALL( cudaMalloc(&devVectorIN0, SegSize * sizeof(int)) );
    SAFE_CALL( cudaMalloc(&devVectorIN1, SegSize * sizeof(int)) );
	
	int sum1, sum2, sum;
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dev_TM.start();
    
    SAFE_CALL( cudaMemcpyAsync(devVectorIN0, VectorIN, SegSize * sizeof(int),
             cudaMemcpyHostToDevice, stream0) );
    ReduceKernelSharedLessDivergence<<<DIV(SegSize, BLOCK_SIZE), BLOCK_SIZE, 1024 , stream0>>>
                        (devVectorIN0, SegSize);
    ReduceKernelSharedLessDivergence<<<DIV(SegSize, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE, 1024 , stream0>>>
                         (devVectorIN0, DIV(SegSize, BLOCK_SIZE));
    ReduceKernelSharedLessDivergence<<<DIV(SegSize, BLOCK_SIZE * BLOCK_SIZE * 128), 128, 1024 , stream0>>>
                         (devVectorIN0, DIV(SegSize, BLOCK_SIZE * 128));

    SAFE_CALL( cudaMemcpyAsync(devVectorIN1, VectorIN+SegSize, SegSize * sizeof(int),
             cudaMemcpyHostToDevice, stream1) );
    ReduceKernelSharedLessDivergence<<<DIV(SegSize, BLOCK_SIZE), BLOCK_SIZE, 1024 , stream1>>>
                        (devVectorIN1, N);
    ReduceKernelSharedLessDivergence<<<DIV(SegSize, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE, 1024 , stream1>>>
                         (devVectorIN1, DIV(SegSize, BLOCK_SIZE));
    ReduceKernelSharedLessDivergence<<<DIV(SegSize, BLOCK_SIZE * BLOCK_SIZE * 128), 128, 1024 , stream1>>>
                         (devVectorIN1, DIV(SegSize, BLOCK_SIZE * 128));

	dev_TM.stop();
	dev_time = dev_TM.duration();
	CHECK_CUDA_ERROR;

	SAFE_CALL( cudaMemcpy(&sum1, devVectorIN0, sizeof(int),
                            cudaMemcpyDeviceToHost) );
    SAFE_CALL( cudaMemcpy(&sum2, devVectorIN1, sizeof(int),
                            cudaMemcpyDeviceToHost) );
    sum = sum1+sum2;
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
    SAFE_CALL( cudaFree(devVectorIN0) );
    SAFE_CALL( cudaFree(devVectorIN1) );
    cudaDeviceReset();
}
