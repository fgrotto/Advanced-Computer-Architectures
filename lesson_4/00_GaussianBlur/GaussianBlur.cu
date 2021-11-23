#include <chrono>
#include <math.h>
#include <iomanip>
#include <iostream>
#include "Timer.cuh"
#include "CheckError.cuh"
#include <opencv2/opencv.hpp>
using namespace timer;

const int N = 10; // it has to be divisible by 2
const int WIDTH  =1000;
const int HEIGHT = 500;
const int CHANNELS = 3;
const int BLOCK_SIZE = 32;

template <class T>
void printImageForDebug(T *image, int N, int width, int channels)
{
	std::cout << std::endl;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			std::cout << "[ ";
			for (int k = 0; k < channels; k++)
			{
				std::cout << image[(i * width + j) * channels + k];
				k == channels - 1 ? std::cout << " " : std::cout << " , ";
			}
			std::cout << "]";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


void buildGaussianFilterMask(float *Mask, int N, float sigma)
{	
    double r, s = 2.0 * sigma * sigma;
    double sum = 0.0;

    for (int x = -N/2; x < N/2; x++) {
        for (int y = -N/2; y < N/2; y++) {
            r = sqrt(x * x + y * y);
            Mask[(x + N/2) * N + (y + N/2)] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += Mask[(x + N/2) * N + (y + N/2)];
        }
    }
 
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Mask[i *N + j] /= sum;
}

__global__ void GaussianBlurDevice(const unsigned char *image,
								   const float *mask,
								   unsigned char *image_out,
								   int N)
{

	int globalId_x = threadIdx.x + blockIdx.x * blockDim.x;
	int globalId_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (globalId_x < WIDTH && globalId_y < HEIGHT)
	{
		for (int channel = 0; channel < CHANNELS; channel++)
		{
			float pixel_value = 0;
			for (int u = 0; u < N; u++)
			{
				for (int v = 0; v < N; v++)
				{
					int new_x = min(WIDTH, max(0, globalId_x + u - N / 2));
					int new_y = min(HEIGHT, max(0, globalId_y + v - N / 2));
					pixel_value += mask[v * N + u] * image[(new_y * WIDTH + new_x) * CHANNELS + channel];
				}
			}
			image_out[(globalId_y * WIDTH + globalId_x) * CHANNELS + channel] = (unsigned char)pixel_value;
		}
	}
}

void GaussianBlurHost(const unsigned char *image,
					  const float *mask,
					  unsigned char *image_out)
{

	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			for (int channel = 0; channel < CHANNELS; channel++)
			{
				float pixel_value = 0;
				for (int u = 0; u < N; u++)
				{
					for (int v = 0; v < N; v++)
					{
						int new_x = min(WIDTH, max(0, x + u - N / 2));
						int new_y = min(HEIGHT, max(0, y + v - N / 2));
						pixel_value += mask[v * N + u] * image[(new_y * WIDTH + new_x) * CHANNELS + channel];
					}
				}
				image_out[(y * WIDTH + x) * CHANNELS + channel] = (unsigned char)pixel_value;
			}
		}
	}
}

int main()
{
	Timer<DEVICE> TM_device;
	Timer<HOST> TM_host;

	cv::Mat img = cv::imread("../image.png");

	if (img.empty())
	{
		std::cout << "Failed imread(): image not found" << std::endl;
		exit(0);
	}

	// cv::namedWindow("Display window");
	// cv::imshow("Display window", img);
	// cv::waitKey(0);

	// -------------------------------------------------------------------------
	// HOST MEMORY ALLOCATION
	unsigned char *image = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	unsigned char *host_image_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
	unsigned char *device_image_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];

	float *mask = new float[N * N];
	float sigma = 1.0;
	image = img.data;

	// Build a gaussian filter for the image
	buildGaussianFilterMask(mask, N, sigma);

	// -------------------------------------------------------------------------
	// HOST EXECUTIION
	TM_host.start();

	GaussianBlurHost(image, mask, host_image_out);

	TM_host.stop();
	TM_host.print("GaussianBlur host:   ");

	// cv::Mat A(HEIGHT, WIDTH, CV_8UC3, host_image_out);
	// cv::imshow("Result of gaussian blur (host)", A);
	// cv::waitKey(0);

	// -------------------------------------------------------------------------
	// DEVICE MEMORY ALLOCATION

	unsigned char *dev_image, *dev_image_out;
	float *dev_mask;

	SAFE_CALL(cudaMalloc(&dev_image, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char)));
	SAFE_CALL(cudaMalloc(&dev_image_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char)));
	SAFE_CALL(cudaMalloc(&dev_mask, N * N * sizeof(float)));

	// -------------------------------------------------------------------------
	// COPY DATA FROM HOST TO DEVICE

	SAFE_CALL(cudaMemcpy(dev_image, image, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(dev_mask, mask, N * N * sizeof(float), cudaMemcpyHostToDevice));

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 num_blocks(ceil(float(WIDTH) / BLOCK_SIZE), ceil(float(HEIGHT) / BLOCK_SIZE), 1);

	TM_device.start();

	GaussianBlurDevice<<<block_size, num_blocks>>>(dev_image, dev_mask, dev_image_out, N);

	TM_device.stop();
	CHECK_CUDA_ERROR
	TM_device.print("GaussianBlur device: ");

	std::cout << std::setprecision(1)
			  << "Speedup: " << TM_host.duration() / TM_device.duration()
			  << "x\n\n";

	// -------------------------------------------------------------------------
	// COPY DATA FROM DEVICE TO HOST

	SAFE_CALL(cudaMemcpy(device_image_out, dev_image_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------
	// RESULT CHECK

	// cv::Mat B(HEIGHT, WIDTH, CV_8UC3, device_image_out);
	// cv::imshow("Result of gaussian blur (device)", B);
	// cv::waitKey(0);

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			if (device_image_out[i * WIDTH + j] != host_image_out[i * WIDTH + j] 
					|| device_image_out[i * WIDTH + j] != host_image_out[i * WIDTH + j])
			{
				std::cerr << "wrong result at [" << i << "][" << j << "]!" << std::endl;
				std::cerr << "image_out: " << (short)host_image_out[i * WIDTH + j] << std::endl;
				std::cerr << "device_image_out: " << (short)device_image_out[i * WIDTH + j] << std::endl;
				cudaDeviceReset();
				std::exit(EXIT_FAILURE);
			}
		}
	}
	std::cout << "<> Correct\n\n";

	// -------------------------------------------------------------------------
	// HOST MEMORY DEALLOCATION
	delete[] host_image_out;
	delete[] device_image_out;
	delete[] mask;

	// -------------------------------------------------------------------------
	// DEVICE MEMORY DEALLOCATION
	SAFE_CALL(cudaFree(dev_image))
	SAFE_CALL(cudaFree(dev_image_out))
	SAFE_CALL(cudaFree(dev_mask))

	// -------------------------------------------------------------------------
	//SAFE_CALL(cudaFree());
	cudaDeviceReset();
}
