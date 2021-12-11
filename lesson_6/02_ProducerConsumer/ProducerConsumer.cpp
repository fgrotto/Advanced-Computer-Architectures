#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "Timer.hpp"

void test_producer_consumer(int Buffer[32]) {
	int i = 0;
	int count = 0;

	while (i < 35000) {					// number of test

		// PRODUCER
		if ((rand() % 50) == 0) {		// some random computations

			if (count < 31) {
				++count;
				std::cout << "Thread:\t" << omp_get_thread_num()
                          << "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations

			if (count >= 1) {
				int var = Buffer[count];
				std::cout << "Thread:\t" << omp_get_thread_num()
                          << "\tConsume on index: " << count
                          << "\tvalue: " << var << std::endl;
				--count;
			}
		}
		i++;
	}
}

void test_producer_consumer_critical(int Buffer[32]) {
	int i;
	int count = 0;

	#pragma omp parallel for private(i) shared(Buffer, count) schedule(auto)
	for (i = 0; i < 35000; i++) {
		// PRODUCER
		if ((rand() % 50) == 0) {
			#pragma omp critical(buffer)
			{
				if (count < 31) {
					++count;
					std::cout << "Thread:\t" << omp_get_thread_num()
							<< "\tProduce on index: " << count << std::endl;
					Buffer[count] = omp_get_thread_num();
				}
			}
		}
		// CONSUMER
		if ((std::rand() % 51) == 0) {
			#pragma omp critical(buffer)
			{
				if (count >= 1) {
					
					int var = Buffer[count];
					std::cout << "Thread:\t" << omp_get_thread_num()
							<< "\tConsume on index: " << count
							<< "\tvalue: " << var << std::endl;
					--count;
				}
			}
		}
		i++;
	}
}

void test_producer_consumer_locks(int Buffer[32]) {
	int count = 0;
	int i;

	omp_lock_t lock;
    omp_init_lock(&lock);

	#pragma omp parallel for private(i) shared(Buffer, count) schedule(auto)
	for (i = 0; i < 35000; i++) {
		// PRODUCER
		if ((rand() % 50) == 0) {
			omp_set_lock(&lock);
			if (count < 31) {
				++count;
				std::cout << "Thread:\t" << omp_get_thread_num()
						<< "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
			omp_unset_lock(&lock);
		}
		// CONSUMER
		if ((std::rand() % 51) == 0) {
			omp_set_lock(&lock);
			if (count >= 1) {
				
				int var = Buffer[count];
				std::cout << "Thread:\t" << omp_get_thread_num()
						<< "\tConsume on index: " << count
						<< "\tvalue: " << var << std::endl;
				--count;
			}
			omp_unset_lock(&lock);
		}
	}
}

int main() {
	using namespace timer;
	int Buffer[32];
	Timer<HOST> TM;
	std::srand(time(NULL));

	omp_set_num_threads(2);

	TM.start();
	test_producer_consumer(Buffer);
	TM.stop();
	float time_seq = TM.duration();

	TM.start();
	test_producer_consumer_critical(Buffer);
	TM.stop();
	float time_critical = TM.duration();

	TM.start();
	test_producer_consumer_locks(Buffer);
	TM.stop();
	float time_locks = TM.duration();
	
	std::cout << "Sequential time: " << time_seq << "\n" << std::endl;
	std::cout << "Critical region time: " << time_critical << "\n" << std::endl;
	std::cout << "Locks region time: " << time_locks << "\n" << std::endl;
}
