#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
#include "Timer.hpp"

// ---------- DO NOT MODIFY ----------------------------------------------------
int partition(int array[], const int start, const int end) {
    int p = start;
    int pivot_element = array[end];
    for(int i = start; i < end; i++) {
        if(array[i] < pivot_element) {
            std::swap(array[i], array[p]);
            ++p;
        }
    }
    std::swap(array[p], array[end]);
    return p;
}
//------------------------------------------------------------------------------

void quick_sort(int array[], const int start, const int end) {
    if(start < end) {
        int pivot = partition(array, start, end);
        quick_sort(array, start, pivot - 1);
        quick_sort(array, pivot + 1, end);
    }
}

void quick_sort_parallel(int array[], const int start, const int end) {
    if(start < end) {
        int pivot = partition(array, start, end);
        #pragma omp parallel sections
        {   
            #pragma omp section
            quick_sort(array, start, pivot - 1);
            #pragma omp section
            quick_sort(array, pivot + 1, end);
        }
    }
}

template<typename T>
void print_array(T* array, int size, const char* str) {
    std::cout << str << "\n";
    for (int i = 0; i < size; ++i)
        std::cout << array[i] << ' ';
    std::cout << "\n" << std::endl;
}


int main() {
    const int N = 1000000;
    using namespace timer;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    int* input = new int[N];
    for (int i = 0; i < N; ++i)
        input[i] = distribution(generator);

    int* input2 = new int[N];
    std::copy(input, input + N, input2);

    Timer<HOST> TM;
    TM.start();
    quick_sort(input, 0, N - 1);
    TM.stop();
    TM.print("Sequential quick sort");

    TM.start();
    quick_sort_parallel(input2, 0, N - 1);
    TM.stop();
    TM.print("Parallel quick sort");

    for (int i = 0; i < N; ++i) {
        if (input[i] != input2[i]) {
            std::cout << "Solution non correct at " << i
                      << input[i] << " != " << input2[i]
                      << std::endl;
            return 0; 
        }
    }
    std::cout << "<> Correct" << std::endl;
}
