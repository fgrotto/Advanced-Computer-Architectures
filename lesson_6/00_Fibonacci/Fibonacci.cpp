#include <iostream>
#include <omp.h>
#include "Timer.hpp"

long long int fibonacci(long long int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left  = fibonacci(value - 1, level + 1);
    fib_right = fibonacci(value - 2, level + 1);

    return fib_left + fib_right;
}

long long int fibonacci_tasks_parallel(long long int value, int level) {
    if (value <= 1)
        return 1;
        
    long long int fib_left, fib_right;
    #pragma omp task shared(fib_left)
    fib_left  = fibonacci(value - 1, level + 1);
    #pragma omp task shared(fib_right)
    fib_right = fibonacci(value - 2, level + 1);

    #pragma omp taskwait
    return fib_left + fib_right;
}

long long int fibonacci_sections_parallel(long long int value, int level) {
    if (value <= 1)
        return 1;
        
    long long int fib_left, fib_right;
    #pragma omp parallel sections
    {   
        #pragma omp section
        fib_left  = fibonacci(value - 1, level + 1);
        #pragma omp section
        fib_right = fibonacci(value - 2, level + 1);
    }

    return fib_left + fib_right;
}

long long int fibonacci_iterative(long long int value, int level) {
   long long int x = 0, y = 1, z = 0;
   for (long long int i = 0; i < value; i++) {
      z = x + y;
      x = y;
      y = z;
   }
   return z;
}


int main() {
    //  ------------------------- TEST FIBONACCI ----------------------
    using namespace timer;
    omp_set_dynamic(0);
    int value = 47;

    Timer<HOST> TM;
    TM.start();
    long long int result = fibonacci(value, 1);
    TM.stop();
    TM.print("Sequential Fibonacci");

    std::cout << "\nresult: " << result << "\n" << std::endl;

    TM.start();
    long long int result2 = fibonacci_tasks_parallel(value, 1);
    TM.stop();
    TM.print("Parallel Fibonacci (tasks)");

    std::cout << "\nresult: " << result2 << "\n" << std::endl;

    TM.start();
    long long int result3 = fibonacci_sections_parallel(value, 1);
    TM.stop();
    TM.print("Parallel Fibonacci (sections)");

    std::cout << "\nresult: " << result3 << "\n" << std::endl;

    TM.start();
    long long int result4 = fibonacci_iterative(value, 1);
    TM.stop();
    TM.print("Iterative Fibonacci");

    std::cout << "\nresult: " << result4 << "\n" << std::endl;
}
