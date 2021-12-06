#include <iostream>
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    int N = 268435456;
    double factorial = 1;
    
    /* Sequential implementation of factorial:*/
    
    Timer<HOST> TM;
    TM.start();
    
    for (int i = 1; i <= N; ++i)
        factorial *= i;
    
    TM.stop();
    TM.print("Sequential Factorial");
    std::cout << factorial << std::endl;
    
    //--------------------------------------------------------------------------
    
    /* Parallel implementation of Factorial: */
    
    double parallelResult = 1;
    int i;
    TM.start();
    
    #pragma omp parallel for shared(N) private(i) reduction(*:parallelResult)
    for (int i = 1; i <= N; ++i){
        parallelResult *= i;
    }

    TM.stop();
    TM.print("Parallel Factorial");
    std::cout << parallelResult << std::endl;
}
