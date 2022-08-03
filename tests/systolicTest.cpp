//
// Created by alireza on 1/13/22.
//

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"
#include "../gem5x_codes/smm_gem.h"
#include <ctime>

int main() {
    uint32_t outputArray[Nx * Pw / W_DATA] = {0};
    clock_t tStart, tEnd;
    tStart = clock();
    simdCompute(Nx, inputArray, outputArray, weights, M, Pw);
    tEnd = clock();
    std::cout << "Time: " << (double)(tEnd - tStart) / CLOCKS_PER_SEC << std::endl;

    // Print the output
    for (int i = Nx-1; i < Nx; i++) {
        for (int j = 0; j < Pw / W_DATA; j++)
            std::cout << std::hex << (uint32_t) mem2d(outputArray, Pw / W_DATA, i, j) << "\t";
        std::cout << std::endl;
    }

    return 0;
}

