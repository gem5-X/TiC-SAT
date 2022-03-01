//
// Created by alireza on 1/13/22.
//

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"
#include "../kernels/matMulSystolic.h"

using namespace lh;

int main() {
    uint32_t outputArray[Nx * Pw / W_DATA] = {0};
    MatMulSystolic matMulSystolic;
    matMulSystolic.compute(Nx, inputArray, outputArray, weights, M, Pw);

    // Print the output
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Pw / W_DATA; j++)
            std::cout << std::hex << (uint32_t) mem2d(outputArray, Pw / W_DATA, i, j) << "\t";
        std::cout << std::endl;
    }

    return 0;
}

