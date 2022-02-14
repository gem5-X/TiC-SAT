//
// Created by alireza on 1/13/22.
//

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"

void test() {
    uint32_t outputArray[Nx * Pw / W_DATA] = {0};

    SystolicMatrixMultiplication systolicMM;
    for (int tileCol = 0; tileCol < Pw / KERNEL_DIM; tileCol++) {
        std::cout<<"Tile Column : "<<tileCol <<std::endl;
        for (int tileRow = 0; tileRow < M / KERNEL_DIM; tileRow++) {
            // Load the kernel with the corresponding weight
            int rowStart = tileRow * KERNEL_DIM;
            int colStart = tileCol * KERNEL_DIM / W_DATA;
            int rowBlockSize = KERNEL_DIM;
            int colBlockSize = KERNEL_DIM / W_DATA;
            for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                for (int j = colStart; j < colStart + colBlockSize; j++) {
                    uint32_t weight = mem2d(weights, Pw / W_DATA, i, j);
                    systolicMM.loadWeights(i - rowStart, j - colStart, weight);
                }
            }

            // Process the multiplication
            int base_col_idx = tileRow * MAX_COL;
            int outputIndex = 0;
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < MAX_COL; j++) {
                    uint32_t mult = systolicMM.streamInOut(j % MAX_COL,
                                                           mem2d(inputArray, M / W_DATA, i, j + base_col_idx));
                    if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                        mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize,
                              colStart + outputIndex % colBlockSize) += mult;
                        outputIndex++;
                    }
                }
            }
            for (int i = Nx * MAX_COL; i < MAX_COL * (Nx + 2 * KERNEL_DIM - 1) - 1; i++) {
                uint32_t mult = systolicMM.streamInOut(i % MAX_COL, 0);
                if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                    mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize,
                          colStart + outputIndex % colBlockSize) += mult;
                    outputIndex++;
                }
            }
        }
    }

    // Print the output
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Pw / W_DATA; j++)
            std::cout << std::hex << (uint32_t) mem2d(outputArray, Pw / W_DATA, i, j) << "\t";
        std::cout << std::endl;
    }

}

int main() {
    test();
    return 0;
}

