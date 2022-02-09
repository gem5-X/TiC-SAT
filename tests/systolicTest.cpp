//
// Created by alireza on 1/13/22.
//

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"

void test() {
    uint32_t weights[M * MAX_COL] = {0x2000100, 0x1000000, 0x3000300, 0x10201, 0x2010303, 0x2010303, 0x10103, 0x3030301,
                                     0x30100, 0x102, 0x2000202, 0x30203, 0x1000002, 0x2000301, 0x102, 0x3030301,
                                     0x2010201, 0x20201, 0x20200, 0x3020200, 0x301, 0x3030303, 0x10300, 0x2010201,
                                     0x10302, 0x20101, 0x2030202, 0x2030201, 0x3020201, 0x3030003, 0x2020100, 0x1000300,
                                     0x2000000, 0x20302, 0x1000102, 0x2000300, 0x203, 0x3030102, 0x2010300, 0x10100,
                                     0x1010301, 0x303, 0x2020003, 0x3000302, 0x3020303, 0x10101, 0x3020000, 0x1030001};

    uint32_t inputArray[Nx* M/W_DATA] = {0x2000200, 0x2020001, 0x20302, 0x20202, 0x2010102, 0x1030002, 0x30103,
                                          0x3020103, 0x1020000, 0x3000002, 0x2020001, 0x3030100, 0x3020001,
                                          0x1000301, 0x1030301, 0x2030202, 0x3030303, 0x2020300, 0x3020002,
                                          0x1020102, 0x20301, 0x10301, 0x1010003, 0x2010003, 0x10103, 0x1030101,
                                          0x1000203, 0x3000201, 0x1030003, 0x2010301, 0x2010303, 0x0, 0x3020003,
                                          0x2010100, 0x3030301, 0x101, 0x1020002, 0x2020102, 0x30002, 0x3020100,
                                          0x3010000, 0x1030300, 0x3000300, 0x3000101, 0x1010202, 0x30003, 0x3010000,
                                          0x3020301, 0x2020102, 0x1020303, 0x2000202, 0x1020102, 0x2010203, 0x3000102,
                                          0x10002, 0x20300, 0x1030300, 0x20302, 0x3000100, 0x1000300};

    uint32_t outputArray[Nx * MAX_COL] = {0};


    SystolicMatrixMultiplication systolicMM;
    for (int tile = 0; tile < M / KERNEL_DIM; tile++) {
        // Load the kernel with the corresponding weights
        int base_id = tile * KERNEL_DIM * MAX_COL;
        for (int i = 0; i < KERNEL_DIM * MAX_COL; i++) {
            systolicMM.loadWeights(i / MAX_COL, i % MAX_COL, weights[base_id + i]);
        }

        // Process the multiplication
        int base_col_idx = tile * MAX_COL;
        int outputIndex = 0;
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < MAX_COL; j++) {
                uint32_t mult = systolicMM.streamInOut(j % MAX_COL,
                                                       mem2d(inputArray, M / W_DATA, i, j + base_col_idx));
                if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                    std::cout << std::hex << mult << ", " << i << ", " << j <<  std::endl;
                    outputArray[outputIndex++] += mult;
                }
            }
        }
        for (int i = Nx * MAX_COL; i < MAX_COL * (Nx + 2 * KERNEL_DIM - 1) - 1; i++) {
            uint32_t mult = systolicMM.streamInOut(i % MAX_COL, 0);
            if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                outputArray[outputIndex++] += mult;
                std::cout << std::hex << mult << ", " << i <<  std::endl;
            }
        }
    }

    // Print the output
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < MAX_COL; j++)
            std::cout << std::hex << (uint32_t) mem2d(outputArray, MAX_COL, i, j) << "\t";
        std::cout << std::endl;
    }

}

int main() {
    test();
    return 0;
}

