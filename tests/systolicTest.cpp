//
// Created by alireza on 1/13/22.
//

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"

uint32_t add8in32(uint32_t memory, uint32_t systolicResult);

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
            uint32_t mult;
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < MAX_COL; j++) {
                    if (j == MAX_COL -1){
                        mult = systolicMM.streamInOut(mem2d(inputArray, M / W_DATA, i, j + base_col_idx));
                    }
                    else{
                        mult = systolicMM.inputQueue(j % MAX_COL, mem2d(inputArray, M / W_DATA, i, j + base_col_idx));
                    }

                    if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {    // check if the output is valid
                        mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize, colStart + outputIndex % colBlockSize) =
                                add8in32(mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize,
                                               colStart + outputIndex % colBlockSize), mult);
                        outputIndex++;
                    }
                }
            }
            for (int i = Nx * MAX_COL; i < MAX_COL * (Nx + 2 * KERNEL_DIM - 1) - 1; i++) {
                if ((i % MAX_COL) == MAX_COL -1){
                    mult = systolicMM.streamInOut(0);
                }
                else{
                    mult = systolicMM.inputQueue(i % MAX_COL, 0);
                }
                if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                    mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize, colStart + outputIndex % colBlockSize) =
                    add8in32(mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize,
                                         colStart + outputIndex % colBlockSize), mult);
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

uint32_t add8in32(uint32_t memory, uint32_t systolicResult){
    /*
     * This function separates every 32-bit input to four 8-bit integers and add them. Then, packing again and
     * make a 32-bit  unsigned int.
     */
    uint32_t result = 0;
    for (int i = 0; i < W_DATA; i++) {
        auto mem8 =  (int8_t) ((memory >> (8 * (W_DATA - i -1))) & 0xff);
        auto sysRes8 =  (int8_t)((systolicResult >> (8 * (W_DATA - i -1))) & 0xff);

        result |= (uint8_t) (mem8 + sysRes8) << (8 * (W_DATA - i - 1));
    }
    return result;
}

int main() {
    test();
    return 0;
}

