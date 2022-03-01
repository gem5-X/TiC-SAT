//
// Created by alireza on 3/1/22.
//

#include "matMulSystolic.h"

void MatMulSystolic::compute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                                std::size_t input_size_, std::size_t output_size_) {

    SystolicMatrixMultiplication systolicMM;
    for (int tileCol = 0; tileCol < output_size_ / KERNEL_DIM; tileCol++) {
        std::cout<<"Tile Column : "<<tileCol <<std::endl;
        for (int tileRow = 0; tileRow < input_size_ / KERNEL_DIM; tileRow++) {
            // Load the kernel with the corresponding weight
            int rowStart = tileRow * KERNEL_DIM;
            int colStart = tileCol * KERNEL_DIM / W_DATA;
            int rowBlockSize = KERNEL_DIM;
            int colBlockSize = KERNEL_DIM / W_DATA;
            for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                for (int j = colStart; j < colStart + colBlockSize; j++) {
                    uint32_t weight = mem2d(weights, output_size_ / W_DATA, i, j);
                    systolicMM.loadWeights(i - rowStart, j - colStart, weight);
                }
            }

            // Process the multiplication
            int base_col_idx = tileRow * MAX_COL;
            int outputIndex = 0;
            uint32_t mult;
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < MAX_COL; j++) {
                    if (j == MAX_COL -1){
                        mult = systolicMM.streamInOut(mem2d(input, input_size_ / W_DATA, i, j + base_col_idx));
                    }
                    else{
                        mult = systolicMM.inputQueue(j % MAX_COL, mem2d(input, input_size_ / W_DATA, i, j + base_col_idx));
                    }

                    if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {    // check if the output is valid
                        mem2d(output, output_size_ / W_DATA, outputIndex / colBlockSize, colStart + outputIndex % colBlockSize) =
                                add8in32(mem2d(output, output_size_ / W_DATA, outputIndex / colBlockSize,
                                               colStart + outputIndex % colBlockSize), mult);
                        outputIndex++;
                    }
                }
            }
            for (int i = seq_len * MAX_COL; i < MAX_COL * (seq_len + 2 * KERNEL_DIM - 1) - 1; i++) {
                if ((i % MAX_COL) == MAX_COL -1){
                    mult = systolicMM.streamInOut(0);
                }
                else{
                    mult = systolicMM.inputQueue(i % MAX_COL, 0);
                }
                if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                    mem2d(output, output_size_ / W_DATA, outputIndex / colBlockSize, colStart + outputIndex % colBlockSize) =
                            add8in32(mem2d(output, output_size_ / W_DATA, outputIndex / colBlockSize,
                                           colStart + outputIndex % colBlockSize), mult);
                    outputIndex++;
                }
            }
        }
    }
}

uint32_t MatMulSystolic::add8in32(uint32_t memory, uint32_t systolicResult) {
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
