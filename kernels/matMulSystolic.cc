//
// Created by alireza on 3/1/22.
//

#include "matMulSystolic.h"

void MatMulSystolic::compute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                             std::size_t input_size_, std::size_t output_size_) {

    SystolicMatrixMultiplication systolicMM;
    int rowMaxL2 = std::min(8, (int) (input_size_ / KERNEL_DIM));
    int colMaxL2 = std::min(8, (int) (output_size_ / KERNEL_DIM));
    std::cout<< rowMaxL2 << "\t\t" << colMaxL2 <<std::endl;
    for (int l2Row=0; l2Row < (input_size_ / KERNEL_DIM) / rowMaxL2; l2Row++){
        for (int l2Col=0; l2Col < (output_size_ / KERNEL_DIM) / colMaxL2; l2Col++){
            for (int tileRow = 0; tileRow < rowMaxL2; tileRow++) {
                for (int tileCol = 0; tileCol < colMaxL2; tileCol++) {
                    //        std::cout << "Tile Column : " << tileCol << std::endl;

                    // Load the kernel with the corresponding weight
                    int rowStart = (l2Row * rowMaxL2 + tileRow) * KERNEL_DIM;
                    int colStart = (l2Col * colMaxL2 + tileCol) * KERNEL_DIM / W_DATA;
                    int rowBlockSize = KERNEL_DIM;
                    int colBlockSize = KERNEL_DIM / W_DATA;
                    uint32_t* wPtr = weights + rowStart * (output_size_/W_DATA);
                    for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                        for (int j = colStart; j < colStart + colBlockSize; j++) {
                            uint32_t weight = * (wPtr + j);
                            systolicMM.loadWeights(i - rowStart, j - colStart, weight);
                        }
                        wPtr += output_size_ / W_DATA;
                    }

                    // Process the multiplication
                    int base_col_idx = (l2Row * rowMaxL2 + tileRow) * MAX_COL;
                    int outputIndex = 0;
                    uint32_t mult;
                    const uint32_t * inPtr = input + base_col_idx;
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < MAX_COL; j++) {
                            if (j == MAX_COL - 1) {
                                mult = systolicMM.streamInOut(*(inPtr + j));
                            } else {
                                mult = systolicMM.inputQueue(j % MAX_COL,*(inPtr + j));
                            }

                            if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {    // check if the output is valid
                                add8in32(mem2d(output, output_size_ / W_DATA, outputIndex / colBlockSize,
                                               colStart + outputIndex % colBlockSize), mult);
                                outputIndex++;
                            }
                        }
                        inPtr += (input_size_ / W_DATA);
                    }
                    for (int i = seq_len * MAX_COL; i < MAX_COL * (seq_len + 2 * KERNEL_DIM - 1) - 1; i++) {
                        if ((i % MAX_COL) == MAX_COL - 1) {
                            mult = systolicMM.streamInOut(0);
                        } else {
                            mult = systolicMM.inputQueue(i % MAX_COL, 0);
                        }
                        if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                            add8in32(mem2d(output, output_size_ / W_DATA, outputIndex / colBlockSize,
                                           colStart + outputIndex % colBlockSize), mult);
                            outputIndex++;
                        }
                    }
                }
            }
        }
    }
}

void MatMulSystolic::add8in32(uint32_t &memory, uint32_t &systolicResult) {
    /*
     * This function separates every 32-bit input to four 8-bit integers and add them. Then, packing again and
     * make a 32-bit  unsigned int.
     */
    auto *mem_ptr = (int8_t *) (&memory);
    auto *sys_ptr = (int8_t *) (&systolicResult);
    for (int i = 0; i < W_DATA; i++) {
        *mem_ptr = (int8_t) (*mem_ptr + *sys_ptr);
        mem_ptr++;
        sys_ptr++;
    }
}
