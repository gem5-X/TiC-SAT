//
// Created by alireza on 3/1/22.
//

#include "matMulSystolic.h"

void MatMulSystolic::compute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                             std::size_t input_size_, std::size_t output_size_) {

    SystolicMatrixMultiplication systolicMM;
    int ROWS_IN_BLOCK = std::min(64, (int) (seq_len));;
    int rowMaxL1 = std::min(128, (int) (input_size_)) / KERNEL_DIM;
    int ratio = 128 / std::min(128, (int) (input_size_));
    int colMaxL1 = std::min(64 * ratio, (int) (output_size_)) / KERNEL_DIM;

    int ROWS_IN_L2 = std::min(512, (int) (seq_len)) / ROWS_IN_BLOCK;
    int rowMaxL2 = std::min(512, (int) (input_size_)) / KERNEL_DIM / rowMaxL1;
    int colMaxL2 = std::min(512, (int) (output_size_)) / KERNEL_DIM / colMaxL1;
    std::cout << rowMaxL2 << "\t\t" << colMaxL2 << std::endl;
    for (int l2In = 0; l2In < (seq_len / ROWS_IN_BLOCK / ROWS_IN_L2); l2In++) {
        for (int l2Row = 0; l2Row < (input_size_ / KERNEL_DIM) / rowMaxL2 / rowMaxL1; l2Row++) {
            for (int l2Col = 0; l2Col < (output_size_ / KERNEL_DIM) / colMaxL2 / colMaxL1; l2Col++) {
                for (int tileInL2 = 0; tileInL2 < (ROWS_IN_L2); tileInL2++) {
                    for (int tileRowL2 = 0; tileRowL2 < rowMaxL2; tileRowL2++) {
                        for (int tileColL2 = 0; tileColL2 < colMaxL2; tileColL2++) {
                            for (int tileRowL1 = 0; tileRowL1 < rowMaxL1; tileRowL1++) {
                                for (int tileColL1 = 0; tileColL1 < colMaxL1; tileColL1++) {
                                    int tileRow = tileRowL2 * rowMaxL1 + tileRowL1;
                                    int tileCol = tileColL2 * colMaxL1 + tileColL1;
                                    int seqBlockIdx = l2In * ROWS_IN_L2 + tileInL2;

                                    // Load the kernel with the corresponding weight
                                    int rowStart = (l2Row * rowMaxL2 * rowMaxL1 + tileRow) * KERNEL_DIM;
                                    int colStart = (l2Col * colMaxL2 * colMaxL1 + tileCol) * KERNEL_DIM / W_DATA;
                                    int rowBlockSize = KERNEL_DIM;
                                    int colBlockSize = KERNEL_DIM / W_DATA;
                                    uint32_t *wPtr = weights + rowStart * (output_size_ / W_DATA);
                                    for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                                        for (int j = colStart; j < colStart + colBlockSize; j++) {
                                            uint32_t weight = *(wPtr + j);
                                            systolicMM.loadWeights(i - rowStart, j - colStart, weight);
                                        }
                                        wPtr += output_size_ / W_DATA;
                                    }

                                    // Process the multiplication
                                    int base_col_idx = (l2Row * rowMaxL2 * rowMaxL1 + tileRow) * MAX_COL;
                                    int seqBlockLen = ROWS_IN_BLOCK;
                                    int outputIndex = 0;
                                    uint32_t *outPtr = output + seqBlockIdx * seqBlockLen * (output_size_ / W_DATA);
                                    uint32_t mult;
                                    const uint32_t *inPtr = input + base_col_idx + seqBlockIdx * seqBlockLen * (input_size_ / W_DATA);
                                    for (int i = 0; i < seqBlockLen; i++) {
                                        for (int j = 0; j < MAX_COL; j++) {
                                            if (j == MAX_COL - 1) {
                                                mult = systolicMM.streamInOut(*(inPtr + j));
                                            } else {
                                                mult = systolicMM.inputQueue(j % MAX_COL, *(inPtr + j));
                                            }

                                            if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {    // check if the output is valid
                                                add8in32(mem2d(outPtr, output_size_ / W_DATA, outputIndex / colBlockSize,
                                                              colStart + outputIndex % colBlockSize), mult);
                                                outputIndex++;
                                            }
                                        }
                                        inPtr += (input_size_ / W_DATA);
                                    }
                                    for (int i = seqBlockLen * MAX_COL; i < MAX_COL * (seqBlockLen + 2 * KERNEL_DIM - 1) - 1; i++) {
                                        if ((i % MAX_COL) == MAX_COL - 1) {
                                            mult = systolicMM.streamInOut(0);
                                        } else {
                                            mult = systolicMM.inputQueue(i % MAX_COL, 0);
                                        }
                                        if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                                            add8in32(mem2d(outPtr, output_size_ / W_DATA, outputIndex / colBlockSize,
                                                           colStart + outputIndex % colBlockSize), mult);
                                            outputIndex++;
                                        }
                                    }
                                }
                            }
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
