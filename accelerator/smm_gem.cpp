//
// Created by alireza on 3/3/22.
//

#include "iostream"
#include "smm_gem.h"
#include <cmath>

#ifdef SIMD
#include <arm_neon.h>
#endif

#define W_DATA 4
#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/W_DATA)
#define mem2d(data, data_len, row, col)   data[((row)*(data_len))+(col)]


void add8in32(uint32_t &memory, uint32_t &systolicResult);

//#define DEVELOP

#ifndef DEVELOP

/* CM Core Process (MVM)
* Instruction format: |____Opcode___|__rm__|_X|__ra__|__rn__|__rd__|
* Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
* Binary layout:      |0000_0001_000|0_1000|_0|001_11|01_001|0_1010|
* Hex layout:         |__0____1____0|____8_|__|_1____|D____2|____A_|
* gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
*
* Queueing arguments:
* -- rd = Parameter value.
* -- rm = Parameter x index.
* -- ra = Unused.
* -- rn = Parameter y index.
*/

uint64_t smmStream(uint64_t rn) {
    uint64_t res;

    __asm__ volatile(
    "MOV X7, %[input_k];"
    ".long 0x01081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_k] "r"(rn)
    : "x7", "x10"
    );

    return res;
}

/* CM Core Queue (MVM)
* Instruction format: |____Opcode___|__rm__|_X|__ra__|__rn__|__rd__|
* Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
* Binary layout:      |0010_0001_000|0_1000|_0|001_11|01_001|0_1010|
* Hex layout:         |__2____1____0|____8_|__|_1____|D____2|____A_|
* gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
*
* Queueing arguments:
* -- rd = Parameter value.
* -- rm = Parameter x index.
* -- ra = Unused.
* -- rn = Parameter y index.
*/
uint64_t smmQueue(uint64_t rm, uint64_t rn) {
    uint64_t res;

    __asm__ volatile(
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    ".long 0x21089D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_j] "r"(rm), [input_k] "r"(rn)
    : "x7", "x9", "x10"
    );

    return res;
}

/* CM Core Parameter Write
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0100_0001_000|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__4____1____0|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Success code.
 * -- rm = Parameter x index.
 * -- ra = Parameter value.
 * -- rn = Parameter y index.
 */
uint64_t smmParamWrite(uint64_t rm, uint64_t rn, uint64_t ra) {
    uint64_t res;

    __asm__ volatile(
    "MOV X8, %[input_i];"
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    ".long 0x41081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_i] "r"(ra), [input_j] "r"(rm), [input_k] "r"(rn)
    : "x7", "x8", "x9", "x10"
    );

    return res;

}

#else

#include "systolic_m2m.hh"

SystolicMatrixMultiplication smm = SystolicMatrixMultiplication();

void smmParamWrite(int rm, int rn, uint32_t ra) {
    smm.loadWeights(rm, rn, ra);
}

uint32_t smmQueue(int rm, uint32_t ra) {
    return smm.inputQueue(rm, ra);
}

uint32_t smmStream(uint32_t rn) {
    return smm.streamInOut(rn);
}

#endif


void smmCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights, uint32_t *flag,
                std::size_t input_size_, std::size_t output_size_, bool sparse) {

    int ROWS_IN_BLOCK = std::min(128, (int) (seq_len));
    int rowMaxL1 = std::min(64, (int) (input_size_)) / KERNEL_DIM;
    int ratio = 64 / std::min(64, (int) (input_size_));
    int colMaxL1 = std::min(32 * ratio, (int) (output_size_)) / KERNEL_DIM;

    int ROWS_IN_L2 = std::min(512 / ROWS_IN_BLOCK, (int) ceil((float) (seq_len) / (float) ROWS_IN_BLOCK));
    int rowMaxL2 = std::min(256, (int) (input_size_)) / KERNEL_DIM / rowMaxL1;
    int colMaxL2 = std::min(256, (int) (output_size_)) / KERNEL_DIM / colMaxL1;

    auto *flag_ptr = (bool *) (flag);
    int counter = 0;
    int total_counter =0;

    for (int l2In = 0; l2In < (int) ceil((float) seq_len / (float) ROWS_IN_BLOCK / (float) ROWS_IN_L2); l2In++) {
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
//                                    if (sparse) {
//                                        if (*(flag_ptr + rowStart * output_size_ / (KERNEL_DIM * KERNEL_DIM) +
//                                              colStart / MAX_COL)) {
//                                            continue;
//                                        }
//                                    }

                                    bool non_zero_tile = false;
                                    for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                                        for (int j = colStart; j < colStart + colBlockSize; j++) {
                                            uint32_t weight = *(wPtr + j);
                                            smmParamWrite(i - rowStart, j - colStart, weight);
                                            non_zero_tile += (weight != 0x0);
                                        }
                                        wPtr += output_size_ / W_DATA;
                                    }
                                    total_counter ++;
                                    if (!non_zero_tile && sparse) {
                                        counter++;
                                        continue;
                                    }

                                    // Process the multiplication
                                    int base_col_idx = (l2Row * rowMaxL2 * rowMaxL1 + tileRow) * MAX_COL;
                                    int seqBlockLen = std::min(ROWS_IN_BLOCK,
                                                               (int) (seq_len - seqBlockIdx * ROWS_IN_BLOCK));
                                    int outputIndex = 0;
                                    uint32_t *outPtr = output + seqBlockIdx * ROWS_IN_BLOCK * (output_size_ / W_DATA);
                                    uint32_t mult;
                                    const uint32_t *inPtr =
                                            input + base_col_idx + seqBlockIdx * ROWS_IN_BLOCK * (input_size_ / W_DATA);
                                    for (int i = 0; i < seqBlockLen; i++) {
                                        for (int j = 0; j < MAX_COL; j++) {
                                            if (j == MAX_COL - 1) {
                                                mult = smmStream(*(inPtr + j));
                                            } else {
                                                mult = smmQueue(j % MAX_COL, *(inPtr + j));
                                            }

                                            if ((i * MAX_COL + j) >=
                                                (MAX_COL * (2 * KERNEL_DIM - 1) -
                                                 1)) {    // check if the output is valid
                                                add8in32(
                                                        mem2d(outPtr, output_size_ / W_DATA, outputIndex / colBlockSize,
                                                              colStart + outputIndex % colBlockSize), mult);
                                                outputIndex++;
                                            }
                                        }
                                        inPtr += (input_size_ / W_DATA);
                                    }
                                    for (int i = seqBlockLen * MAX_COL;
                                         i < MAX_COL * (seqBlockLen + 2 * KERNEL_DIM - 1) - 1; i++) {
                                        if ((i % MAX_COL) == MAX_COL - 1) {
                                            mult = smmStream(0);
                                        } else {
                                            mult = smmQueue(i % MAX_COL, 0);
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
#ifdef DEVELOP
    std::cout << "Sparse : " << counter << " Out of : " << total_counter
    << " So " << (float)counter / (float) total_counter << "%" << std::endl;
    //    print_arr(output, seq_len, output_size_);
    //    getchar();
#endif
}

unsigned int *memory_rearrangement(uint32_t *weights, std::size_t output_size_, std::size_t input_size_) {
    auto rearranged = new uint32_t[output_size_ * input_size_ / W_DATA]();
    int counter = 0;
    for (int l2Row = 0; l2Row < input_size_ / KERNEL_DIM; l2Row++) {
        for (int l2Col = 0; l2Col < output_size_ / KERNEL_DIM; l2Col++) {
            // Load the kernel with the corresponding weight
            int rowStart = l2Row * KERNEL_DIM;
            int colStart = l2Col * KERNEL_DIM / W_DATA;
            int rowBlockSize = KERNEL_DIM;
            int colBlockSize = KERNEL_DIM / W_DATA;
            uint32_t *wPtr = weights + rowStart * (output_size_ / W_DATA);
            bool non_zero_tile = false;
            for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                for (int j = colStart; j < colStart + colBlockSize; j++) {
                    uint32_t weight = *(wPtr + j);
                    rearranged[counter++] = weight;
                    non_zero_tile += (weight != 0x0);
                }
                wPtr += output_size_ / W_DATA;
            }
        }
    }
    return rearranged;
}

unsigned int *input_rearrangement(const uint32_t *inputs, std::size_t seq_len, std::size_t input_size_) {
    auto rearranged = new uint32_t[seq_len * input_size_ / W_DATA]();
    int counter = 0;
    for (int l2Row = 0; l2Row < input_size_ / KERNEL_DIM; l2Row++) {
        int base_col_idx = l2Row * MAX_COL;
        const uint32_t *inPtr = inputs + base_col_idx;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < MAX_COL; j++) {
                rearranged[counter++] = *(inPtr + j);
            }
            inPtr += (input_size_ / W_DATA);
        }
    }
    return rearranged;
}


void smmComputeRearranged(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                          uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse) {
    int counter = 0;
    int total_counter =0;

    for (int l2Col = 0; l2Col < output_size_ / KERNEL_DIM; l2Col++) {
        for (int l2Row = 0; l2Row < input_size_ / KERNEL_DIM; l2Row++) {
            // Load the kernel with the corresponding weight
            int rowBlockSize = KERNEL_DIM;
            int colBlockSize = KERNEL_DIM / W_DATA;

#ifdef LOAD_SKIP
            bool non_zero_tile = false;
            for (int i = 0; i < rowBlockSize * colBlockSize; i++) {
                uint32_t weight = *(weights++);
                smmParamWrite(i / colBlockSize, i % colBlockSize, weight);
                non_zero_tile += (weight != 0x0);
            }

            total_counter++;
            if (!non_zero_tile && sparse) {
                counter++;
                continue;
            }
#else
            if (sparse) {
                if (counter == 32) {
                    counter = 0;
                    flag++;
                }
                if (*flag & (0x00000001 << counter++)) {
                    continue;
                }
            }

            for (int i = 0; i < rowBlockSize * colBlockSize; i++) {
                uint32_t weight = *(weights++);
                smmParamWrite(i / colBlockSize, i % colBlockSize, weight);
            }
#endif

            // Process the multiplication
            int base_col_idx = l2Row * MAX_COL * seq_len;
            uint32_t *outPtr = output + l2Col * MAX_COL * seq_len;
            uint32_t mult;
            const uint32_t *inPtr = input + base_col_idx;
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < MAX_COL; j++) {
                    uint32_t content = *(inPtr);
                    if (j == MAX_COL - 1) {
                        mult = smmStream(*(inPtr++));
                    } else {
                        mult = smmQueue(j % MAX_COL, *(inPtr++));
                    }

                    if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                        // check if the output is valid
                        add8in32(*(outPtr++), mult);
                    }
                }
            }
            for (int i = seq_len * MAX_COL;
                 i < MAX_COL * (seq_len + 2 * KERNEL_DIM - 1) - 1; i++) {
                if ((i % MAX_COL) == MAX_COL - 1) {
                    mult = smmStream(0);
                } else {
                    mult = smmQueue(i % MAX_COL, 0);
                }
                if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                    add8in32(*(outPtr++), mult);
                }
            }
        }
    }

#ifdef DEVELOP
    std::cout << "Sparse : " << counter << " Out of : " << total_counter
    << " So " << (float)counter / (float) total_counter << "%" << std::endl;

    //    print_arr(output, output_size_ / KERNEL_DIM, seq_len * KERNEL_DIM);
    //    getchar();
#endif
}

void add8in32(uint32_t &memory, uint32_t &systolicResult) {
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

void print_arr(uint32_t *array, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p / W_DATA; j++)
            std::cout << std::hex << (uint32_t) mem2d(array, p / W_DATA, i, j) << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int j = 0; j < 4; j++)
        std::cout << std::hex << (uint32_t) mem2d(array, p / W_DATA, 0, j) << "\t";
    std::cout << std::endl;
    for (int j = 0; j < 4; j++)
        std::cout << std::hex << (uint32_t) mem2d(array, p / W_DATA, n - 1, j) << "\t";
    std::cout << std::endl;

    for (int j = p / W_DATA - 2; j < p / W_DATA; j++)
        std::cout << std::hex << (uint32_t) mem2d(array, p / W_DATA, n - 1, j) << "\t";
    std::cout << std::endl;

}


void smmComputeEigen(std::size_t seq_len, const int8_t *input, int8_t *output, int8_t *weights,
                     std::size_t input_size_, std::size_t output_size_) {

    uint32_t tmp = 0;

    for (int tileRow = 0; tileRow < input_size_ / KERNEL_DIM; tileRow++) {
        for (int tileCol = 0; tileCol < output_size_ / KERNEL_DIM; tileCol++) {

            int rowBlockSize = KERNEL_DIM;
            int colBlockSize = KERNEL_DIM;
            int8_t *wPtr = weights + (tileRow * output_size_ + tileCol ) * KERNEL_DIM;

            for (int i = 0; i < rowBlockSize; i++) {
                for (int j = 0; j < colBlockSize; j++) {
                    tmp |= (*(wPtr + j) & 0xff) << (8 * (j % 4));
                    if (j % 4 == 3) {
                        smmParamWrite(i, j / W_DATA, tmp);
                        tmp = 0;
                    }
                }
                wPtr += output_size_;
            }

            tmp = 0;

            // Process the multiplication
            int outputIndex = 0;
            int8_t *outPtr = output + tileCol * KERNEL_DIM;
            uint32_t mult;
            const int8_t *inPtr = input + tileRow * KERNEL_DIM;
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < KERNEL_DIM; j++) {
                    tmp |= (*(inPtr + j) & 0xff) << (8 * (j % 4));
                    if (j % 4 == 3) {
                        if (j == KERNEL_DIM - 1) {
                            mult = smmStream(tmp);
                        } else {
                            mult = smmQueue(j / W_DATA, tmp);
                        }
                        tmp = 0;

                        if ((i * KERNEL_DIM + j) >=
                        (KERNEL_DIM * (2 * KERNEL_DIM - 1) - W_DATA)) {
                            // check if the output is valid
                            for (int k = 0; k < 4; k++) {
                                mem2d(outPtr, output_size_, outputIndex / colBlockSize,
                                      outputIndex % colBlockSize) += (
                                              (mult >> (8 * (k % 4))) & 0xFF);
                                outputIndex++;
                            }
                        }
                    }
                }
                inPtr += (input_size_);
            }
            for (int i = seq_len * MAX_COL;
                 i < MAX_COL * (seq_len + 2 * KERNEL_DIM - 1) - 1; i++) {
                if ((i % MAX_COL) == MAX_COL - 1) {
                    mult = smmStream(0);
                } else {
                    mult = smmQueue(i % MAX_COL, 0);
                }
                if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                    for (int k = 0; k < 4; k++) {
                        mem2d(outPtr, output_size_, outputIndex / colBlockSize,
                              outputIndex % colBlockSize) += (
                                (mult >> (8 * (k % 4))) & 0xFF);
                        outputIndex++;
                    }
                }
            }
        }
    }

#ifdef DEVELOP
    //    print_arr(output, seq_len, output_size_);
    //    getchar();
#endif
}

void conventionalCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_) {
    for (int length = 0; length < seq_len; length++) {
        for (int out_idx = 0; out_idx < (output_size_ / W_DATA); out_idx++) {
            // std::cout<< "out : " << out_idx << std::endl;
            auto *weight_ptr = (int8_t *) (weight + out_idx);
            auto *output_ptr = (int8_t *) (output + (length * output_size_ / W_DATA) + out_idx);
            for (int w = 0; w < W_DATA; w++) {
                auto *input_ptr = (int8_t *) (input + (length * input_size_ / W_DATA));
                int sum = 0;
                for (int i = 0; i < input_size_; i++) {
                    sum += *(weight_ptr + (i + 3 - 2 * (i % W_DATA)) * output_size_ + w) * (*(input_ptr));
                    input_ptr++;
                }
                *(output_ptr + w) = (int8_t) sum;
            }
        }
    }
}

void tiledCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weight,
                  std::size_t input_size_, std::size_t output_size_) {
    int ROWS_IN_BLOCK = std::min(128, (int) (seq_len));
    int COLS_IN_BLOCK = std::min(32, (int) (input_size_));
    int ratio = 32 / COLS_IN_BLOCK;
    int W_COL_BLOCKS = std::min(32 * ratio, (int) (output_size_));

    int ROWS_IN_L2 = std::min(512 / ROWS_IN_BLOCK, (int) (seq_len / ROWS_IN_BLOCK));
    int COLS_IN_L2 = std::min(256 / COLS_IN_BLOCK, (int) (input_size_ / COLS_IN_BLOCK));
    int W_COL_IN_L2 = std::min(256 / W_COL_BLOCKS, (int) (output_size_ / W_COL_BLOCKS));

    for (int blk_row_idx = 0; blk_row_idx < (seq_len / ROWS_IN_BLOCK / ROWS_IN_L2); blk_row_idx++) {
        for (int blk_col_idx = 0; blk_col_idx < (input_size_ / COLS_IN_BLOCK / COLS_IN_L2); blk_col_idx++) {
            for (int w_blk_col_idx = 0; w_blk_col_idx < (output_size_ / W_COL_BLOCKS / W_COL_IN_L2); w_blk_col_idx++) {
                for (int l2_row_idx = 0; l2_row_idx < ROWS_IN_L2; l2_row_idx++) {
                    for (int l2_col_idx = 0; l2_col_idx < COLS_IN_L2; l2_col_idx++) {
                        for (int l2_w_idx = 0; l2_w_idx < W_COL_IN_L2; l2_w_idx++) {
                            for (int i = 0; i < ROWS_IN_BLOCK; i++) {
                                auto *input_ptr = (int8_t *) (input +
                                                              (((blk_row_idx * ROWS_IN_L2 + l2_row_idx) *
                                                                ROWS_IN_BLOCK + i) * input_size_ / W_DATA) +
                                                              // index of the input row
                                                              (blk_col_idx * COLS_IN_L2 + l2_col_idx) * COLS_IN_BLOCK /
                                                              W_DATA);   // block index
                                auto *output_ptr = (int8_t *) (output +
                                                               (((blk_row_idx * ROWS_IN_L2 + l2_row_idx) *
                                                                 ROWS_IN_BLOCK + i) * output_size_ / W_DATA) +
                                                               (w_blk_col_idx * W_COL_IN_L2 + l2_w_idx) * W_COL_BLOCKS /
                                                               W_DATA);
                                auto *weight_ptr = (int8_t *) (weight +
                                                               (blk_col_idx * COLS_IN_L2 + l2_col_idx) * COLS_IN_BLOCK *
                                                               output_size_ / W_DATA +
                                                               (w_blk_col_idx * W_COL_IN_L2 + l2_w_idx) * W_COL_BLOCKS /
                                                               W_DATA);
                                for (int j = 0; j < W_COL_BLOCKS; j++) {
                                    int sum = 0;
                                    for (int k = 0; k < COLS_IN_BLOCK; k++) {
                                        sum += *(input_ptr + k) *
                                               *(weight_ptr + (k + 3 - 2 * (k % W_DATA)) * output_size_ + j);
                                        // a bias is added because of the endianness
                                    }
                                    *(output_ptr + j) = (int8_t) ((*(output_ptr + j)) + sum);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void tiledL1Compute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weight,
                    std::size_t input_size_, std::size_t output_size_) {
    int ROWS_IN_BLOCK = std::min(128, (int) (seq_len));
    int COLS_IN_BLOCK = std::min(32, (int) (input_size_));
    int ratio = 32 / COLS_IN_BLOCK;
    int W_COL_BLOCKS = std::min(32 * ratio, (int) (output_size_));

    int ROWS_IN_L2 = (int) (seq_len / ROWS_IN_BLOCK);
    int COLS_IN_L2 = (int) (input_size_ / COLS_IN_BLOCK);
    int W_COL_IN_L2 = (int) (output_size_ / W_COL_BLOCKS);

    for (int l2_row_idx = 0; l2_row_idx < ROWS_IN_L2; l2_row_idx++) {
        for (int l2_col_idx = 0; l2_col_idx < COLS_IN_L2; l2_col_idx++) {
            for (int l2_w_idx = 0; l2_w_idx < W_COL_IN_L2; l2_w_idx++) {
                for (int i = 0; i < ROWS_IN_BLOCK; i++) {
                    auto *input_ptr = (int8_t *) (input +
                                                  ((l2_row_idx * ROWS_IN_BLOCK + i) * input_size_ / W_DATA) +
                                                  // index of the input row
                                                  (l2_col_idx) * COLS_IN_BLOCK /
                                                  W_DATA);   // block index
                    auto *output_ptr = (int8_t *) (output +
                                                   (((l2_row_idx) *
                                                     ROWS_IN_BLOCK + i) * output_size_ / W_DATA) +
                                                   (l2_w_idx) * W_COL_BLOCKS /
                                                   W_DATA);
                    auto *weight_ptr = (int8_t *) (weight +
                                                   (l2_col_idx) * COLS_IN_BLOCK *
                                                   output_size_ / W_DATA +
                                                   (l2_w_idx) * W_COL_BLOCKS /
                                                   W_DATA);
                    for (int j = 0; j < W_COL_BLOCKS; j++) {
                        int sum = 0;
                        for (int k = 0; k < COLS_IN_BLOCK; k++) {
                            sum += *(input_ptr + k) *
                                   *(weight_ptr + (k + 3 - 2 * (k % W_DATA)) * output_size_ + j);
                            // a bias is added because of the endianness
                        }
                        *(output_ptr + j) = (int8_t) ((*(output_ptr + j)) + sum);
                    }
                }
            }
        }
    }
}




#ifdef SIMD

void print_int8 (int8x16_t data, char* name) {
    int i;
    static int8_t p[16];

    vst1q_s8 (p, data);

    printf ("%s = ", name);
    for (i = 0; i < 16; i++) {
        printf ("%02d ", p[i]);
    }
    printf ("\n");
}


void simdCompute(size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t * weight,
                 uint32_t *flag, size_t input_size_, size_t output_size_, bool sparse) {

    int ROWS_IN_BLOCK = 16;
    int COLS_IN_BLOCK = 16;
    int W_COL_BLOCKS = 16;

    int ROWS_IN_L2 = (int) (seq_len / ROWS_IN_BLOCK);
    int COLS_IN_L2 = (int) (input_size_ / COLS_IN_BLOCK);
    int W_COL_IN_L2 = (int) (output_size_ / W_COL_BLOCKS);

    int8x16_t A0;
    int8x16_t A1;
    int8x16_t A2;
    int8x16_t A3;
    int8x16_t A4;
    int8x16_t A5;
    int8x16_t A6;
    int8x16_t A7;
    int8x16_t A8;
    int8x16_t A9;
    int8x16_t A10;
    int8x16_t A11;
    int8x16_t A12;
    int8x16_t A13;
    int8x16_t A14;
    int8x16_t A15;

    int8x16_t B0;
    int8x16_t B1;
    int8x16_t B2;
    int8x16_t B3;
    int8x16_t B4;
    int8x16_t B5;
    int8x16_t B6;
    int8x16_t B7;
    int8x16_t B8;
    int8x16_t B9;
    int8x16_t B10;
    int8x16_t B11;
    int8x16_t B12;
    int8x16_t B13;
    int8x16_t B14;
    int8x16_t B15;

    int8x16_t C0;
    int8x16_t C1;
    int8x16_t C2;
    int8x16_t C3;
    int8x16_t C4;
    int8x16_t C5;
    int8x16_t C6;
    int8x16_t C7;
    int8x16_t C8;
    int8x16_t C9;
    int8x16_t C10;
    int8x16_t C11;
    int8x16_t C12;
    int8x16_t C13;
    int8x16_t C14;
    int8x16_t C15;


    for (int l2_row_idx = 0; l2_row_idx < ROWS_IN_L2; l2_row_idx++) {
        for (int l2_col_idx = 0; l2_col_idx < COLS_IN_L2; l2_col_idx++) {
            C0=vmovq_n_s8(0);
            C1=vmovq_n_s8(0);
            C2=vmovq_n_s8(0);
            C3=vmovq_n_s8(0);
            C4=vmovq_n_s8(0);
            C5=vmovq_n_s8(0);
            C6=vmovq_n_s8(0);
            C7=vmovq_n_s8(0);
            C8=vmovq_n_s8(0);
            C9=vmovq_n_s8(0);
            C10=vmovq_n_s8(0);
            C11=vmovq_n_s8(0);
            C12=vmovq_n_s8(0);
            C13=vmovq_n_s8(0);
            C14=vmovq_n_s8(0);
            C15=vmovq_n_s8(0);

            int8_t* output8_t = (int8_t * ) output;
            int C_idx = ((l2_row_idx) * ROWS_IN_BLOCK) * output_size_;

            for (int l2_w_idx = 0; l2_w_idx < W_COL_IN_L2; l2_w_idx++) {
                bool print_bool = (l2_row_idx == 0 && l2_col_idx == 0 && l2_w_idx == 0);
                int A_idx = ((l2_row_idx * ROWS_IN_BLOCK) * input_size_) +  (l2_col_idx) * COLS_IN_BLOCK ;
                int8_t* input8_t = (int8_t * ) input;

                A0 = vld1q_s8(input8_t + A_idx);
                A1 = vld1q_s8(input8_t + A_idx + input_size_);
                A2 = vld1q_s8(input8_t + A_idx + 2*input_size_);
                A3 = vld1q_s8(input8_t + A_idx + 3*input_size_);
                A4 = vld1q_s8(input8_t + A_idx + 4*input_size_);
                A5 = vld1q_s8(input8_t + A_idx + 5*input_size_);
                A6 = vld1q_s8(input8_t + A_idx + 6*input_size_);
                A7 = vld1q_s8(input8_t + A_idx + 7*input_size_);
                A8 = vld1q_s8(input8_t + A_idx + 8*input_size_);
                A9 = vld1q_s8(input8_t + A_idx + 9*input_size_);
                A10 = vld1q_s8(input8_t + A_idx + 10*input_size_);
                A11 = vld1q_s8(input8_t + A_idx + 11*input_size_);
                A12 = vld1q_s8(input8_t + A_idx + 12*input_size_);
                A13 = vld1q_s8(input8_t + A_idx + 13*input_size_);
                A14 = vld1q_s8(input8_t + A_idx + 14*input_size_);
                A15 = vld1q_s8(input8_t + A_idx + 15*input_size_);

                if (print_bool){
                    print_int8(A0, "A0");
                    print_int8(A1, "A1");
                }

                int B_idx = (l2_col_idx) * COLS_IN_BLOCK *  output_size_ + (l2_w_idx) * W_COL_BLOCKS;
                int8_t* weight8_t = (int8_t * ) weight;
                B0 = vld1q_s8(weight8_t + B_idx);
                C0 = vmlaq_s8(C0, A0, vmovq_n_s8(vgetq_lane_s8(B0, 0)));
                if (print_bool){
                    print_int8(B0, "B0");
                    print_int8(C0, "C0_0");
                }
                C0 = vmlaq_s8(C0, A1, vmovq_n_s8(vgetq_lane_s8(B0, 1)));
                if (print_bool){
                    print_int8(B0, "B0");
                    print_int8(C0, "C0_1");
                }
                C0 = vmlaq_s8(C0, A2, vmovq_n_s8(vgetq_lane_s8(B0, 2)));
                C0 = vmlaq_s8(C0, A3, vmovq_n_s8(vgetq_lane_s8(B0, 3)));
                C0 = vmlaq_s8(C0, A4, vmovq_n_s8(vgetq_lane_s8(B0, 4)));
                C0 = vmlaq_s8(C0, A5, vmovq_n_s8(vgetq_lane_s8(B0, 5)));
                C0 = vmlaq_s8(C0, A6, vmovq_n_s8(vgetq_lane_s8(B0, 6)));
                C0 = vmlaq_s8(C0, A7, vmovq_n_s8(vgetq_lane_s8(B0, 7)));
                C0 = vmlaq_s8(C0, A8, vmovq_n_s8(vgetq_lane_s8(B0, 8)));
                C0 = vmlaq_s8(C0, A9, vmovq_n_s8(vgetq_lane_s8(B0, 9)));
                C0 = vmlaq_s8(C0, A10, vmovq_n_s8(vgetq_lane_s8(B0, 10)));
                C0 = vmlaq_s8(C0, A11, vmovq_n_s8(vgetq_lane_s8(B0, 11)));
                C0 = vmlaq_s8(C0, A12, vmovq_n_s8(vgetq_lane_s8(B0, 12)));
                C0 = vmlaq_s8(C0, A13, vmovq_n_s8(vgetq_lane_s8(B0, 13)));
                C0 = vmlaq_s8(C0, A14, vmovq_n_s8(vgetq_lane_s8(B0, 14)));
                C0 = vmlaq_s8(C0, A15, vmovq_n_s8(vgetq_lane_s8(B0, 15)));

                B1 = vld1q_s8(weight8_t + B_idx + output_size_);
                C1 = vmlaq_s8(C1, A0, vmovq_n_s8(vgetq_lane_s8(B1, 0)));
                C1 = vmlaq_s8(C1, A1, vmovq_n_s8(vgetq_lane_s8(B1, 1)));
                C1 = vmlaq_s8(C1, A2, vmovq_n_s8(vgetq_lane_s8(B1, 2)));
                C1 = vmlaq_s8(C1, A3, vmovq_n_s8(vgetq_lane_s8(B1, 3)));
                C1 = vmlaq_s8(C1, A4, vmovq_n_s8(vgetq_lane_s8(B1, 4)));
                C1 = vmlaq_s8(C1, A5, vmovq_n_s8(vgetq_lane_s8(B1, 5)));
                C1 = vmlaq_s8(C1, A6, vmovq_n_s8(vgetq_lane_s8(B1, 6)));
                C1 = vmlaq_s8(C1, A7, vmovq_n_s8(vgetq_lane_s8(B1, 7)));
                C1 = vmlaq_s8(C1, A8, vmovq_n_s8(vgetq_lane_s8(B1, 8)));
                C1 = vmlaq_s8(C1, A9, vmovq_n_s8(vgetq_lane_s8(B1, 9)));
                C1 = vmlaq_s8(C1, A10, vmovq_n_s8(vgetq_lane_s8(B1, 10)));
                C1 = vmlaq_s8(C1, A11, vmovq_n_s8(vgetq_lane_s8(B1, 11)));
                C1 = vmlaq_s8(C1, A12, vmovq_n_s8(vgetq_lane_s8(B1, 12)));
                C1 = vmlaq_s8(C1, A13, vmovq_n_s8(vgetq_lane_s8(B1, 13)));
                C1 = vmlaq_s8(C1, A14, vmovq_n_s8(vgetq_lane_s8(B1, 14)));
                C1 = vmlaq_s8(C1, A15, vmovq_n_s8(vgetq_lane_s8(B1, 15)));

                B2 = vld1q_s8(weight8_t + B_idx + 2 * output_size_);
                C2 = vmlaq_s8(C2, A0, vmovq_n_s8(vgetq_lane_s8(B2, 0)));
                C2 = vmlaq_s8(C2, A1, vmovq_n_s8(vgetq_lane_s8(B2, 1)));
                C2 = vmlaq_s8(C2, A2, vmovq_n_s8(vgetq_lane_s8(B2, 2)));
                C2 = vmlaq_s8(C2, A3, vmovq_n_s8(vgetq_lane_s8(B2, 3)));
                C2 = vmlaq_s8(C2, A4, vmovq_n_s8(vgetq_lane_s8(B2, 4)));
                C2 = vmlaq_s8(C2, A5, vmovq_n_s8(vgetq_lane_s8(B2, 5)));
                C2 = vmlaq_s8(C2, A6, vmovq_n_s8(vgetq_lane_s8(B2, 6)));
                C2 = vmlaq_s8(C2, A7, vmovq_n_s8(vgetq_lane_s8(B2, 7)));
                C2 = vmlaq_s8(C2, A8, vmovq_n_s8(vgetq_lane_s8(B2, 8)));
                C2 = vmlaq_s8(C2, A9, vmovq_n_s8(vgetq_lane_s8(B2, 9)));
                C2 = vmlaq_s8(C2, A10, vmovq_n_s8(vgetq_lane_s8(B2, 10)));
                C2 = vmlaq_s8(C2, A11, vmovq_n_s8(vgetq_lane_s8(B2, 11)));
                C2 = vmlaq_s8(C2, A12, vmovq_n_s8(vgetq_lane_s8(B2, 12)));
                C2 = vmlaq_s8(C2, A13, vmovq_n_s8(vgetq_lane_s8(B2, 13)));
                C2 = vmlaq_s8(C2, A14, vmovq_n_s8(vgetq_lane_s8(B2, 14)));
                C2 = vmlaq_s8(C2, A15, vmovq_n_s8(vgetq_lane_s8(B2, 15)));

                B3 = vld1q_s8(weight8_t + B_idx + 3 * output_size_);
                C3 = vmlaq_s8(C3, A0,  vmovq_n_s8(vgetq_lane_s8(B3, 0)));
                C3 = vmlaq_s8(C3, A1,  vmovq_n_s8(vgetq_lane_s8(B3, 1)));
                C3 = vmlaq_s8(C3, A2,  vmovq_n_s8(vgetq_lane_s8(B3, 2)));
                C3 = vmlaq_s8(C3, A3,  vmovq_n_s8(vgetq_lane_s8(B3, 3)));
                C3 = vmlaq_s8(C3, A4,  vmovq_n_s8(vgetq_lane_s8(B3, 4)));
                C3 = vmlaq_s8(C3, A5,  vmovq_n_s8(vgetq_lane_s8(B3, 5)));
                C3 = vmlaq_s8(C3, A6,  vmovq_n_s8(vgetq_lane_s8(B3, 6)));
                C3 = vmlaq_s8(C3, A7,  vmovq_n_s8(vgetq_lane_s8(B3, 7)));
                C3 = vmlaq_s8(C3, A8,  vmovq_n_s8(vgetq_lane_s8(B3, 8)));
                C3 = vmlaq_s8(C3, A9,  vmovq_n_s8(vgetq_lane_s8(B3, 9)));
                C3 = vmlaq_s8(C3, A10, vmovq_n_s8(vgetq_lane_s8(B3, 10)));
                C3 = vmlaq_s8(C3, A11, vmovq_n_s8(vgetq_lane_s8(B3, 11)));
                C3 = vmlaq_s8(C3, A12, vmovq_n_s8(vgetq_lane_s8(B3, 12)));
                C3 = vmlaq_s8(C3, A13, vmovq_n_s8(vgetq_lane_s8(B3, 13)));
                C3 = vmlaq_s8(C3, A14, vmovq_n_s8(vgetq_lane_s8(B3, 14)));
                C3 = vmlaq_s8(C3, A15, vmovq_n_s8(vgetq_lane_s8(B3, 15)));

                B4 = vld1q_s8(weight8_t + B_idx + 4 * output_size_);
                C4 = vmlaq_s8(C4, A0,  vmovq_n_s8(vgetq_lane_s8(B4, 0)));
                C4 = vmlaq_s8(C4, A1,  vmovq_n_s8(vgetq_lane_s8(B4, 1)));
                C4 = vmlaq_s8(C4, A2,  vmovq_n_s8(vgetq_lane_s8(B4, 2)));
                C4 = vmlaq_s8(C4, A3,  vmovq_n_s8(vgetq_lane_s8(B4, 3)));
                C4 = vmlaq_s8(C4, A4,  vmovq_n_s8(vgetq_lane_s8(B4, 4)));
                C4 = vmlaq_s8(C4, A5,  vmovq_n_s8(vgetq_lane_s8(B4, 5)));
                C4 = vmlaq_s8(C4, A6,  vmovq_n_s8(vgetq_lane_s8(B4, 6)));
                C4 = vmlaq_s8(C4, A7,  vmovq_n_s8(vgetq_lane_s8(B4, 7)));
                C4 = vmlaq_s8(C4, A8,  vmovq_n_s8(vgetq_lane_s8(B4, 8)));
                C4 = vmlaq_s8(C4, A9,  vmovq_n_s8(vgetq_lane_s8(B4, 9)));
                C4 = vmlaq_s8(C4, A10, vmovq_n_s8(vgetq_lane_s8(B4, 10)));
                C4 = vmlaq_s8(C4, A11, vmovq_n_s8(vgetq_lane_s8(B4, 11)));
                C4 = vmlaq_s8(C4, A12, vmovq_n_s8(vgetq_lane_s8(B4, 12)));
                C4 = vmlaq_s8(C4, A13, vmovq_n_s8(vgetq_lane_s8(B4, 13)));
                C4 = vmlaq_s8(C4, A14, vmovq_n_s8(vgetq_lane_s8(B4, 14)));
                C4 = vmlaq_s8(C4, A15, vmovq_n_s8(vgetq_lane_s8(B4, 15)));

                B5 = vld1q_s8(weight8_t + B_idx + 5 * output_size_);
                C5 = vmlaq_s8(C5, A0,  vmovq_n_s8(vgetq_lane_s8(B5, 0)));
                C5 = vmlaq_s8(C5, A1,  vmovq_n_s8(vgetq_lane_s8(B5, 1)));
                C5 = vmlaq_s8(C5, A2,  vmovq_n_s8(vgetq_lane_s8(B5, 2)));
                C5 = vmlaq_s8(C5, A3,  vmovq_n_s8(vgetq_lane_s8(B5, 3)));
                C5 = vmlaq_s8(C5, A4,  vmovq_n_s8(vgetq_lane_s8(B5, 4)));
                C5 = vmlaq_s8(C5, A5,  vmovq_n_s8(vgetq_lane_s8(B5, 5)));
                C5 = vmlaq_s8(C5, A6,  vmovq_n_s8(vgetq_lane_s8(B5, 6)));
                C5 = vmlaq_s8(C5, A7,  vmovq_n_s8(vgetq_lane_s8(B5, 7)));
                C5 = vmlaq_s8(C5, A8,  vmovq_n_s8(vgetq_lane_s8(B5, 8)));
                C5 = vmlaq_s8(C5, A9,  vmovq_n_s8(vgetq_lane_s8(B5, 9)));
                C5 = vmlaq_s8(C5, A10, vmovq_n_s8(vgetq_lane_s8(B5, 10)));
                C5 = vmlaq_s8(C5, A11, vmovq_n_s8(vgetq_lane_s8(B5, 11)));
                C5 = vmlaq_s8(C5, A12, vmovq_n_s8(vgetq_lane_s8(B5, 12)));
                C5 = vmlaq_s8(C5, A13, vmovq_n_s8(vgetq_lane_s8(B5, 13)));
                C5 = vmlaq_s8(C5, A14, vmovq_n_s8(vgetq_lane_s8(B5, 14)));
                C5 = vmlaq_s8(C5, A15, vmovq_n_s8(vgetq_lane_s8(B5, 15)));

                B6 = vld1q_s8(weight8_t + B_idx + 6 * output_size_);
                C6 = vmlaq_s8(C6, A0,  vmovq_n_s8(vgetq_lane_s8(B6, 0)));
                C6 = vmlaq_s8(C6, A1,  vmovq_n_s8(vgetq_lane_s8(B6, 1)));
                C6 = vmlaq_s8(C6, A2,  vmovq_n_s8(vgetq_lane_s8(B6, 2)));
                C6 = vmlaq_s8(C6, A3,  vmovq_n_s8(vgetq_lane_s8(B6, 3)));
                C6 = vmlaq_s8(C6, A4,  vmovq_n_s8(vgetq_lane_s8(B6, 4)));
                C6 = vmlaq_s8(C6, A5,  vmovq_n_s8(vgetq_lane_s8(B6, 5)));
                C6 = vmlaq_s8(C6, A6,  vmovq_n_s8(vgetq_lane_s8(B6, 6)));
                C6 = vmlaq_s8(C6, A7,  vmovq_n_s8(vgetq_lane_s8(B6, 7)));
                C6 = vmlaq_s8(C6, A8,  vmovq_n_s8(vgetq_lane_s8(B6, 8)));
                C6 = vmlaq_s8(C6, A9,  vmovq_n_s8(vgetq_lane_s8(B6, 9)));
                C6 = vmlaq_s8(C6, A10, vmovq_n_s8(vgetq_lane_s8(B6, 10)));
                C6 = vmlaq_s8(C6, A11, vmovq_n_s8(vgetq_lane_s8(B6, 11)));
                C6 = vmlaq_s8(C6, A12, vmovq_n_s8(vgetq_lane_s8(B6, 12)));
                C6 = vmlaq_s8(C6, A13, vmovq_n_s8(vgetq_lane_s8(B6, 13)));
                C6 = vmlaq_s8(C6, A14, vmovq_n_s8(vgetq_lane_s8(B6, 14)));
                C6 = vmlaq_s8(C6, A15, vmovq_n_s8(vgetq_lane_s8(B6, 15)));

                B7 = vld1q_s8(weight8_t + B_idx + 7 * output_size_);
                C7 = vmlaq_s8(C7, A0, vmovq_n_s8(vgetq_lane_s8(B7, 0)));
                C7 = vmlaq_s8(C7, A1, vmovq_n_s8(vgetq_lane_s8(B7, 1)));
                C7 = vmlaq_s8(C7, A2, vmovq_n_s8(vgetq_lane_s8(B7, 2)));
                C7 = vmlaq_s8(C7, A3, vmovq_n_s8(vgetq_lane_s8(B7, 3)));
                C7 = vmlaq_s8(C7, A4, vmovq_n_s8(vgetq_lane_s8(B7, 4)));
                C7 = vmlaq_s8(C7, A5, vmovq_n_s8(vgetq_lane_s8(B7, 5)));
                C7 = vmlaq_s8(C7, A6, vmovq_n_s8(vgetq_lane_s8(B7, 6)));
                C7 = vmlaq_s8(C7, A7, vmovq_n_s8(vgetq_lane_s8(B7, 7)));
                C7 = vmlaq_s8(C7, A8, vmovq_n_s8(vgetq_lane_s8(B7, 8)));
                C7 = vmlaq_s8(C7, A9, vmovq_n_s8(vgetq_lane_s8(B7, 9)));
                C7 = vmlaq_s8(C7, A10,vmovq_n_s8(vgetq_lane_s8(B7, 10)));
                C7 = vmlaq_s8(C7, A11,vmovq_n_s8(vgetq_lane_s8(B7, 11)));
                C7 = vmlaq_s8(C7, A12,vmovq_n_s8(vgetq_lane_s8(B7, 12)));
                C7 = vmlaq_s8(C7, A13,vmovq_n_s8(vgetq_lane_s8(B7, 13)));
                C7 = vmlaq_s8(C7, A14,vmovq_n_s8(vgetq_lane_s8(B7, 14)));
                C7 = vmlaq_s8(C7, A15,vmovq_n_s8(vgetq_lane_s8(B7, 15)));

                B8 = vld1q_s8(weight8_t + B_idx + 8 * output_size_);
                C8 = vmlaq_s8(C8, A0, vmovq_n_s8(vgetq_lane_s8(B8, 0)));
                C8 = vmlaq_s8(C8, A1, vmovq_n_s8(vgetq_lane_s8(B8, 1)));
                C8 = vmlaq_s8(C8, A2, vmovq_n_s8(vgetq_lane_s8(B8, 2)));
                C8 = vmlaq_s8(C8, A3, vmovq_n_s8(vgetq_lane_s8(B8, 3)));
                C8 = vmlaq_s8(C8, A4, vmovq_n_s8(vgetq_lane_s8(B8, 4)));
                C8 = vmlaq_s8(C8, A5, vmovq_n_s8(vgetq_lane_s8(B8, 5)));
                C8 = vmlaq_s8(C8, A6, vmovq_n_s8(vgetq_lane_s8(B8, 6)));
                C8 = vmlaq_s8(C8, A7, vmovq_n_s8(vgetq_lane_s8(B8, 7)));
                C8 = vmlaq_s8(C8, A8, vmovq_n_s8(vgetq_lane_s8(B8, 8)));
                C8 = vmlaq_s8(C8, A9, vmovq_n_s8(vgetq_lane_s8(B8, 9)));
                C8 = vmlaq_s8(C8, A10,vmovq_n_s8(vgetq_lane_s8(B8, 10)));
                C8 = vmlaq_s8(C8, A11,vmovq_n_s8(vgetq_lane_s8(B8, 11)));
                C8 = vmlaq_s8(C8, A12,vmovq_n_s8(vgetq_lane_s8(B8, 12)));
                C8 = vmlaq_s8(C8, A13,vmovq_n_s8(vgetq_lane_s8(B8, 13)));
                C8 = vmlaq_s8(C8, A14,vmovq_n_s8(vgetq_lane_s8(B8, 14)));
                C8 = vmlaq_s8(C8, A15,vmovq_n_s8(vgetq_lane_s8(B8, 15)));

                B9 = vld1q_s8(weight8_t + B_idx + 9 * output_size_);
                C9 = vmlaq_s8(C9, A0, vmovq_n_s8(vgetq_lane_s8(B9, 0)));
                C9 = vmlaq_s8(C9, A1, vmovq_n_s8(vgetq_lane_s8(B9, 1)));
                C9 = vmlaq_s8(C9, A2, vmovq_n_s8(vgetq_lane_s8(B9, 2)));
                C9 = vmlaq_s8(C9, A3, vmovq_n_s8(vgetq_lane_s8(B9, 3)));
                C9 = vmlaq_s8(C9, A4, vmovq_n_s8(vgetq_lane_s8(B9, 4)));
                C9 = vmlaq_s8(C9, A5, vmovq_n_s8(vgetq_lane_s8(B9, 5)));
                C9 = vmlaq_s8(C9, A6, vmovq_n_s8(vgetq_lane_s8(B9, 6)));
                C9 = vmlaq_s8(C9, A7, vmovq_n_s8(vgetq_lane_s8(B9, 7)));
                C9 = vmlaq_s8(C9, A8, vmovq_n_s8(vgetq_lane_s8(B9, 8)));
                C9 = vmlaq_s8(C9, A9, vmovq_n_s8(vgetq_lane_s8(B9, 9)));
                C9 = vmlaq_s8(C9, A10,vmovq_n_s8(vgetq_lane_s8(B9, 10)));
                C9 = vmlaq_s8(C9, A11,vmovq_n_s8(vgetq_lane_s8(B9, 11)));
                C9 = vmlaq_s8(C9, A12,vmovq_n_s8(vgetq_lane_s8(B9, 12)));
                C9 = vmlaq_s8(C9, A13,vmovq_n_s8(vgetq_lane_s8(B9, 13)));
                C9 = vmlaq_s8(C9, A14,vmovq_n_s8(vgetq_lane_s8(B9, 14)));
                C9 = vmlaq_s8(C9, A15,vmovq_n_s8(vgetq_lane_s8(B9, 15)));

                B10 = vld1q_s8(weight8_t + B_idx + 10 * output_size_);
                C10 = vmlaq_s8(C10, A0, vmovq_n_s8(vgetq_lane_s8(B10, 0)));
                C10 = vmlaq_s8(C10, A1, vmovq_n_s8(vgetq_lane_s8(B10, 1)));
                C10 = vmlaq_s8(C10, A2, vmovq_n_s8(vgetq_lane_s8(B10, 2)));
                C10 = vmlaq_s8(C10, A3, vmovq_n_s8(vgetq_lane_s8(B10, 3)));
                C10 = vmlaq_s8(C10, A4, vmovq_n_s8(vgetq_lane_s8(B10, 4)));
                C10 = vmlaq_s8(C10, A5, vmovq_n_s8(vgetq_lane_s8(B10, 5)));
                C10 = vmlaq_s8(C10, A6, vmovq_n_s8(vgetq_lane_s8(B10, 6)));
                C10 = vmlaq_s8(C10, A7, vmovq_n_s8(vgetq_lane_s8(B10, 7)));
                C10 = vmlaq_s8(C10, A8, vmovq_n_s8(vgetq_lane_s8(B10, 8)));
                C10 = vmlaq_s8(C10, A9, vmovq_n_s8(vgetq_lane_s8(B10, 9)));
                C10 = vmlaq_s8(C10, A10,vmovq_n_s8(vgetq_lane_s8(B10, 10)));
                C10 = vmlaq_s8(C10, A11,vmovq_n_s8(vgetq_lane_s8(B10, 11)));
                C10 = vmlaq_s8(C10, A12,vmovq_n_s8(vgetq_lane_s8(B10, 12)));
                C10 = vmlaq_s8(C10, A13,vmovq_n_s8(vgetq_lane_s8(B10, 13)));
                C10 = vmlaq_s8(C10, A14,vmovq_n_s8(vgetq_lane_s8(B10, 14)));
                C10 = vmlaq_s8(C10, A15,vmovq_n_s8(vgetq_lane_s8(B10, 15)));

                B11 = vld1q_s8(weight8_t + B_idx + 11 * output_size_);
                C11 = vmlaq_s8(C11, A0, vmovq_n_s8(vgetq_lane_s8(B11, 0)));
                C11 = vmlaq_s8(C11, A1, vmovq_n_s8(vgetq_lane_s8(B11, 1)));
                C11 = vmlaq_s8(C11, A2, vmovq_n_s8(vgetq_lane_s8(B11, 2)));
                C11 = vmlaq_s8(C11, A3, vmovq_n_s8(vgetq_lane_s8(B11, 3)));
                C11 = vmlaq_s8(C11, A4, vmovq_n_s8(vgetq_lane_s8(B11, 4)));
                C11 = vmlaq_s8(C11, A5, vmovq_n_s8(vgetq_lane_s8(B11, 5)));
                C11 = vmlaq_s8(C11, A6, vmovq_n_s8(vgetq_lane_s8(B11, 6)));
                C11 = vmlaq_s8(C11, A7, vmovq_n_s8(vgetq_lane_s8(B11, 7)));
                C11 = vmlaq_s8(C11, A8, vmovq_n_s8(vgetq_lane_s8(B11, 8)));
                C11 = vmlaq_s8(C11, A9, vmovq_n_s8(vgetq_lane_s8(B11, 9)));
                C11 = vmlaq_s8(C11, A10,vmovq_n_s8(vgetq_lane_s8(B11, 10)));
                C11 = vmlaq_s8(C11, A11,vmovq_n_s8(vgetq_lane_s8(B11, 11)));
                C11 = vmlaq_s8(C11, A12,vmovq_n_s8(vgetq_lane_s8(B11, 12)));
                C11 = vmlaq_s8(C11, A13,vmovq_n_s8(vgetq_lane_s8(B11, 13)));
                C11 = vmlaq_s8(C11, A14,vmovq_n_s8(vgetq_lane_s8(B11, 14)));
                C11 = vmlaq_s8(C11, A15,vmovq_n_s8(vgetq_lane_s8(B11, 15)));

                B12 = vld1q_s8(weight8_t + B_idx + 12 * output_size_);
                C12 = vmlaq_s8(C12, A0, vmovq_n_s8(vgetq_lane_s8(B12, 0)));
                C12 = vmlaq_s8(C12, A1, vmovq_n_s8(vgetq_lane_s8(B12, 1)));
                C12 = vmlaq_s8(C12, A2, vmovq_n_s8(vgetq_lane_s8(B12, 2)));
                C12 = vmlaq_s8(C12, A3, vmovq_n_s8(vgetq_lane_s8(B12, 3)));
                C12 = vmlaq_s8(C12, A4, vmovq_n_s8(vgetq_lane_s8(B12, 4)));
                C12 = vmlaq_s8(C12, A5, vmovq_n_s8(vgetq_lane_s8(B12, 5)));
                C12 = vmlaq_s8(C12, A6, vmovq_n_s8(vgetq_lane_s8(B12, 6)));
                C12 = vmlaq_s8(C12, A7, vmovq_n_s8(vgetq_lane_s8(B12, 7)));
                C12 = vmlaq_s8(C12, A8, vmovq_n_s8(vgetq_lane_s8(B12, 8)));
                C12 = vmlaq_s8(C12, A9, vmovq_n_s8(vgetq_lane_s8(B12, 9)));
                C12 = vmlaq_s8(C12, A10,vmovq_n_s8(vgetq_lane_s8(B12, 10)));
                C12 = vmlaq_s8(C12, A11,vmovq_n_s8(vgetq_lane_s8(B12, 11)));
                C12 = vmlaq_s8(C12, A12,vmovq_n_s8(vgetq_lane_s8(B12, 12)));
                C12 = vmlaq_s8(C12, A13,vmovq_n_s8(vgetq_lane_s8(B12, 13)));
                C12 = vmlaq_s8(C12, A14,vmovq_n_s8(vgetq_lane_s8(B12, 14)));
                C12 = vmlaq_s8(C12, A15,vmovq_n_s8(vgetq_lane_s8(B12, 15)));

                B13 = vld1q_s8(weight8_t + B_idx + 13 * output_size_);
                C13 = vmlaq_s8(C13, A0, vmovq_n_s8(vgetq_lane_s8(B13, 0)));
                C13 = vmlaq_s8(C13, A1, vmovq_n_s8(vgetq_lane_s8(B13, 1)));
                C13 = vmlaq_s8(C13, A2, vmovq_n_s8(vgetq_lane_s8(B13, 2)));
                C13 = vmlaq_s8(C13, A3, vmovq_n_s8(vgetq_lane_s8(B13, 3)));
                C13 = vmlaq_s8(C13, A4, vmovq_n_s8(vgetq_lane_s8(B13, 4)));
                C13 = vmlaq_s8(C13, A5, vmovq_n_s8(vgetq_lane_s8(B13, 5)));
                C13 = vmlaq_s8(C13, A6, vmovq_n_s8(vgetq_lane_s8(B13, 6)));
                C13 = vmlaq_s8(C13, A7, vmovq_n_s8(vgetq_lane_s8(B13, 7)));
                C13 = vmlaq_s8(C13, A8, vmovq_n_s8(vgetq_lane_s8(B13, 8)));
                C13 = vmlaq_s8(C13, A9, vmovq_n_s8(vgetq_lane_s8(B13, 9)));
                C13 = vmlaq_s8(C13, A10,vmovq_n_s8(vgetq_lane_s8(B13, 10)));
                C13 = vmlaq_s8(C13, A11,vmovq_n_s8(vgetq_lane_s8(B13, 11)));
                C13 = vmlaq_s8(C13, A12,vmovq_n_s8(vgetq_lane_s8(B13, 12)));
                C13 = vmlaq_s8(C13, A13,vmovq_n_s8(vgetq_lane_s8(B13, 13)));
                C13 = vmlaq_s8(C13, A14,vmovq_n_s8(vgetq_lane_s8(B13, 14)));
                C13 = vmlaq_s8(C13, A15,vmovq_n_s8(vgetq_lane_s8(B13, 15)));

                B14 = vld1q_s8(weight8_t + B_idx + 14 * output_size_);
                C14 = vmlaq_s8(C14, A0, vmovq_n_s8(vgetq_lane_s8(B14, 0)));
                C14 = vmlaq_s8(C14, A1, vmovq_n_s8(vgetq_lane_s8(B14, 1)));
                C14 = vmlaq_s8(C14, A2, vmovq_n_s8(vgetq_lane_s8(B14, 2)));
                C14 = vmlaq_s8(C14, A3, vmovq_n_s8(vgetq_lane_s8(B14, 3)));
                C14 = vmlaq_s8(C14, A4, vmovq_n_s8(vgetq_lane_s8(B14, 4)));
                C14 = vmlaq_s8(C14, A5, vmovq_n_s8(vgetq_lane_s8(B14, 5)));
                C14 = vmlaq_s8(C14, A6, vmovq_n_s8(vgetq_lane_s8(B14, 6)));
                C14 = vmlaq_s8(C14, A7, vmovq_n_s8(vgetq_lane_s8(B14, 7)));
                C14 = vmlaq_s8(C14, A8, vmovq_n_s8(vgetq_lane_s8(B14, 8)));
                C14 = vmlaq_s8(C14, A9, vmovq_n_s8(vgetq_lane_s8(B14, 9)));
                C14 = vmlaq_s8(C14, A10,vmovq_n_s8(vgetq_lane_s8(B14, 10)));
                C14 = vmlaq_s8(C14, A11,vmovq_n_s8(vgetq_lane_s8(B14, 11)));
                C14 = vmlaq_s8(C14, A12,vmovq_n_s8(vgetq_lane_s8(B14, 12)));
                C14 = vmlaq_s8(C14, A13,vmovq_n_s8(vgetq_lane_s8(B14, 13)));
                C14 = vmlaq_s8(C14, A14,vmovq_n_s8(vgetq_lane_s8(B14, 14)));
                C14 = vmlaq_s8(C14, A15,vmovq_n_s8(vgetq_lane_s8(B14, 15)));

                B15 = vld1q_s8(weight8_t + B_idx + 15 * output_size_);
                C15 = vmlaq_s8(C15, A0, vmovq_n_s8(vgetq_lane_s8(B15, 0)));
                C15 = vmlaq_s8(C15, A1, vmovq_n_s8(vgetq_lane_s8(B15, 1)));
                C15 = vmlaq_s8(C15, A2, vmovq_n_s8(vgetq_lane_s8(B15, 2)));
                C15 = vmlaq_s8(C15, A3, vmovq_n_s8(vgetq_lane_s8(B15, 3)));
                C15 = vmlaq_s8(C15, A4, vmovq_n_s8(vgetq_lane_s8(B15, 4)));
                C15 = vmlaq_s8(C15, A5, vmovq_n_s8(vgetq_lane_s8(B15, 5)));
                C15 = vmlaq_s8(C15, A6, vmovq_n_s8(vgetq_lane_s8(B15, 6)));
                C15 = vmlaq_s8(C15, A7, vmovq_n_s8(vgetq_lane_s8(B15, 7)));
                C15 = vmlaq_s8(C15, A8, vmovq_n_s8(vgetq_lane_s8(B15, 8)));
                C15 = vmlaq_s8(C15, A9, vmovq_n_s8(vgetq_lane_s8(B15, 9)));
                C15 = vmlaq_s8(C15, A10,vmovq_n_s8(vgetq_lane_s8(B15, 10)));
                C15 = vmlaq_s8(C15, A11,vmovq_n_s8(vgetq_lane_s8(B15, 11)));
                C15 = vmlaq_s8(C15, A12,vmovq_n_s8(vgetq_lane_s8(B15, 12)));
                C15 = vmlaq_s8(C15, A13,vmovq_n_s8(vgetq_lane_s8(B15, 13)));
                C15 = vmlaq_s8(C15, A14,vmovq_n_s8(vgetq_lane_s8(B15, 14)));
                C15 = vmlaq_s8(C15, A15,vmovq_n_s8(vgetq_lane_s8(B15, 15)));
            }
            bool print_bool = (l2_row_idx == 0 && l2_col_idx == 0);
            if (print_bool)
                print_int8(C0, "C0_final");

            vst1q_s8(output8_t + C_idx, C0);
            vst1q_s8(output8_t + C_idx + W_COL_BLOCKS, C1);
            vst1q_s8(output8_t + C_idx + 2* W_COL_BLOCKS, C2);
            vst1q_s8(output8_t + C_idx + 3* W_COL_BLOCKS, C3);
            vst1q_s8(output8_t + C_idx + 4* W_COL_BLOCKS, C4);
            vst1q_s8(output8_t + C_idx + 5* W_COL_BLOCKS, C5);
            vst1q_s8(output8_t + C_idx + 6* W_COL_BLOCKS, C6);
            vst1q_s8(output8_t + C_idx + 7* W_COL_BLOCKS, C7);
            vst1q_s8(output8_t + C_idx + 8* W_COL_BLOCKS, C8);
            vst1q_s8(output8_t + C_idx + 9* W_COL_BLOCKS, C9);
            vst1q_s8(output8_t + C_idx + 10* W_COL_BLOCKS, C10);
            vst1q_s8(output8_t + C_idx + 11* W_COL_BLOCKS, C11);
            vst1q_s8(output8_t + C_idx + 12* W_COL_BLOCKS, C12);
            vst1q_s8(output8_t + C_idx + 13* W_COL_BLOCKS, C13);
            vst1q_s8(output8_t + C_idx + 14* W_COL_BLOCKS, C14);
            vst1q_s8(output8_t + C_idx + 15* W_COL_BLOCKS, C15);
        }
    }
}

#endif



