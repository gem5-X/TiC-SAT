//
// Created by alireza on 1/19/22.
//
#include <cstdint>
#include "iostream"
#define W_DIM 16
#define W_DATA 4
#define MAX_COL 4
#define KERNEL_DIM 16
#define Nx 512
#define M 512
#define Pw 64

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]


extern uint32_t inputArray[];
extern uint32_t weights[];

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
uint64_t smmStream(uint64_t rm, uint64_t rn)
{
    uint64_t res;

    __asm__ volatile(
            "MOV X9, %[input_j];"
            "MOV X7, %[input_k];"
            ".long 0x01081D2A;"
            "MOV %[output], X10;"
            : [output] "=r" (res)
            : [input_j] "r" (rm), [input_k] "r" (rn)
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
uint64_t smmParamWrite(uint64_t rm, uint64_t rn, uint64_t ra)
{
    uint64_t res;

    __asm__ volatile(
            "MOV X8, %[input_i];"
            "MOV X9, %[input_j];"
            "MOV X7, %[input_k];"
            ".long 0x41081D2A;"
            "MOV %[output], X10;"
            : [output] "=r" (res)
            : [input_i] "r" (ra), [input_j] "r" (rm), [input_k] "r" (rn)
            : "x7", "x8", "x9", "x10"
            );

    return res;
}



int main() {
    uint32_t outputArray[Nx * Pw / W_DATA] = {0};
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
                    smmParamWrite(i - rowStart, j - colStart, weight);
                }
            }

            // Process the multiplication
            int base_col_idx = tileRow * MAX_COL;
            int outputIndex = 0;
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < MAX_COL; j++) {
                    uint32_t mult = smmStream(j % MAX_COL, mem2d(inputArray, M / W_DATA, i, j + base_col_idx));
                    if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                        mem2d(outputArray, Pw / W_DATA, outputIndex / colBlockSize,
                              colStart + outputIndex % colBlockSize) += mult;
                        outputIndex++;
                    }
                }
            }
            for (int i = Nx * MAX_COL; i < MAX_COL * (Nx + 2 * KERNEL_DIM - 1) - 1; i++) {
                uint32_t mult = smmStream(i % MAX_COL, 0);
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

    return 0;
}