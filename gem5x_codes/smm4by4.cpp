//
// Created by alireza on 1/19/22.
//
#include <cstdint>
#include "iostream"
#define W_DIM 4

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
uint64_t smmStream(uint64_t rm, uint64_t rn, uint64_t ra)
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
    uint32_t weights[] = {0x01020103, 0x01020103, 0x01020103, 0x01020103};
    uint32_t inputArray[] = {0x00000001, 0x00000102, 0x00030402, 0x01010101, 0x04020300, 0x01000000, 0x02000000, 0, 0, 0, 0};
    for (int i=0; i< W_DIM; i++){
        smmParamWrite(i, 0, weights[i]);
    }
    for(uint32_t in : inputArray){
        std::cout<< smmStream(in, 0, 0) << std::endl;
    }
    return 0;
}