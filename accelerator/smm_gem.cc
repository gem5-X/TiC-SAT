//
// Created by alireza on 3/3/22.
//
#include "smm_gem.h"

void addActIn32(uint32_t &memory, uint32_t &systolicResult);

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
* -- rd = Systolic Array Output.
* -- rm = Unused.
* -- ra = Thread index.
* -- rn = Parameter value.
*/

uint64_t smmStream(uint64_t rn, uint64_t tid=0) {
    uint64_t res;

    __asm__ volatile(
    "MOV X7, %[input_k];"
    "MOV X8, %[input_i];"
    ".long 0x01081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_k] "r"(rn), [input_i] "r"(tid)
    : "x7", "x8", "x10"
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
* -- rd = Systolic Array output.
* -- rm = Parameter index.
* -- ra = Thread index.
* -- rn = Parameter value.
*/
uint64_t smmQueue(uint64_t rm, uint64_t rn, uint64_t tid=0) {
    uint64_t res;

    __asm__ volatile(
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    "MOV X8, %[input_i];"
    ".long 0x21089D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_j] "r"(rm), [input_i] "r"(tid), [input_k] "r"(rn)
    : "x7", "x8", "x9", "x10"
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
* -- rd = Zero tile code.
* -- rm = Parameter index.
* -- ra = Thread index.
* -- rn = Parameter value.
 */
uint64_t smmParamWrite(uint64_t rm, uint64_t rn, uint64_t tid) {
    uint64_t res;

    __asm__ volatile(
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    "MOV X8, %[input_i];"
    ".long 0x41081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_j] "r"(rm), [input_i] "r"(tid), [input_k] "r"(rn)
    : "x7", "x8", "x9", "x10"
    );

    return res;

}

/* CM Core Memory Read
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0110_0001_000|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__6____1____0|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Read value.
 * -- rm = _.
 * -- ra = 'val' parameter to readFlag.
 * -- rn = _.
*/
uint64_t smmReadFlag(uint64_t val, uint64_t tid) {
    uint64_t res;

    uint64_t rm = 0xdeadbeef;

    __asm__ volatile(
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    "MOV X8, %[input_i];"
    ".long 0x61081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_j] "r"(rm), [input_i] "r"(tid), [input_k] "r"(val)
    : "x7", "x8", "x9", "x10"
    );

    return res;
}

void printWeights(int tid) {
  // Do nothing
}

#else

#include "systolic_m2m.hh"

SystolicMatrixMultiplication smm0 = SystolicMatrixMultiplication();
SystolicMatrixMultiplication smm1 = SystolicMatrixMultiplication();
SystolicMatrixMultiplication smm2 = SystolicMatrixMultiplication();
SystolicMatrixMultiplication smm3 = SystolicMatrixMultiplication();
SystolicMatrixMultiplication smmList[] = {smm0, smm1, smm2, smm3};

bool smmParamWrite(int rm, uint32_t ra, int tid) {
    return smmList[tid].loadWeights(rm, ra);
}

uint32_t smmQueue(int rm, uint32_t ra, int tid) {
    return smmList[tid].inputQueue(rm, ra);
}

uint32_t smmStream(int rm, uint32_t rn, int tid) {
    return smmList[tid].streamInOut(rm, rn);
}

uint64_t smmReadFlag(uint64_t val, uint64_t tid) {
    return smmList[tid].readFlag(val);
}

void printWeights(int tid) {
  smmList[tid].printWeights();
}

#endif

void addActIn32(uint32_t &memory, uint32_t &systolicResult) {
    /*
     * This function separates every 32-bit input to four 8-bit integers and add them. Then, packing again and
     * make a 32-bit  unsigned int.
     */
    auto *mem_ptr = (activation_t *) (&memory);
    auto *sys_ptr = (activation_t *) (&systolicResult);
    for (int i = 0; i < ACT_PER_BUS; i++) {
#if ACTIVATION_FP == 1
        arith_activation_t in_mem, in_sys, out;
        in_mem.bin = *mem_ptr;
        in_sys.bin = *sys_ptr;
        out.fp = in_mem.fp + in_sys.fp;
        *mem_ptr = out.bin;
#else
        *mem_ptr = (activation_t) (*mem_ptr + *sys_ptr);
#endif
        mem_ptr++;
        sys_ptr++;
    }
}
