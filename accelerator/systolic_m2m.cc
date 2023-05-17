/*
 * Copyright (c) 2020 EPFL
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Alireza Amirshahi
 */
// Constructor.
#include <cstdint>
#include <iostream>
#include "systolic_m2m.hh"

bool SystolicMatrixMultiplication::loadWeights(int row, int col, uint32_t val) {
    int idx= row * KERNEL_DIM + col * W_DATA;
    for (int i=0; i < W_DATA; i++){
        auto currVal = (int8_t)((val >> (8 * (W_DATA -i-1))) & 0xff);
        weights[idx + i] = currVal;
    }
    if (val!=0)
        non_zero_tile = true;
    return non_zero_tile;
}

uint32_t SystolicMatrixMultiplication::inputQueue(int col, uint32_t val) {
    // Split the input to an array
    for (int i=0; i < W_DATA; i++){
        auto currVal = (int8_t)((val >> (8 * (W_DATA - i -1))) & 0xff);
        int row_index = (col*W_DATA+i);
        mem2d(inWaitingMemory, KERNEL_DIM, row_index, KERNEL_DIM - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }

    // Return the output
    uint32_t result = 0;
    int result_idx = ((col+1)%MAX_COL) * W_DATA;
    for (int i = 0; i < W_DATA; i++) {
        result |=  mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - result_idx - i - 1, result_idx + i ) << (8 * (W_DATA - i - 1));
    }
    return result;
}

void SystolicMatrixMultiplication::printWeights() {
    std::cout << std::hex << (uint32_t) weights[0] << std::endl;
}

uint32_t SystolicMatrixMultiplication::streamInOut(uint32_t val) {
    non_zero_tile = false;
	int col = MAX_COL - 1;
    // Split the input to an array
    for (int i=0; i < W_DATA; i++){
        auto currVal = (int8_t)((val >> (8 * (W_DATA - i -1))) & 0xff);
        int row_index = (col*W_DATA+i);
        mem2d(inWaitingMemory, KERNEL_DIM, row_index, KERNEL_DIM - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }

    // Shift the waiting memory to the right for skewing
    for (int i = 0; i < KERNEL_DIM; i++) {
        mem2d(inputMemory, KERNEL_DIM, i, 0) = mem2d(inWaitingMemory, KERNEL_DIM, i, KERNEL_DIM - 1);
        for (int j= KERNEL_DIM - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
            mem2d(inWaitingMemory, KERNEL_DIM, i, j) = mem2d(inWaitingMemory, KERNEL_DIM, i, j - 1);
        }
    }

    // Multiply the input to the weight and accumulate to the output
    for (int i= KERNEL_DIM * KERNEL_DIM - 1; i >= 0 ; i--){
        outputMemory[i + KERNEL_DIM] = int(inputMemory[i] * weights[i]) + outputMemory[i];
    }

    // Shift the input memory to the right
    for (int i = 0; i < KERNEL_DIM; i++) {
        for (int j= KERNEL_DIM - 1; j > 0; j--){
            inputMemory[i * KERNEL_DIM + j] = inputMemory[i * KERNEL_DIM + j - 1];
        }
    }

    // Shift the outWaitingMemory because of the skew in the output
    for (int j = 0; j < KERNEL_DIM; j++) {
        for (int i= KERNEL_DIM - 1; i > 0; i--){ // TODO: shift only the right-hand triangle
            mem2d(outWaitingMemory, KERNEL_DIM, i, j) = mem2d(outWaitingMemory, KERNEL_DIM, i - 1, j);
        }
//        std::cout << std::hex << mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) << std::endl;
        mem2d(outWaitingMemory, KERNEL_DIM, 0, j) = (uint8_t)(mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) & 0xFF);
    }


    // Return the output
    uint32_t result = 0;
    for (int i = 0; i < W_DATA; i++) {
        result |=  mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ) << (8 * (W_DATA - i - 1));
//        std::cout << std::hex << (int) mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ) << ",";
    }
    return result;

}
