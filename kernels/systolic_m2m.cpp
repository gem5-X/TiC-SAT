//
// Created by alireza on 1/13/22.
//

#include <cstdint>
#include "iostream"
#include "systolic_m2m.h"

SystolicMatrixMultiplication::SystolicMatrixMultiplication() = default;

void SystolicMatrixMultiplication::loadWeights(int row, int col, uint32_t val) {
    int idx= row * KERNEL_DIM + col * W_DATA;
    for (int i=0; i < W_DATA; i++){
        uint8_t currVal = ((val >> (8 * (W_DATA -i-1))) & 0xff);
        weights[idx + i] = currVal;
    }
}

void SystolicMatrixMultiplication::printWeights() {
    for (int i=0; i < KERNEL_DIM; i++){
        for (int j=0; j < KERNEL_DIM; j++)
            std::cout << std::hex << (uint32_t) mem2d(weights, KERNEL_DIM, i, j) << "\t";
        std::cout << std::endl;
    }
}

uint32_t SystolicMatrixMultiplication::streamInOut(int col, uint32_t val) {
    // Split the input to an array
    for (int i=0; i < W_DATA; i++){
        uint8_t currVal = (val >> (8 * (W_DATA - i -1))) & 0xff;
        int row_index = (col*W_DATA+i);
        mem2d(inWaitingMemory, KERNEL_DIM, row_index, KERNEL_DIM - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }

    // If the col is the last on -> start process
    if (col == (MAX_COL - 1)){
        // Shift the waiting memory to the right for skewing
        for (int i = 0; i < KERNEL_DIM; i++) {
            mem2d(inputMemory, KERNEL_DIM, i, 0) = mem2d(inWaitingMemory, KERNEL_DIM, i, KERNEL_DIM - 1);
            for (int j= KERNEL_DIM - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
                mem2d(inWaitingMemory, KERNEL_DIM, i, j) = mem2d(inWaitingMemory, KERNEL_DIM, i, j - 1);
            }
        }

        // Multiply the input to the weight and accumulate to the output
        for (int i= KERNEL_DIM * KERNEL_DIM - 1; i >= 0 ; i--){
            outputMemory[i + KERNEL_DIM] = (inputMemory[i] * weights[i]) + outputMemory[i];
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
            mem2d(outWaitingMemory, KERNEL_DIM, 0, j) = mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j);
        }
    }


    // Return the output
    uint32_t result = 0;
    int result_idx = ((col+1)%MAX_COL) * W_DATA;
//    int result_idx = col * W_DATA;
    for (int i = 0; i < W_DATA; i++) {
        result |=  mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - result_idx - i - 1, result_idx + i ) << (8 * (W_DATA - i - 1));
    }

    return result;

}

