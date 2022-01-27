//
// Created by alireza on 1/13/22.
//

#include <cstdint>
#include "iostream"
#include "systolic_m2m.h"

SystolicMatrixMultiplication::SystolicMatrixMultiplication() = default;

void SystolicMatrixMultiplication::loadWeights(int row, int col, uint32_t val) {
    int idx= row * W_DIM + col* W_DATA;
    for (int i=0; i < W_DATA; i++){
        uint8_t currVal = ((val >> (8 * (W_DATA -i-1))) & 0xff);
        weights[idx + i] = currVal;
    }
}

void SystolicMatrixMultiplication::printWeights() {
    for (unsigned char data : inputMemory)
        std::cout << std::hex << (uint32_t) data << std::endl;
}

uint32_t SystolicMatrixMultiplication::streamInOut(int col, uint32_t val) {
    // Split the input to an array
    for (int i=0; i < W_DATA; i++){
        uint8_t currVal = (val >> (8 * (W_DATA - i -1))) & 0xff;
        int row_index = (col*W_DATA+i);
        mem2d(waitingMemory, W_DIM, row_index, W_DIM-row_index-1) = currVal; // off-diagonal of the waiting memory
    }

    // If the col is the last on -> start process
    if (col == (MAX_COL - 1)){
        // Shift the waiting memory to the right for skewing
        for (int i = 0; i < W_DIM; i++) {
            mem2d(inputMemory, W_DIM, i, 0) = mem2d(waitingMemory, W_DIM, i, W_DIM-1);
            for (int j=W_DIM-1; j>0; j--){ // TODO: shift only the right-hand triangle
                mem2d(waitingMemory, W_DIM, i, j) = mem2d(waitingMemory, W_DIM, i, j-1);
            }
        }

        // Multiply the input to the weight and accumulate to the output
        for (int i=W_DIM*W_DIM-1; i>=0 ; i--){
            outputMemory[i+W_DIM] = (inputMemory[i] * weights[i]) + outputMemory[i];
        }

        // Shift the input memory to the right
        for (int i = 0; i < W_DIM; i++) {
            for (int j=W_DIM-1; j>0; j--){
                inputMemory[i*W_DIM + j] = inputMemory[i*W_DIM + j-1];
            }
        }
    }

    // Return the output
    uint32_t result = 0;
    int result_idx = W_DIM*W_DIM + ((col+1)%MAX_COL) * W_DATA;
    for (int i = 0; i < W_DATA; i++) {
        result |=  outputMemory[result_idx + i]<< (8 * i);
    }

    return result;

}

