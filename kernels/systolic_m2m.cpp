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
        uint8_t currVal = ((val >> (8 * i)) & 0xff);
        weights[idx + i] = currVal;
    }
}

void SystolicMatrixMultiplication::printWeights() {
    std::cout << std::hex << (uint32_t) inputMemory[0] << std::endl;
}

uint32_t SystolicMatrixMultiplication::streamInOut(int col, uint32_t val) {
    // Split the input to an array
    for (int i=0; i < W_DATA; i++){
        uint8_t currVal = (val >> (8 * i)) & 0xff;
        inputMemory[(col*W_DATA+i)*W_DIM] = currVal; // First column of the input memory
    }

    // If the col is the last on -> start process
    if (col == (MAX_COL - 1)){
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

