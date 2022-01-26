//
// Created by alireza on 1/13/22.
//

#include <cstdint>
#include "iostream"
#include "systolic_m2m.h"

SystolicMatrixMultiplication::SystolicMatrixMultiplication() = default;

void SystolicMatrixMultiplication::loadWeights(int row, uint32_t val) {
    for (int i=0; i < W_DIM; i++){
        int idx= row * W_DIM + i;
        uint8_t currVal = ((val >> (8 * i)) & 0xff);
        weights[idx] = currVal;
    }
}

void SystolicMatrixMultiplication::printWeights() {
    std::cout << std::hex << (uint32_t) inputMemory[0] << std::endl;
}

uint32_t SystolicMatrixMultiplication::streamInOut(uint32_t val) {
    // Split the input to an array
    for (int i=0; i < W_DIM; i++){
        uint8_t currVal = (val >> (8 * i)) & 0xff;
        inputMemory[i*W_DIM] = currVal; // First column of the input memory
    }

    // Multiply the input to the weight and accumulate to the output
    for (int i=W_DIM*W_DIM-1; i>=0 ; i--){
        outputMemory[i+W_DIM] = (inputMemory[i] * weights[i]) + outputMemory[i];
    }

    // Return the output
    uint32_t result = 0;
    for (int i = 0; i < W_DIM; i++) {
        result |=  outputMemory[W_DIM*W_DIM + i]<< (8 * i);
    }

    // Shift the input memory to the right
    for (int i = 0; i < W_DIM; i++) {
        for (int j=W_DIM-1; j>0; j--){
            inputMemory[i*W_DIM + j] = inputMemory[i*W_DIM + j-1];
        }
    }

    return result;

}

