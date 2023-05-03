#include "dense.h"
#include <exception>
//#include <mkl.h>
#include <memory.h>
#include <iostream>

Dense::Dense(std::size_t input_size, std::size_t  output_size, uint32_t *weightDense, uint32_t *flagDense) {
    input_size_  = input_size;
    output_size_ = output_size;
    std::cout << "Input Size : " << input_size_ << std::endl;
    std::cout << "Output Size : " << output_size_ << std::endl;
    weight = weightDense;
    flag = flagDense;
    bias = nullptr;
}

Dense::~Dense() {
//    delete weight;
//    delete[] bias;
}

void Dense::multiplyweight(std::size_t seq_len, uint32_t *input, uint32_t *output) {
#ifdef REARRANGE
    smmComputeRearranged(seq_len, input, output, weight, flag, input_size_, output_size_, true);
#else
    #ifdef SIMD
        simdCompute(seq_len, input, output, weight, flag, input_size_, output_size_, true);
    #else
        smmCompute(seq_len, input, output, weight, flag, input_size_, output_size_, true);
    #endif
#endif
}

void Dense::addbias(std::size_t seq_len, uint32_t *output) {

    for (std::size_t idx = 0; idx < seq_len; idx++) {
        for (std::size_t feature_idx = 0; feature_idx < output_size_; feature_idx++) {
            output[idx * output_size_ + feature_idx] += bias[feature_idx];
        }
    }
}

void Dense::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    // input shape [batch_size, input_size_]
    // output shape [batch_size, output_size_]

    multiplyweight(seq_len, input, output);
    // add bias vector here
    if (bias != nullptr) {
        addbias(seq_len, output);
    }
}
