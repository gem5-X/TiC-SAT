//
// Created by alireza on 3/2/22.
//

#include "addNorm.h"
#include <cmath>

AddNormalize::AddNormalize(std::size_t seq_len, std::size_t input_dim,
                           std::size_t kernelDim, std::size_t maxCol) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
    kernel_dim_ = kernelDim;
    max_col_ = maxCol;
}

void AddNormalize::compute(uint32_t *input, uint32_t *output) {
    for (int i =0; i< seq_len_; i++){
        auto* input_ptr = (int8_t*) (input + i * (input_dim_ >> 2));
        auto* output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        int32_t sum = 0;
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) (*output_ptr + *input_ptr);
            sum += *output_ptr;
            output_ptr ++;
            input_ptr ++;
        }

        output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        for (int j=0; j< input_dim_; j++){
            variance+= (*output_ptr++ - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
        }
        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) ((*output_ptr - mean) * (sd_inv) >> 2);
            output_ptr ++;
        }

    }
}


void AddNormalize::computeRearranged(uint32_t *input, uint32_t *output) {
    auto* input_ptr = (int8_t*) (input );
    auto* output_ptr = (int8_t*) (output);
    for (int i =0; i< seq_len_* input_dim_; i++){
        *output_ptr = (int8_t) (*output_ptr + *input_ptr);
        output_ptr ++;
        input_ptr ++;
    }

    for (int i=0; i< seq_len_; i++){
        output_ptr = ((int8_t*) output) + i*kernel_dim_;
        int sum = 0;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            output_ptr += seq_len_* kernel_dim_;
            for (int k=0; k< kernel_dim_; k++) {
                sum += *(output_ptr+k);
            }
        }

        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        output_ptr = (int8_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            output_ptr += seq_len_* kernel_dim_;
            for (int k=0; k< kernel_dim_; k++) {
                variance+= (*(output_ptr+k) - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
            }
        }

        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (int8_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            output_ptr += seq_len_* kernel_dim_;
            for (int k=0; k< kernel_dim_; k++) {
                *(output_ptr+k) = (int8_t) ((*(output_ptr+k) - mean) * (sd_inv) >> 2);
            }
        }
    }
}