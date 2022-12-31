//
// Created by alireza on 3/2/22.
//

#include "addNorm.h"
#include <cmath>

AddNormalize::AddNormalize(std::size_t seq_len, std::size_t input_dim) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
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

        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) ((*output_ptr - mean) * (sd_inv) >> 2);
            output_ptr ++;
        }

    }
}
