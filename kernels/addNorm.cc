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
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) (*output_ptr / (sum >> 2));
        }

    }
}
