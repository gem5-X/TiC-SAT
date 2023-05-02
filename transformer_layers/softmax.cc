#include "softmax.h"
#include <cmath>
#include <iostream>

static const  uint8_t  lookup[32] = {
        4, 5, 7, 8, 11, 14, 18, 23, 30, 38, 49, 63, 80, 103, 132, 170, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3,
};

Softmax::Softmax()= default;

Softmax::~Softmax()= default;

void Softmax::compute(uint32_t *input, std::size_t seq_len){
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = (uint8_t*) (input + i * (seq_len >> 2));

        for (int j=0; j< seq_len; j++){
            *(input_uptr) = lookup[(* (uint8_t *) input_uptr) >> 3]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
            sum += *(input_uptr);
            input_uptr ++;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = (uint8_t*) (input + i * (seq_len >> 2));
        for (int j=0; j< seq_len; j++){
//            std::cout << "Ptr " << i << "\t: " << (int) *(input_ptr)  << std::endl;
            *(input_uptr) = (uint8_t) ((*(input_uptr)) /(sum >> 8)); // divide the sum by 256 otherwise all the outputs will be 0!
//            std::cout << "LUT " << i << "\t: " << (int) *(input_ptr) << std::endl;
            input_uptr ++;
        }
//        input_uptr = (uint8_t*) (input + i * (seq_len >> 2));
//        int sum_softmax = 0;
//        for (int j=0; j< seq_len; j++)
//            sum_softmax += *(input_uptr++);
    }
}

void Softmax::computeRearranged(uint32_t *input, std::size_t seq_len, std::size_t kernelDim) {
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = ((uint8_t*) input) + i * kernelDim;
        for (int j =0; j< seq_len / kernelDim; j++){
            for (int k=0; k< kernelDim; k++) {
                *(input_uptr+k) = lookup[(* (uint8_t *) (input_uptr+ k)) >> 3]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
                sum += *(input_uptr+k);
            }
            input_uptr += seq_len* kernelDim;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = ((uint8_t*) input) + i * kernelDim;
        for (int j =0; j< seq_len / kernelDim; j++){
            for (int k=0; k< kernelDim; k++) {
                *(input_uptr+k) = (uint8_t) ((*(input_uptr+k)) /(sum >> 8));
            }
            input_uptr += seq_len* kernelDim;
        }
    }
}

void Softmax::post_softmax(uint32_t *input, std::size_t seq_len, std::size_t headSize){
    auto* input_ptr = (int8_t*) input;
    for (int i =0; i< seq_len * headSize; i++){
        *input_ptr = (int8_t) (*(input_ptr) >> 6);
        input_ptr++;
    }
}


void Softmax::computeFloat(uint32_t *in_matrix, std::size_t seq_len) {
    int32_t fractional_bits = 16;

    // Split the input uint32_t array into int8_t values
    std::vector<int8_t> input_int8(seq_len);
    for (std::size_t i = 0; i < seq_len / 4; ++i) {
        uint32_t value = in_matrix[i];
        for (int j = 0; j < 4; ++j) {
            input_int8[i * 4 + j] = static_cast<int8_t>((value >> (8 * j)) & 0xFF);
        }
    }

    // Convert int8_t values into fixed-point representation
    std::vector<int32_t> input_fixed(seq_len);
    for (std::size_t i = 0; i < seq_len; ++i) {
        input_fixed[i] = float_to_fixed(static_cast<float>(input_int8[i]), fractional_bits);
    }

    // Calculate the softmax for each chunk using fixed-point arithmetic
    softmax_fixed(input_fixed, fractional_bits);

    // (Optional) Convert the fixed-point result back to floating-point representation
    std::vector<float> result(seq_len);
    for (std::size_t i = 0; i < seq_len; ++i) {
        result[i] = fixed_to_float(input_fixed[i], fractional_bits);
        std::cout << "result[" << i << "]: " << result[i] << std::endl;
    }
}

int32_t Softmax::float_to_fixed(float value, int32_t fractional_bits) {
    return static_cast<int32_t>(round(value * (1 << fractional_bits)));
}

float Softmax::fixed_to_float(int32_t fixed_value, int32_t fractional_bits) {
    return static_cast<float>(fixed_value) / (1 << fractional_bits);
}

void Softmax::softmax_fixed(std::vector<int32_t> &input, int32_t fractional_bits) {
    std::size_t chunk_size = input.size() / 4;

    for (std::size_t i = 0; i < 4; ++i) {
        int32_t max_val = input[i * chunk_size];

        // Find the maximum value in the chunk
        for (std::size_t j = i * chunk_size + 1; j < (i + 1) * chunk_size; ++j) {
            if (input[j] > max_val) {
                max_val = input[j];
            }
        }

        int32_t sum_exp = 0;
        for (std::size_t j = i * chunk_size; j < (i + 1) * chunk_size; ++j) {
            // Subtract max_val and exponentiate using fixed-point arithmetic
            int32_t exp_val = float_to_fixed(exp(fixed_to_float(input[j] - max_val, fractional_bits)), fractional_bits);
            input[j] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize the values in the chunk by dividing by the sum of exponentiated values
        for (std::size_t j = i * chunk_size; j < (i + 1) * chunk_size; ++j) {
            input[j] = (input[j] << fractional_bits) / sum_exp;
        }
    }
}

