#include "softmax.h"
#include <cmath>
#include <iostream>

// TODO: division by the square root of the hidden size (d_q) are hardcoded for a hidden size of 64
// TODO: also, lookup table is hardcoded for 8-bit quantization divided by 8
// TODO: temporal fix to fit lookup is to divide by a value so that lookup is indexed between 0 and 31

static const  u_activation_t lookup[32] = {
        4, 5, 7, 8, 11, 14, 18, 23, 30, 38, 49, 63, 80, 103, 132, 170, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3,
};

Softmax::Softmax()= default;

Softmax::~Softmax()= default;

#if ACTIVATION_FP == 1
void Softmax::compute(uint32_t *input, std::size_t seq_len){
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        arith_activation_t in_aux;
        float sum = 0.0;
        auto* input_uptr = (u_activation_t*) (input + i * (seq_len / ACT_PER_BUS));

        for (int j=0; j< seq_len; j++){
            in_aux.bin = *input_uptr;
            in_aux.fp = exp(in_aux.fp) / 8; // divide by the sqrt od the d_q which is sqrt(64) -> 8
            sum += in_aux.fp;
            *input_uptr = in_aux.bin;
            input_uptr ++;
        }
        sum = (sum==0.0) ? sum + 1 : sum;
        input_uptr = (u_activation_t*) (input + i * (seq_len / ACT_PER_BUS));
        for (int j=0; j< seq_len; j++){
//            std::cout << "Ptr " << i << "\t: " << (int) *(input_ptr)  << std::endl;
            in_aux.bin = *input_uptr;
            in_aux.fp = in_aux.fp / ((sum / 256) + 1); // divide the sum by 256 otherwise all the outputs will be 0!
            *input_uptr = in_aux.bin;
//            std::cout << "LUT " << i << "\t: " << (int) *(input_ptr) << std::endl;
            input_uptr ++;
        }
//        input_uptr = (uint8_t*) (input + i * (seq_len / ACT_PER_BUS));
//        int sum_softmax = 0;
//        for (int j=0; j< seq_len; j++)
//            sum_softmax += *(input_uptr++);
    }
}

void Softmax::computeRearranged(uint32_t *input, std::size_t seq_len) {
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        arith_activation_t in_aux;
        float sum = 0.0;
        auto* input_uptr = ((u_activation_t*) input) + i * KERNEL_DIM;
        for (int j =0; j< seq_len / KERNEL_DIM; j++){
            for (int k=0; k< KERNEL_DIM; k++) {
                in_aux.bin = *(input_uptr+k);
                in_aux.fp = exp(in_aux.fp) / 8; // divide by the sqrt od the d_q which is sqrt(64) -> 8
                sum += in_aux.fp;
                *(input_uptr+k) = in_aux.bin;
            }
            input_uptr += seq_len* KERNEL_DIM;
        }
        sum = (sum==0.0) ? sum + 1 : sum;
        input_uptr = ((u_activation_t*) input) + i * KERNEL_DIM;
        for (int j =0; j< seq_len / KERNEL_DIM; j++){
            for (int k=0; k< KERNEL_DIM; k++) {
                in_aux.bin = *(input_uptr+k);
                in_aux.fp = in_aux.fp / ((sum / 256) + 1);
                *(input_uptr+k) = in_aux.bin;
            }
            input_uptr += seq_len* KERNEL_DIM;
        }
    }
}

void Softmax::post_softmax(uint32_t *input, std::size_t seq_len, std::size_t headSize){
    auto* input_ptr = (activation_t*) input;
    arith_activation_t in_aux;
    for (int i =0; i< seq_len * headSize; i++){
        in_aux.fp = *input_ptr;
        in_aux.fp = in_aux.fp / 64;
        *input_ptr = in_aux.bin;
        input_ptr++;
    }
}
#else   // ACTIVATION_FP == 0
void Softmax::compute(uint32_t *input, std::size_t seq_len){
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = (u_activation_t*) (input + i * (seq_len / ACT_PER_BUS));

        for (int j=0; j< seq_len; j++){
            *(input_uptr) = lookup[uint((* (u_activation_t *) input_uptr) >> (ACTIVATION_BITS - 5))]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
            sum += *(input_uptr);
            input_uptr ++;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = (u_activation_t*) (input + i * (seq_len / ACT_PER_BUS));
        for (int j=0; j< seq_len; j++){
//            std::cout << "Ptr " << i << "\t: " << (int) *(input_ptr)  << std::endl;
            *(input_uptr) = (u_activation_t) ((*(input_uptr)) /((sum >> 8)+1)); // divide the sum by 256 otherwise all the outputs will be 0!
//            std::cout << "LUT " << i << "\t: " << (int) *(input_ptr) << std::endl;
            input_uptr ++;
        }
//        input_uptr = (uint8_t*) (input + i * (seq_len / ACT_PER_BUS));
//        int sum_softmax = 0;
//        for (int j=0; j< seq_len; j++)
//            sum_softmax += *(input_uptr++);
    }
}

void Softmax::computeRearranged(uint32_t *input, std::size_t seq_len) {
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = ((u_activation_t*) input) + i * KERNEL_DIM;
        for (int j =0; j< seq_len / KERNEL_DIM; j++){
            for (int k=0; k< KERNEL_DIM; k++) {
                *(input_uptr+k) = lookup[(* (u_activation_t *) (input_uptr+ k)) >> (ACTIVATION_BITS - 5)]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
                sum += *(input_uptr+k);
            }
            input_uptr += seq_len* KERNEL_DIM;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = ((u_activation_t*) input) + i * KERNEL_DIM;
        for (int j =0; j< seq_len / KERNEL_DIM; j++){
            for (int k=0; k< KERNEL_DIM; k++) {
                *(input_uptr+k) = (u_activation_t) ((*(input_uptr+k)) /((sum >> 8)+1));
            }
            input_uptr += seq_len* KERNEL_DIM;
        }
    }
}

void Softmax::post_softmax(uint32_t *input, std::size_t seq_len, std::size_t headSize){
    auto* input_ptr = (activation_t*) input;
    for (int i =0; i< seq_len * headSize; i++){
        *input_ptr = (activation_t) (*(input_ptr) / 64);
        input_ptr++;
    }
}
#endif  // ACTIVATION_FP

void Softmax::computeFloat(uint32_t *in_matrix, std::size_t seq_len) {
    int32_t fractional_bits = 16;

    // Split the input uint32_t array into int8_t values
    std::vector<activation_t> input_act(seq_len);
    for (std::size_t i = 0; i < seq_len / ACT_PER_BUS; ++i) {
        uint32_t value = in_matrix[i];
        for (int j = 0; j < ACT_PER_BUS; ++j) {
            input_act[i * ACT_PER_BUS + j] = static_cast<activation_t>((value >> (ACTIVATION_BITS * j)) & ACTIVATION_MASK);
        }
    }

    // Convert int8_t values into fixed-point representation
    std::vector<int32_t> input_fixed(seq_len);
    for (std::size_t i = 0; i < seq_len; ++i) {
        input_fixed[i] = float_to_fixed(static_cast<float>(input_act[i]), fractional_bits);
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
    std::size_t chunk_size = input.size() / ACT_PER_BUS;

    for (std::size_t i = 0; i < ACT_PER_BUS; ++i) {
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

