#pragma once

#include "util.h"
//void attn_qk(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, T* query, T* key, T* output, const T** q_array, const T** k_array, T** pointer_qk_array);
//
//void attn_sv(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, T* sim, T* value, T* output, const T** sim_array, const T** value_array, T** pointer_sv_array);
//
//void batchMatMul(std::size_t batch_size, std::size_t seq_len, float* input, float* output, float *weight,
//                 std::size_t input_size_, std::size_t output_size_);
class Transpose {
public:
    static void transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) ;
};