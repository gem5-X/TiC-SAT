#pragma once

#include "util.h"

class Transpose {
public:
    static void transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) ;
    static void transpose_rearranged(uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) ;

    static void multihead_transpose(const uint32_t* input, uint32_t* output, std::size_t seq_len,
                                        std::size_t head_hidden_size, std::size_t num_head);
};