#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"
#include "../accelerator/sparseMatrixMultiplication.h"

class Dense {
public:
    Dense(std::size_t input_dim, std::size_t output_dim,
          uint32_t *weight, uint32_t *flag, const uint32_t* hidden_flag, Format format);

    ~Dense();

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output);

private:
    void multiplyweight(std::size_t seq_len, uint32_t *input, uint32_t *output);

    void addbias(std::size_t seq_len, uint32_t *output);

    std::size_t input_size_;
    std::size_t output_size_;
    uint32_t *weight; // shape [input_size_, output_size_]
    uint32_t *flag; // shape [input_size_/KERNEL_DIM, output_size_/KERNEL_DIM/32]
    uint32_t *bias;   // shape [output_size_]
    const uint32_t *hidden_flag_;
    Format format_;

};