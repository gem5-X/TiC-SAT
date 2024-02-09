#include "dense.h"
#include <exception>
//#include <mkl.h>
#include <memory.h>
#include <iostream>

Dense::Dense(std::size_t input_size, std::size_t output_size, uint32_t *weightDense, uint32_t *flagDense,
             const uint32_t *hidden_flag, Format format) {
    input_size_ = input_size;
    output_size_ = output_size;
    std::cout << "Input Size : " << input_size_ << std::endl;
    std::cout << "Output Size : " << output_size_ << std::endl;
    weight = weightDense;
    flag = flagDense;
    bias = nullptr;
    hidden_flag_ = hidden_flag;
    format_ = format;
}

Dense::Dense(std::size_t input_size, std::size_t output_size, uint32_t *weightDense,
             int *col_ptr, int *row_ptr, uint32_t **values, Format format) {
    input_size_ = input_size;
    output_size_ = output_size;
    std::cout << "Input Size : " << input_size_ << std::endl;
    std::cout << "Output Size : " << output_size_ << std::endl;
    weight = weightDense;
    bias = nullptr;
    col_ptr_ = col_ptr;
    row_ptr_ = row_ptr;
    values_ = values;
    format_ = format;
}

Dense::~Dense() {
}

void Dense::multiplyweight(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    auto *sparseMatrixMultiplier = new SparseMatrixMultiplier(input_size_, output_size_,
                                                              seq_len, format_);
    if (format_ == Format::WITH_FLAG || format_ == Format::DYNAMIC ||
        format_ == Format::NON_PRUNED) {
        sparseMatrixMultiplier->compute(input, output, (const int *) flag, nullptr, weight);
    } else if (format_ == Format::HIDDEN_KEY) {
        sparseMatrixMultiplier->compute(input, output, (const int *) hidden_flag_, nullptr, weight);
    } else if (format_ == Format::META_DATA) {
        sparseMatrixMultiplier->compute(input, output,(const int *) flag, (const int *) hidden_flag_, weight);
    } else if (format_ == Format::CSC || format_ == Format::CSR ) {
        sparseMatrixMultiplier->compute( input, output, (const int *)col_ptr_, (const int *)row_ptr_,
                                        (const uint32_t**) values_);
    } else {

            throw std::runtime_error("Unsupported format");
    }
//    smmComputeRearranged(seq_len, input, output, weight, flag, input_size_, output_size_, true, hidden_flag_);
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
