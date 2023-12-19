//
// Created by alireza on 12/19/23.
//

#ifndef FVLLMONTITRANSFORMER_SPARSEMATRIXMULTIPLICATION_H
#define FVLLMONTITRANSFORMER_SPARSEMATRIXMULTIPLICATION_H
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <cstring>
#include "smm_gem.h"

class SparseMatrixMultiplier {
private:
    uint32_t *input;
    uint32_t *output;
    std::size_t input_size_;
    std::size_t output_size_;
    int seq_len;

    int KERNEL_DIM;
    int W_DATA;
    int MAX_COL;

    void processMultiplication(int row, int col, const uint32_t *values);

public:
    SparseMatrixMultiplier(uint32_t *input, uint32_t *output, std::size_t input_size_,
                           std::size_t output_size_, int seq_len, int KERNEL_DIM, int W_DATA, int MAX_COL);

    void computeCSR(const int *row_ptr, const int *col_ind, const uint32_t **values);
    void computeCSC(const int *col_ptr, const int *row_ind, const uint32_t **values);
};


#endif //FVLLMONTITRANSFORMER_SPARSEMATRIXMULTIPLICATION_H
