//
// Created by alireza on 12/19/23.
//

#include "sparseMatrixMultiplication.h"
SparseMatrixMultiplier::SparseMatrixMultiplier(uint32_t *input, uint32_t *output, std::size_t input_size_,
                                               std::size_t output_size_, int seq_len, int KERNEL_DIM,
                                               int W_DATA, int MAX_COL){
    this->input = input;
    this->output = output;
    this->input_size_ = input_size_;
    this->output_size_ = output_size_;
    this->seq_len = seq_len;
    this->KERNEL_DIM = KERNEL_DIM;
    this->W_DATA = W_DATA;
    this->MAX_COL = MAX_COL;
}

void SparseMatrixMultiplier::processMultiplication(int row, int col, const uint32_t *values) {
    int rowBlockSize = KERNEL_DIM;
    int colBlockSize = KERNEL_DIM / W_DATA;
    uint32_t *inPtr;
    uint32_t *outPtr;
    uint32_t* weightPtr;

    for (int k = 0; k < rowBlockSize * colBlockSize; k++) {
        uint32_t weight = values[k];
        smmParamWrite(k * W_DATA, weight, 0);
    }

    // Process the multiplication
    int base_col_idx = row * MAX_COL * seq_len;
    outPtr = output + col * MAX_COL * seq_len;
    uint32_t mult;
    inPtr = input + base_col_idx;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MAX_COL; j++) {
            uint32_t content = *(inPtr);
            if (j == MAX_COL - 1) {
                mult = smmStream(*(inPtr++), 0);
            } else {
                mult = smmQueue(j % MAX_COL, *(inPtr++), 0);
            }

            if ((i * MAX_COL + j) >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) {
                // check if the output is valid
                add8in32(*(outPtr++), mult);
            }
        }
    }
    for (int i = seq_len * MAX_COL;
    i < MAX_COL * (seq_len + 2 * KERNEL_DIM - 1) - 1; i++) {
        if ((i % MAX_COL) == MAX_COL - 1) {
            mult = smmStream(0, 0);
        } else {
            mult = smmQueue(i % MAX_COL, 0, 0);
        }
        if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
            add8in32(*(outPtr++), mult);
        }
    }

}

void SparseMatrixMultiplier::computeCSR(const int *row_ptr, const int *col_ind, const uint32_t **values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;

    for (int i=0; i< row_in_w; i++){
        for (int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            int row = i;
            int col = col_ind[j];
            processMultiplication(row, col, values[j]);
        }
    }
}

void SparseMatrixMultiplier::computeCSC(const int *col_ptr, const int *row_ind, const uint32_t **values) {
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    for (int i=0; i< col_in_w; i++){
        for (int j=col_ptr[i]; j<col_ptr[i+1]; j++){
            int row = row_ind[j];
            int col = i;
            processMultiplication(row, col, values[j]);
        }
    }
}
