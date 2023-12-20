//
// Created by alireza on 12/19/23.
//

#include "sparseMatrixMultiplication.h"
SparseMatrixMultiplier::SparseMatrixMultiplier(uint32_t *input, uint32_t *output, std::size_t input_size_,
                                               std::size_t output_size_, int seq_len, int KERNEL_DIM,
                                               int MAX_COL, Format format){
    this->input = input;
    this->output = output;
    this->input_size_ = input_size_;
    this->output_size_ = output_size_;
    this->seq_len = seq_len;
    this->KERNEL_DIM = KERNEL_DIM;
    this->MAX_COL = MAX_COL; // MAX_COL = KERNEL_DIM / W_DATA
    this->format_ = format;
}

void SparseMatrixMultiplier::processMultiplication(int row, int col, const uint32_t *values) {
    int rowBlockSize = KERNEL_DIM;
    int colBlockSize = MAX_COL;
    int W_DATA = KERNEL_DIM/ MAX_COL;
    uint32_t *inPtr;
    uint32_t *outPtr;

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

void SparseMatrixMultiplier::computeMetaData(const bool* m1, const bool* m2, const uint32_t *values){
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    for (int i=0; i< col_in_w; i++){
        if (!m1[i]) continue;
        for (int j=0; j<row_in_w; j++){
            if (!m2[j]) continue;
            int row = j;
            int col = i;
            processMultiplication(row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}

void SparseMatrixMultiplier::computeDense(uint32_t *flag, const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    int counter = 0;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            if (counter == 32) {
                counter = 0;
                flag++;
            }
            if (*flag & (0x80000000 >> counter++)) {
                continue;
            }
            int row = j;
            int col = i;
            processMultiplication(row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}


void SparseMatrixMultiplier::computeHiddenKey(const uint32_t *hiddenKey, const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            if (*hiddenKey  == *values){
                values++;
                continue;
            }
            int row = j;
            int col = i;
            processMultiplication(row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}


void SparseMatrixMultiplier::computeDynamic(const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            int row = j;
            int col = i;
            bool zero_tile = true;
            for (int k = 0; k < KERNEL_DIM * MAX_COL; i++) {
                if (values[k] != 0x0) { // check if the tile is zero
                    zero_tile = false;
                    break;  // if not, break
                }
            }
            if (zero_tile) {
                values += KERNEL_DIM * MAX_COL; // skip the tile
                continue;
            }

            processMultiplication(row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}

void SparseMatrixMultiplier::computeNonPruned(const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            int row = j;
            int col = i;
            processMultiplication(row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}


void SparseMatrixMultiplier::compute(const int *row_ptr, const int *col_ind, const uint32_t **values) {
    if (this->format_ == Format::CSR) {
        computeCSR(row_ptr, col_ind, values);
    } else if (this->format_ == Format::CSC) {
        computeCSC(row_ptr, col_ind, values);
    }
}

void SparseMatrixMultiplier::compute(const int *row_ptr, const int *col_ind, const uint32_t *values) {
    if (this->format_ == Format::META_DATA) {
        computeMetaData((const bool*)row_ptr, (const bool*)col_ind, values);
    } else if (this->format_ == Format::DENSE) {
        computeDense((uint32_t*)row_ptr, values);
    } else if (this->format_ == Format::HIDDEN_KEY) {
        computeHiddenKey((uint32_t*)row_ptr, values);
    } else if (this->format_ == Format::DYNAMIC) {
        computeDynamic(values);
    } else if (this->format_ == Format::NON_PRUNED) {
        computeNonPruned(values);
    }
}


