//
// Created by alireza on 12/19/23.
//
#include "smm_gem.h"
#include "sparseMatrixMultiplication.h"

SparseMatrixMultiplier::SparseMatrixMultiplier(std::size_t input_size_, std::size_t output_size_, int seq_len, Format format){
    this->input_size_ = input_size_;
    this->output_size_ = output_size_;
    this->seq_len = seq_len;
    this->format_ = format;
}

void SparseMatrixMultiplier::processMultiplication(uint32_t *input, uint32_t *output, int row, int col, const uint32_t *values) {
    int rowBlockSize = KERNEL_DIM;
    int colBlockSize = MAX_COL;
    int W_DATA = KERNEL_DIM/ MAX_COL;
    uint32_t *inPtr;
    uint32_t *outPtr;

    // std::cout << "Weights at row: " << std::dec << row << " col: " << col << " are: " << std::endl;
    for (int k = 0; k < rowBlockSize * colBlockSize; k++) {
        uint32_t weight = values[k];
        // std::cout << std::hex << weight << " ";
        smmParamWrite(k * W_DATA, weight, 0);
    }
    // std::cout << std::endl;

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

void SparseMatrixMultiplier::computeCSR(uint32_t *input, uint32_t *output,
                                        const int *col_ind, const int *row_ptr, const uint32_t **values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;

    for (int i=0; i< row_in_w; i++){
        for (int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            int row = i;
            int col = col_ind[j];
            processMultiplication(input, output, row, col, values[j]);
        }
    }
}

void SparseMatrixMultiplier::computeCSC(uint32_t *input, uint32_t *output,
                                        const int *col_ptr, const int *row_ind, const uint32_t **values) {
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    for (int i=0; i< col_in_w; i++){
        for (int j=col_ptr[i]; j<col_ptr[i+1]; j++){
            int row = row_ind[j];
            int col = i;
            processMultiplication(input, output, row, col, values[j]);
        }
    }
}

void SparseMatrixMultiplier::computeMetaData(uint32_t *input, uint32_t *output,
                                             const int* m1, const int* m2, const uint32_t *values){
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    for (int i=0; i< col_in_w; i++){
        if (*m1 ++ == 0) {
            continue;
        }
        for (int j=0; j<row_in_w; j++){
            if (*m2++ == 0) {
                continue;
            }
            int row = j;
            int col = i;
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}

void SparseMatrixMultiplier::computeInterleavedMetaData(uint32_t *input, uint32_t *output,
                                                        const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    // In this first implementation, BLOCK_SIZE is one column of tiles = n_row / KERNEL_DIM = row_in_w
    int BLOCK_SIZE = row_in_w;
    int counter32 = 0;
    int metadata_block_size = (BLOCK_SIZE + 32 - 1) / 32;
    int metadata_offset = 32 - std::min(32, row_in_w);  // Offset in case the metadata is not aligned to 32 bits

    std::cout << "Interleaved Meta Data" << std::endl;

    for (int i=0; i<col_in_w; i++){
        // std::cout << "column: " << std::dec << i << std::endl;
        uint32_t* next_block = ((uint32_t*) values) + *values;    // Compute when the next block starts
        // std::cout << "offset: " << std::dec << *values << std::endl;
        values++;
        uint32_t* metadata;
        metadata = (uint32_t*) values;                // Read the metadata
        counter32 = metadata_offset;
        values += metadata_block_size;      // Skip the metadata
        int row = 0;

        while (values < next_block) {
            if (counter32 == 32) {
                counter32 = 0;
                metadata++;
            }
            if (!(*metadata & (0x00000001 << counter32++))) {
                row++;
                continue;
            }
            int col = i;
            processMultiplication(input, output, row++, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}

void SparseMatrixMultiplier::computeWithFlag(uint32_t *input, uint32_t *output,
                                             uint32_t *flag, const uint32_t *values) {
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
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}


void SparseMatrixMultiplier::computeHiddenKey(uint32_t *input, uint32_t *output,
                                              const uint32_t *hiddenKey, const uint32_t *values) {
    std::cout << "Hidden key" << std::endl;
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
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}


void SparseMatrixMultiplier::computeDynamic(uint32_t *input, uint32_t *output,
                                            const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            int row = j;
            int col = i;
            bool zero_tile = true;
            for (int k = 0; k < KERNEL_DIM * MAX_COL; k++) {
                if (values[k] != 0x0) { // check if the tile is zero
                    zero_tile = false;
                    break;  // if not, break
                }
            }
            if (zero_tile) {
                values += KERNEL_DIM * MAX_COL; // skip the tile
                continue;
            }

            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}

void SparseMatrixMultiplier::computeNonPruned(uint32_t *input, uint32_t *output,
                                              const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            int row = j;
            int col = i;
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_COL;
        }
    }
}


void SparseMatrixMultiplier::compute(uint32_t *input, uint32_t *output,
                                     const int *col_ptr, const int *row_ptr, const uint32_t **values) {
    if (this->format_ == Format::CSR) {
        std::cout<<"CSR"<<std::endl;
        system("m5 resetstats");
        computeCSR(input, output, col_ptr, row_ptr, values);
        system("m5 dumpresetstats");
    } else if (this->format_ == Format::CSC) {
        std::cout<<"CSC"<<std::endl;
        system("m5 resetstats");
        computeCSC(input, output, col_ptr, row_ptr, values);
        system("m5 dumpresetstats");
    }
}

void SparseMatrixMultiplier::compute(uint32_t *input, uint32_t *output,
                                     const int *row_ptr, const int *col_ind, const uint32_t *values) {
    if (this->format_ == Format::META_DATA) {
        std::cout<<"META_DATA"<<std::endl;
        system("m5 resetstats");
        computeMetaData(input, output, row_ptr, col_ind, values);
        system("m5 dumpresetstats");
    } else if (this->format_ == Format::INTERLEAVED) {
        std::cout<<"INTERLEAVED"<<std::endl;
        system("m5 resetstats");
        computeInterleavedMetaData(input, output, values);
        system("m5 dumpresetstats");
    } else if (this->format_ == Format::WITH_FLAG) {
        std::cout<<"WITH_FLAG"<<std::endl;
        system("m5 resetstats");
        computeWithFlag(input, output, (uint32_t *) row_ptr, values);
        system("m5 dumpresetstats");
    } else if (this->format_ == Format::HIDDEN_KEY) {
        std::cout<<"HIDDEN_KEY"<<std::endl;
        system("m5 resetstats");
        computeHiddenKey(input, output, (uint32_t*)row_ptr, values);
        system("m5 dumpresetstats");
    } else if (this->format_ == Format::DYNAMIC) {
        std::cout<<"DYNAMIC"<<std::endl;
        system("m5 resetstats");
        computeDynamic(input, output, values);
        system("m5 dumpresetstats");
    } else if (this->format_ == Format::NON_PRUNED) {
        std::cout<<"NON_PRUNED"<<std::endl;
        system("m5 resetstats");
        computeNonPruned(input, output, values);
        system("m5 dumpresetstats");
    }
}


