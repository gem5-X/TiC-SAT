//
// Created by alireza on 12/19/23.
//

#ifndef FVLLMONTITRANSFORMER_SPARSEMATRIXMULTIPLICATION_H
#define FVLLMONTITRANSFORMER_SPARSEMATRIXMULTIPLICATION_H
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <cstring>

#define KERNEL_DIM SA_SIZE
#define MAX_COL (KERNEL_DIM/4)

enum Format {
    CSR,
    CSC,
    WITH_FLAG,
    HIDDEN_KEY,
    DYNAMIC,
    NON_PRUNED,
    META_DATA,
    INTERLEAVED,
};

class SparseMatrixMultiplier {
private:
    std::size_t input_size_;
    std::size_t output_size_;
    int seq_len;

    Format format_;

    void processMultiplication(uint32_t *input, uint32_t *output, int row, int col, const uint32_t *values);
    void computeCSR(uint32_t *input, uint32_t *output, const int *row_ptr, const int *col_ind, const uint32_t **values);
    void computeCSC(uint32_t *input, uint32_t *output, const int *col_ptr, const int *row_ind, const uint32_t **values);
    void computeMetaData(uint32_t *input, uint32_t *output, const int* m1, const int* m2, const uint32_t *values);
    void computeInterleavedMetaData(uint32_t *input, uint32_t *output, const uint32_t *values);
    void computeWithFlag(uint32_t *input, uint32_t *output, uint32_t *flag, const uint32_t *values);
    void computeHiddenKey(uint32_t *input, uint32_t *output, const uint32_t *hiddenKey, const uint32_t *values);
    void computeDynamic(uint32_t *input, uint32_t *output, const uint32_t *values);
    void computeNonPruned(uint32_t *input, uint32_t *output, const uint32_t *values);

public:
    SparseMatrixMultiplier(std::size_t input_size_,
                           std::size_t output_size_, int seq_len, Format format);

    void compute(uint32_t *, uint32_t *, const int *, const int *, const uint32_t **values);
    void compute(uint32_t *, uint32_t *, const int *, const int *, const uint32_t *values);
};


#endif //FVLLMONTITRANSFORMER_SPARSEMATRIXMULTIPLICATION_H
