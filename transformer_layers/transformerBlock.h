//
// Created by alireza on 3/2/22.
//
#include "selfattention.h"
#include "addNorm.h"
#include "dense.h"

#ifndef FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
#define FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H

class TransformerBlock{
public:
    TransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size,
                     Format sparseFormat);

    TransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size,
                     uint32_t **weightVector, uint32_t **flagVector, uint32_t *hidden_flag, Format sparseFormat);

    TransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size,
                     uint32_t **weightVector, int **col_ptr, int **row_ptr, uint32_t ***values, Format sparseFormat);

    virtual ~TransformerBlock();

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t*, uint32_t*, uint32_t*);

private:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t input_dim_;
    std::size_t ff_size_;
    SingleHeadSelfAttn* selfatten[16];
    uint32_t* intermediateFFBlockWise;
    AddNormalize* addNorm;
    Dense* condense;
    Dense* feedForward0;
    Dense* feedForward1;

};

#endif //FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
