//
// Created by alireza on 3/2/22.
//
#include "selfattention.h"
#include "addNorm.h"

#ifndef FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
#define FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H

class TransformerBlock{
public:
    TransformerBlock(std::vector<std::string> names, std::size_t pre_seq_len,
                     std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     uint32_t ** weightVector);

    virtual ~TransformerBlock();

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output);

private:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    SingleHeadSelfAttn* selfatten[8];
    AddNormalize* addNorm;
};

#endif //FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
