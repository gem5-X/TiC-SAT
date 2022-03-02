//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"


TransformerBlock::TransformerBlock(std::vector<std::string> names, std::size_t pre_seq_len,
                                   std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                                   uint32_t ** weightVector) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttn(names, pre_seq_len, input_dim, head_hidden_size, weightVector+n*3);
    }

    addNorm = new AddNormalize(pre_seq_len, input_dim);

}

TransformerBlock::~TransformerBlock() = default;

void TransformerBlock::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {

    for (int n=0; n<num_heads_; n++){
        selfatten[n]->compute(seq_len, input, output + n * (seq_len*head_hidden_size_ >> 2));
    }

    addNorm->compute(input, output);

}
