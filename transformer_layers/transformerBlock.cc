//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t ** weightVector,
                                   uint32_t ** flagVector,
                                   std::size_t kernelDim, std::size_t maxCol,
                                   uint32_t* hidden_flag, Format sparseFormat) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector+n*3,
                                              flagVector+n*3, hidden_flag, sparseFormat);
    }

    condense = new Dense(num_heads* head_hidden_size, input_dim, weightVector[num_heads * 3],
                         flagVector[num_heads * 3], hidden_flag, sparseFormat);

    multihead_out = new uint32_t[pre_seq_len * num_heads * head_hidden_size >> 2]();
    condense_out = new uint32_t[pre_seq_len * input_dim >> 2]();
    intermediateFF = new uint32_t[pre_seq_len * ff_size >> 2]();

    addNorm = new AddNormalize(pre_seq_len, input_dim, kernelDim, maxCol);
    feedForward0 = new Dense(input_dim, ff_size, weightVector[num_heads * 3+ 1],
                             flagVector[num_heads * 3 + 1], hidden_flag, sparseFormat);
    feedForward1 = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2],
                             flagVector[num_heads * 3 + 2], hidden_flag, sparseFormat);
}

TransformerBlock::~TransformerBlock() = default;


void TransformerBlock::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    system("m5 resetstats");
    for (int n=0; n<num_heads_; n++){
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_ >> 2));
    }

    std::cout << "Condense"  << std::endl;
    condense->compute(seq_len, multihead_out, condense_out);

    std::cout << "Add Norm"  << std::endl;
    addNorm->computeRearranged(input, condense_out);

    system("m5 dumpresetstats");

    std::cout << "Feed Forward 0"  << std::endl;
    feedForward0->compute(seq_len, condense_out, intermediateFF);

    std::cout << "Feed Forward 1"  << std::endl;
    feedForward1->compute(seq_len, intermediateFF, output);

    std::cout << "Add Norm"  << std::endl;

    std::cout << "Add norm rearranged"  << std::endl;
    addNorm->computeRearranged(condense_out, output);
    system("m5 dumpresetstats");
}
