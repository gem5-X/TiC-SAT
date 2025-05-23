//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"
#include "../transformer.h"

// A base constructor for the TransformerBlock class.
// It initializes the objects pre_seq_len, input_dim, head_hidden_size, num_heads, ff_size

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size) {
    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    ff_size_ = ff_size;
    addNorm = new AddNormalize(pre_seq_len, input_dim, KERNEL_DIM, MAX_ACT_COL);
}

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t **weightVector,
                                   uint32_t **flagVector,
                                   uint32_t *hidden_flag, Format sparseFormatQVK, Format sparseFormatCondense, Format sparseFormatFF0, Format sparseFormatFF1)
        : TransformerBlock(pre_seq_len, input_dim, head_hidden_size,
                           num_heads, ff_size) {

    // This constructor is for the cases where sparseFormat is not CSC or CSR.

    for (int n = 0; n < num_heads; n++) {
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector + n * 3,
                                              flagVector + n * 3, hidden_flag, sparseFormatQVK);
    }

    condense = new Dense(pre_seq_len, num_heads * head_hidden_size, input_dim, weightVector[num_heads * 3],
                         flagVector[num_heads * 3], hidden_flag, sparseFormatCondense);

    feedForward0 = new Dense(pre_seq_len, input_dim, ff_size, weightVector[num_heads * 3 + 1],
                             flagVector[num_heads * 3 + 1], hidden_flag, sparseFormatFF0);
    feedForward1 = new Dense(pre_seq_len, ff_size, input_dim, weightVector[num_heads * 3 + 2],
                             flagVector[num_heads * 3 + 2], hidden_flag, sparseFormatFF1);
}

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t **weightVector,
                                   int **col_ptr, int **row_ptr, uint32_t ***values, Format sparseFormat)
        : TransformerBlock(pre_seq_len, input_dim, head_hidden_size,
                           num_heads, ff_size) {

    // This constructor is for the cases where sparseFormat is CSC or CSR.

    for (int n = 0; n < num_heads; n++) {
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector + n * 3,
                                              col_ptr + n * 3, row_ptr + n * 3, values + n * 3, sparseFormat);
    }

    condense = new Dense(pre_seq_len, num_heads * head_hidden_size, input_dim, weightVector[num_heads * 3],
                         col_ptr[num_heads * 3], row_ptr[num_heads * 3], values[num_heads * 3], sparseFormat);

    feedForward0 = new Dense(pre_seq_len, input_dim, ff_size, weightVector[num_heads * 3 + 1],
                             col_ptr[num_heads * 3 + 1], row_ptr[num_heads * 3 + 1], values[num_heads * 3 + 1],
                             sparseFormat);

    feedForward1 = new Dense(pre_seq_len, ff_size, input_dim, weightVector[num_heads * 3 + 2],
                             col_ptr[num_heads * 3 + 2], row_ptr[num_heads * 3 + 2], values[num_heads * 3 + 2],
                             sparseFormat);
}


TransformerBlock::~TransformerBlock() = default;


void TransformerBlock::compute(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t *multihead_out,
                               uint32_t *condense_out, uint32_t *intermediateFF) {

#ifndef DEVELOP
  system("m5 resetstats");
#endif

    for (int n = 0; n < num_heads_; n++) {
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_ >> 2));
    }

    std::cout << "Condense" << std::endl;
    condense->compute(seq_len, multihead_out, condense_out);

    std::cout << "Add Norm" << std::endl;
    addNorm->computeRearranged(input, condense_out);

#ifndef DEVELOP
    system("m5 dumpresetstats");
#endif

    std::cout << "Feed Forward 0" << std::endl;
    feedForward0->compute(seq_len, condense_out, intermediateFF);

    std::cout << "Feed Forward 1" << std::endl;
    feedForward1->compute(seq_len, intermediateFF, output);

    std::cout << "Add Norm" << std::endl;

    std::cout << "Add norm rearranged" << std::endl;
    addNorm->computeRearranged(condense_out, output);

#ifndef DEVELOP
    system("m5 dumpresetstats");
#endif

    print_weight(output, seq_len, input_dim_ >> 2);
    std::cout << std::endl;
    print_weight(output, 1, input_dim_ >> 2);
    // getchar();
}
