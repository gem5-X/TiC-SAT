#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
//#include <cstdint>
#include "debuggerFunctions.h"

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t **weightVector, uint32_t **flagVector, uint32_t* hidden_flag) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    hidden_flag_ = hidden_flag;

    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0], flagVector[0], hidden_flag);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1], flagVector[1], hidden_flag);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2], flagVector[2], hidden_flag);
    softmax = new Softmax();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len >> 2]();
}

SingleHeadSelfAttn::~SingleHeadSelfAttn() {

    delete[] query_layer_out;
    delete[] key_layer_out;
    delete[] key_transposed_layer_out;
    delete[] value_layer_out;
    delete[] attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;
}

void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    query_layer->compute(seq_len, input, query_layer_out);
    key_layer->compute(seq_len, input, key_layer_out);
    value_layer->compute(seq_len, input, value_layer_out);


    std::cout << "Rearranged method" << std::endl;
    Transpose::transpose_rearranged(key_layer_out, key_transposed_layer_out, head_hidden_size_,
                                    pre_seq_len_);

    auto* sparseMatrixMultiplier = new SparseMatrixMultiplier(query_layer_out, attention_scores,
                                                               head_hidden_size_, seq_len, seq_len,
                                                               Format::NON_PRUNED
                                                               );
    sparseMatrixMultiplier->compute(nullptr, nullptr, key_transposed_layer_out);
//    smmComputeRearranged(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, nullptr, head_hidden_size_,
//                            seq_len, false, hidden_flag_);
    softmax->computeRearranged(attention_scores, seq_len);
    sparseMatrixMultiplier = new SparseMatrixMultiplier(attention_scores, output,
                                                        seq_len, head_hidden_size_, seq_len,
                                                        Format::NON_PRUNED
                                                        );
    sparseMatrixMultiplier->compute(nullptr, nullptr, value_layer_out);
//    smmComputeRearranged(seq_len, attention_scores, output, value_layer_out, nullptr, seq_len, head_hidden_size_,
//                         false, hidden_flag_);

    softmax->post_softmax(output, seq_len, head_hidden_size_);
}