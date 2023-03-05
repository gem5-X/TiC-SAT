#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
//#include <cstdint>

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t ** weightVector) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;

    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0]);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1]);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2]);
    softmax = new Softmax();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2];
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2];
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2];
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2];
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len >> 2];
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
    system("m5 resetstats");
    query_layer->compute(seq_len, input, query_layer_out);
    system("m5 dumpresetstats");
    key_layer->compute(seq_len, input, key_layer_out);
    system("m5 dumpresetstats");
    value_layer->compute(seq_len, input, value_layer_out);
    system("m5 dumpresetstats");
    std::cout<< "Before: " << std::endl;
//    print_weight(key_layer_out, pre_seq_len_ * head_hidden_size_/64, 8* 2);
    Transpose::transpose_rearranged(key_layer_out, key_transposed_layer_out, head_hidden_size_, pre_seq_len_, 8, 2);
    std::cout<< "After: " << std::endl;
//    print_weight(key_transposed_layer_out, head_hidden_size_, pre_seq_len_ >> 2);
//    getchar();
    system("m5 dumpresetstats");
    smmCompute(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                            seq_len, false);
    system("m5 dumpresetstats");
    softmax->compute(attention_scores, seq_len);
    system("m5 dumpresetstats");
    smmCompute(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_, false);
    system("m5 dumpresetstats");
    softmax->post_softmax(output, seq_len);
    system("m5 dumpresetstats");
}