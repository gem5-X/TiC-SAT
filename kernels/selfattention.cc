#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>

SingleHeadSelfAttn::SingleHeadSelfAttn(std::vector<std::string> names, std::size_t pre_seq_len, std::size_t num_heads,
                                       std::size_t input_dim, std::size_t head_hidden_size,
                                       std::vector<uint32_t *> weightVector) {

    pre_seq_len_ = pre_seq_len;
    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;

    auto startit = names.begin();

    std::vector<std::string> query_names(startit, startit + 2);
    query_layer = new Dense(query_names, input_dim, head_hidden_size, weightVector[0]);
    startit += 2;
    std::vector<std::string> key_names(startit, startit + 2);
    key_layer = new Dense(query_names, input_dim, head_hidden_size, weightVector[1]);
    startit += 2;
    std::vector<std::string> value_names(startit, startit + 2);
    value_layer = new Dense(query_names, input_dim, head_hidden_size, weightVector[2]);
    //        softmax = new Softmax<uint32_t>();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len * num_heads >> 2];
//
//        q_array = new const uint32_t *[pre_batch_size * num_heads];
//        k_array = new const uint32_t *[pre_batch_size * num_heads];
//        pointer_qk_array = new uint32_t* [pre_batch_size * num_heads];
//
//        sim_array = new const uint32_t *[pre_batch_size * num_heads];
//        value_array = new const uint32_t *[pre_batch_size * num_heads];
//        pointer_sv_array = new uint32_t *[pre_batch_size * num_heads];

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
    Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_, pre_seq_len_);
    MatMulSystolic::compute(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                            seq_len);
    softmax->compute(attention_scores, seq_len);
    MatMulSystolic::compute(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
}