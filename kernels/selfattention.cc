#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>

MutiheadselfAttn::MutiheadselfAttn(std::vector<std::string> names, std::size_t pre_seq_len, std::size_t num_heads,
                                   std::size_t input_dim, std::size_t head_hidden_size,
                                   std::vector<uint32_t *> weightVector) {

    pre_seq_len_ = pre_seq_len;
    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;

    auto startit = names.begin();

    for (int head_n=0; head_n < num_heads; head_n ++){
        std::vector<std::string> query_names(startit, startit + 2);
        query_layer = new Dense(query_names, input_dim, head_hidden_size, weightVector[head_n * 3]);
        startit += 2;
        std::vector<std::string> key_names(startit, startit + 2);
        key_layer = new Dense(query_names, input_dim, head_hidden_size, weightVector[head_n * 3+ 1]);
        startit += 2;
        std::vector<std::string> value_names(startit, startit + 2);
        value_layer = new Dense(query_names, input_dim, head_hidden_size, weightVector[head_n * 3 +2]);
        //        softmax = new Softmax<uint32_t>();

        query_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
        key_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
        key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
        value_layer_out = new uint32_t[pre_seq_len * head_hidden_size * num_heads >> 2];
        attention_scores = new uint32_t[pre_seq_len * pre_seq_len * num_heads >> 2];
    }
//
//        q_array = new const uint32_t *[pre_batch_size * num_heads];
//        k_array = new const uint32_t *[pre_batch_size * num_heads];
//        pointer_qk_array = new uint32_t* [pre_batch_size * num_heads];
//
//        sim_array = new const uint32_t *[pre_batch_size * num_heads];
//        value_array = new const uint32_t *[pre_batch_size * num_heads];
//        pointer_sv_array = new uint32_t *[pre_batch_size * num_heads];

}

MutiheadselfAttn::~MutiheadselfAttn() {

    delete query_layer_out;
    delete[] key_layer_out;
    delete[] key_transposed_layer_out;
    delete[] value_layer_out;
    delete[] attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;

//        delete q_array;
//        delete k_array;
//        delete pointer_qk_array;
//
//        delete sim_array;
//        delete value_array;
//        delete pointer_sv_array;
}

void MutiheadselfAttn::compute(std::size_t batch_size, std::size_t seq_len, uint32_t *input, uint64_t *mask,
                               uint32_t *output) {

    query_layer->compute(batch_size, seq_len, input, query_layer_out);
    key_layer->compute(batch_size, seq_len, input, key_layer_out);
    value_layer->compute(batch_size, seq_len, input, value_layer_out);
    Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_, pre_seq_len_);
    MatMulSystolic::compute(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                            seq_len);
    softmax->compute(attention_scores, seq_len);
    MatMulSystolic::compute(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
}