#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
//#include <cstdint>
#include "debuggerFunctions.h"

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t **weightVector, std::size_t kernel_dim, std::size_t max_col) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    kernel_size_ = kernel_dim;
    max_col_ = max_col;

    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0]);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1]);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2]);
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


#ifdef BWMA
    std::cout << "BWMA method" << std::endl;
    Transpose::transpose_rearranged(key_layer_out, key_transposed_layer_out, head_hidden_size_,
                                    pre_seq_len_, kernel_size_, max_col_);
#ifdef SIMD
    simdComputeBWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
                          head_hidden_size_, seq_len);
#else
    smmComputeBWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                   seq_len);
#endif
    softmax->computeRearranged(attention_scores, seq_len, kernel_size_);
#ifdef SIMD
    simdComputeBWMA(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
#else
    smmComputeBWMA(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
#endif
#else
    std::cout<< "RWMA method" << std::endl;
    Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_,
                                    pre_seq_len_);
#ifdef SIMD
    simdComputeRWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
               head_hidden_size_, seq_len);
#else
    smmComputeRWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
                head_hidden_size_, seq_len);
#endif
    softmax->compute(attention_scores, seq_len);
#ifdef SIMD
    simdComputeRWMA(seq_len, attention_scores, output, value_layer_out,
               seq_len, head_hidden_size_);
#else
    smmComputeRWMA(seq_len, attention_scores, output, value_layer_out,
                seq_len, head_hidden_size_);
#endif
#endif

    softmax->post_softmax(output, seq_len, head_hidden_size_);
}