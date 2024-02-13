#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
//#include <cstdint>
#include "debuggerFunctions.h"
#include "sparse_rep.h"

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim,
                                       std::size_t head_hidden_size, Format sparseFormat) {
    // This is the base constructor for the SingleHeadSelfAttn class.
    // It takes in the pre_seq_len, input_dim, head_hidden_size, and sparseFormat as parameters.
    // Other constructors will be inherited from this one.
    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    hidden_flag_ = nullptr;
    softmax = new Softmax();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len >> 2]();

    sparseMatrixMultiplier_QKT = new SparseMatrixMultiplier(head_hidden_size_, pre_seq_len, pre_seq_len,
                                                            Format::NON_PRUNED );
    sparseMatrixMultiplier_att_v = new SparseMatrixMultiplier(pre_seq_len, head_hidden_size_, pre_seq_len,
                                                        Format::NON_PRUNED);
}

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t **weightVector, uint32_t **flagVector, uint32_t *hidden_flag,
                                       Format sparseFormat) :
        SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, sparseFormat) {

    // This is the constructor for the cases where sparseFormat is not CSC or CSR.
    hidden_flag_ = hidden_flag;

    query_layer = new Dense(pre_seq_len, input_dim, head_hidden_size, weightVector[0],
                            flagVector[0], hidden_flag, sparseFormat);
    key_layer = new Dense(pre_seq_len, input_dim, head_hidden_size, weightVector[1],
                          flagVector[1], hidden_flag, sparseFormat);
    value_layer = new Dense(pre_seq_len, input_dim, head_hidden_size, weightVector[2],
                            flagVector[2], hidden_flag, sparseFormat);

}

// Another constructor where sparseFormat is CSC or CSR.
SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t **weightVector, int** col_ptr, int** row_ptr, uint32_t ***values,
                                       Format sparseFormat) :
        SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, sparseFormat) {

    query_layer = new Dense(pre_seq_len, input_dim, head_hidden_size, weightVector[0],
                            col_ptr[0], row_ptr[0], values[0], sparseFormat);
    key_layer = new Dense(pre_seq_len, input_dim, head_hidden_size, weightVector[1],
                          col_ptr[1], row_ptr[1], values[1], sparseFormat);
    value_layer = new Dense(pre_seq_len, input_dim, head_hidden_size, weightVector[2],
                            col_ptr[2], row_ptr[2], values[2], sparseFormat);
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


    sparseMatrixMultiplier_QKT->compute(query_layer_out, attention_scores,
                                    nullptr, nullptr, key_transposed_layer_out);

    softmax->computeRearranged(attention_scores, seq_len);

    sparseMatrixMultiplier_att_v->compute(attention_scores, output,
                                    nullptr, nullptr, value_layer_out);

    softmax->post_softmax(output, seq_len, head_hidden_size_);
}