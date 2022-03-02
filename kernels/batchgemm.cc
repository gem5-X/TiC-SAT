#include "batchgemm.h"
#include <iostream>

//void attn_qk(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float *query,
//             float *key, float *output, const float **q_array, const float **k_array, float **pointer_qk_array) {
//
//    for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++) {
//        q_array[head_idx] = query + head_idx * hidden_size;
//        k_array[head_idx] = key + head_idx * hidden_size;
//        pointer_qk_array[head_idx] = output + head_idx * seq_len;
//        float sum = 0;
//        for (int i = 0; i < seq_len; i++) {
//            sum = 0.0;
//            for (int h = 0; h < hidden_size; h++) {
//                sum += q_array[head_idx][h + i * hidden_size] * k_array[head_idx][h + i * hidden_size];
//            }
//            pointer_qk_array[head_idx][i] = sum;
////                    std::cout << "QK :" << sum << std::endl;
//        }
//    }
//
//
////        CBLAS_TRANSPOSE tranA = CblasNoTrans;
////        CBLAS_TRANSPOSE tranB = CblasTrans;
////        const int m = seq_len, n = seq_len, k = hidden_size, lda_array = hidden_size * num_heads, ldb_array = hidden_size * num_heads, ldc_array = seq_len * num_heads, group_size = batch_size * num_heads;
////        const float alpha = 1.0, beta = 0.0;
////        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, q_array, &lda_array, k_array, &ldb_array, &beta, pointer_qk_array, &ldc_array, 1, &group_size);
//
//}
//
//void
//attn_sv(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float *sim,
//               float *value, float *output, const float **sim_array, const float **value_array,
//               float **pointer_sv_array) {
//
//    for (std::size_t idx = 0; idx < batch_size; idx++) {
//        for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++) {
//            sim_array[idx * num_heads + head_idx] = sim + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
//            value_array[idx * num_heads + head_idx] =
//                    value + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
//            pointer_sv_array[idx * num_heads + head_idx] =
//                    output + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
//
//            float sum = 0;
//            for (int i = 0; i < seq_len; i++) {
//                sum = 0.0;
//                for (int h = 0; h < hidden_size; h++) {
//                    sum += sim_array[idx * num_heads + head_idx][h + i * hidden_size] *
//                           value_array[idx * num_heads + head_idx][h + i * hidden_size];
//                }
//                pointer_sv_array[idx * num_heads + head_idx][i] = sum;
////                    std::cout << "SV :" << sum << std::endl;
//            }
//        }
//    }
////        CBLAS_TRANSPOSE tranA = CblasNoTrans;
////        CBLAS_TRANSPOSE tranB = CblasNoTrans;
////        const int m = seq_len, n = hidden_size, k = seq_len, lda_array = seq_len * num_heads, ldb_array = hidden_size * num_heads, ldc_array = hidden_size * num_heads, group_size = batch_size * num_heads;
////        const float alpha = 1.0, beta = 0.0;
////        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, sim_array, &lda_array, value_array, &ldb_array, &beta, pointer_sv_array, &ldc_array, 1, &group_size);
//
//}

void Transpose::transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) {
    uint32_t swap[4];
    std::size_t height_tile = height >> 2;
    std::size_t width_tile = width >> 2;
    for (int i=0; i < height_tile; i++){
        for (int j=0; j < width_tile; j++){
            for( int k=0; k< 4; k++){
                swap[k] = input[(i * 4 + k) * width_tile + j];
            }

            for ( int k=0; k< 4; k++){
                uint32_t result = 0;
                for (int s=0; s< 4; s++)
                    result |= ((swap [s] >> (24 - 8*k)) & 0xFF) << (24 - 8*s);
                output[(j * 4 + k) * height_tile + i] = result;
            }
        }
    }
}
