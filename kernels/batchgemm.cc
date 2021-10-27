#include "batchgemm.h"
//#include "mkl.h"
#include <iostream>

namespace lh{
    template<>
    void attn_qk<float>(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float* query, float* key, float* output, const float** q_array, const float** k_array, float** pointer_qk_array) {

        for (std::size_t idx = 0; idx < batch_size; idx++)
        {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++)
            {
                q_array[idx * num_heads + head_idx] = query + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                k_array[idx * num_heads + head_idx] = key + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                pointer_qk_array[idx * num_heads + head_idx] = output + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
                float sum = 0;
                for (int i=0; i< seq_len; i++){
                    sum = 0.0;
                    for (int h=0; h< hidden_size; h++){
                        sum += q_array[idx * num_heads + head_idx][h+ i*hidden_size] * k_array[idx * num_heads + head_idx][h + i*hidden_size];
                    }
                    pointer_qk_array[idx * num_heads + head_idx][i] = sum;
//                    std::cout << "QK :" << sum << std::endl;
                }
            }
        }


//        CBLAS_TRANSPOSE tranA = CblasNoTrans;
//        CBLAS_TRANSPOSE tranB = CblasTrans;
//        const int m = seq_len, n = seq_len, k = hidden_size, lda_array = hidden_size * num_heads, ldb_array = hidden_size * num_heads, ldc_array = seq_len * num_heads, group_size = batch_size * num_heads;
//        const float alpha = 1.0, beta = 0.0;
//        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, q_array, &lda_array, k_array, &ldb_array, &beta, pointer_qk_array, &ldc_array, 1, &group_size);
        
    }

    template<>
    void attn_sv<float>(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float* sim, float* value, float* output, const float** sim_array, const float** value_array, float** pointer_sv_array){

        for (std::size_t idx = 0; idx < batch_size; idx++)
        {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++)
            {
                sim_array[idx * num_heads + head_idx] = sim + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
                value_array[idx * num_heads + head_idx] = value + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                pointer_sv_array[idx * num_heads + head_idx] = output + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;

                float sum = 0;
                for (int i=0; i< seq_len; i++){
                    sum = 0.0;
                    for (int h=0; h< hidden_size; h++){
                        sum += sim_array[idx * num_heads + head_idx][h+ i*hidden_size] * value_array[idx * num_heads + head_idx][h + i*hidden_size];
                    }
                    pointer_sv_array[idx * num_heads + head_idx][i] = sum;
//                    std::cout << "SV :" << sum << std::endl;
                }
            }
        }
//        CBLAS_TRANSPOSE tranA = CblasNoTrans;
//        CBLAS_TRANSPOSE tranB = CblasNoTrans;
//        const int m = seq_len, n = hidden_size, k = seq_len, lda_array = seq_len * num_heads, ldb_array = hidden_size * num_heads, ldc_array = hidden_size * num_heads, group_size = batch_size * num_heads;
//        const float alpha = 1.0, beta = 0.0;
//        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, sim_array, &lda_array, value_array, &ldb_array, &beta, pointer_sv_array, &ldc_array, 1, &group_size);
        
    }
//
//    template<>
//    void batchMatMul<float>(std::size_t batch_size, std::size_t seq_len, float* input, float* output, float *weight,
//                            std::size_t input_size_, std::size_t output_size_){
//        for (int b=0; b < batch_size; b++){
//            for (int length=0; length< seq_len; length++){
//                for (int out_idx=0; out_idx < output_size_; out_idx ++){
//                    float sum = 0.0;
//                    for (int i=0; i< input_size_; i++){
//                        sum += weight[out_idx* input_size_ + i] * input[(b*seq_len+length) * input_size_ + i];
//                    }
//                    output[(b*seq_len+length) * output_size_ + out_idx] = sum;
//                }
//            }
//        }
//
//    }
}