//
// Created by alireza on 10/19/21.
//

#include "matmul.h"
#include <iostream>

namespace lh{
    template<class T>
    MatMul<T>::MatMul()= default;

    template<class T>
    MatMul<T>::~MatMul<T>()= default;

    template<class T>
        void MatMul<T>::compute(std::size_t batch_size, std::size_t seq_len, float* input, float* output, T *weight,
                                std::size_t input_size_, std::size_t output_size_){
            for (int b=0; b < batch_size; b++){
                for (int length=0; length< seq_len; length++){
                    for (int out_idx=0; out_idx < output_size_; out_idx ++){
                        float sum = 0.0;
                        for (int i=0; i< input_size_; i++){
                            sum += weight[out_idx* input_size_ + i] * input[(b*seq_len+length) * input_size_ + i];
                        }
                        output[(b*seq_len+length) * output_size_ + out_idx] = sum;
                    }
                }
            }
        }

        template class MatMul<float>;
}