//
// Created by alireza on 10/19/21.
//

#include "util.h"

namespace lh
{
    template <class T>
    class MatMul
        {
        public:
            explicit MatMul();
            ~MatMul();
            void compute(std::size_t batch_size, std::size_t seq_len, float* input, float* output, T *weight,
                         std::size_t input_size_, std::size_t output_size_);
        private:

        };
}
//template<class T>
//void matrix_multiply(std::size_t batch_size, std::size_t seq_len, float* input, float* output, T *weight,
//                     std::size_t input_size_, std::size_t output_size_);

