//
// Created by alireza on 10/19/21.
//

#include <vector>

namespace lh
{
    template <class T>
    class MatMul
        {
        public:
            explicit MatMul();
            ~MatMul();
            void compute(std::size_t seq_len, const int* input, int* output, T *weight,
                         std::size_t input_size_, std::size_t output_size_);
        private:

        };
}
