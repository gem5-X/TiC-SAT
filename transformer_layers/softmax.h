#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"

//    T exp_(T input);
//
//    T sum_(T sum);

class Softmax
{
    public:
        explicit Softmax();
        ~Softmax();
        void compute(uint32_t *input, std::size_t seq_len);
        void computeFloat(uint32_t *input, std::size_t seq_len);
        void computeRearranged(uint32_t *input, std::size_t seq_len);
        void post_softmax(uint32_t *input, size_t seq_len, size_t);
    private:
        int32_t float_to_fixed(float value, int32_t fractional_bits);
        float fixed_to_float(int32_t fixed_value, int32_t fractional_bits);
        void softmax_fixed(std::vector<int32_t> &input, int32_t fractional_bits);

};
