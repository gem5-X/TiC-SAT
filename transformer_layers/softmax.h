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
        void post_softmax(uint32_t *input, size_t seq_len);
    private:

};