#include "softmax.h"
#include <cmath>
#include <iostream>
//
//    float exp_<float>(float input){
//        return expf(input);
//    }
//
//    float sum_<float>(float sum){
//        return sum > 1e-22f ? sum : 1e-22f;
//    }

static const  int8_t lookup[8] = {
    0, 0, 0, 1, 0, 0, 0, 0
};

Softmax::Softmax()= default;

Softmax::~Softmax()= default;

void Softmax::compute(uint32_t *input, std::size_t seq_len){
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_ptr = (int8_t*) (input + i * (seq_len >> 2));

        for (int j=0; j< seq_len; j++){
            *(input_ptr) = lookup[(* (uint8_t *) input_ptr) >> 5]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
            // TODO: better LUT
            sum += *(input_ptr);
            input_ptr ++;
        }
//        std::cout << "EXP " << i << "\t: " << sum << std::endl;
        input_ptr = (int8_t*) (input + i * (seq_len >> 2));
        for (int j=0; j< seq_len; j++){
//            std::cout << "Ptr " << i << "\t: " << (int) *(input_ptr)  << std::endl;
            *(input_ptr) = (int8_t) ((*(input_ptr) << 7) /sum);
//            std::cout << "LUT " << i << "\t: " << (int) *(input_ptr) << std::endl;
            input_ptr ++;
        }
    }
}
