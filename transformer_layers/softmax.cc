#include "softmax.h"
#include <cmath>

static const  uint8_t  lookup[32] = {
        4, 5, 7, 8, 11, 14, 18, 23, 30, 38, 49, 63, 80, 103, 132, 170, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3,
};

Softmax::Softmax()= default;

Softmax::~Softmax()= default;

void Softmax::compute(uint32_t *input, std::size_t seq_len){
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = (uint8_t*) (input + i * (seq_len >> 2));

        for (int j=0; j< seq_len; j++){
            *(input_uptr) = lookup[(* (uint8_t *) input_uptr) >> 3]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
            sum += *(input_uptr);
            input_uptr ++;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = (uint8_t*) (input + i * (seq_len >> 2));
        for (int j=0; j< seq_len; j++){
//            std::cout << "Ptr " << i << "\t: " << (int) *(input_ptr)  << std::endl;
            *(input_uptr) = (uint8_t) ((*(input_uptr)) /(sum >> 8)); // divide the sum by 256 otherwise all the outputs will be 0!
//            std::cout << "LUT " << i << "\t: " << (int) *(input_ptr) << std::endl;
            input_uptr ++;
        }
        input_uptr = (uint8_t*) (input + i * (seq_len >> 2));
        int sum_softmax = 0;
        for (int j=0; j< seq_len; j++)
            sum_softmax += *(input_uptr++);
    }
}

void Softmax::post_softmax(uint32_t *input, std::size_t seq_len){
    for (int i =0; i< seq_len; i++){
        auto* input_ptr = (int8_t*) (input + i * (seq_len >> 2));
        for (int j=0; j< seq_len; j++){
            *input_ptr = (int8_t) (*(input_ptr) >> 6);
            input_ptr++;
        }
    }
}
