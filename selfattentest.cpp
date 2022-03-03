#include"kernels/transformerBlock.h"
//#include"gtest/gtest.h"
#include "selfattentest.h"


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * j);
        }
        kernel[i]=result;
    }
}

void test(){
    uint32_t tensor_in[D_SEQ * D_MODEL >> 2];
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    uint32_t out[D_SEQ*D_MODEL >> 2];

    uint32_t * weightVec[3*NUM_HEAD+2];

    for (int n=0; n<NUM_HEAD; n++){
        uint32_t query_kernel[D_Q* D_MODEL >> 2];
        fill_kernel(query_kernel, D_Q* D_MODEL >> 2);
        weightVec[n*3] = query_kernel;
        uint32_t query_bias[D_Q >> 2];
        fill_kernel(query_bias, D_Q >> 2);

        uint32_t key_kernel[ D_Q* D_MODEL >> 2];
        fill_kernel(key_kernel, D_Q* D_MODEL>> 2);
        weightVec[n*3 + 1] = key_kernel;
        uint32_t key_bias[D_Q >> 2];
        fill_kernel(key_bias, D_Q>> 2);

        uint32_t value_kernel[ D_Q* D_MODEL >> 2];
        fill_kernel(value_kernel, D_Q* D_MODEL >> 2);
        weightVec[n*3 + 2] = value_kernel;
        uint32_t value_bias[D_Q >> 2];
        fill_kernel(value_bias, D_Q >> 2);
    }

    uint32_t ff0_kernel[ D_MODEL* D_FF >> 2];
    fill_kernel(ff0_kernel, D_MODEL* D_FF >> 2);
    weightVec[NUM_HEAD*3] = ff0_kernel;

    uint32_t ff1_kernel[ D_FF* D_MODEL >> 2];
    fill_kernel(ff1_kernel, D_FF* D_MODEL >> 2);
    weightVec[NUM_HEAD*3 + 1] = ff1_kernel;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfatten.compute(D_SEQ, tensor_in, out);
}

int main() {
    test();
    return 0;
}