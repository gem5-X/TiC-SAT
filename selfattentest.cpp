#include"kernels/transformerBlock.h"
//#include"gtest/gtest.h"
#include "selfattentest.h"


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * (4 - i - 1));
        }
        kernel[i]=result;
    }
}

void test(){
    uint32_t tensor_in[D_SEQ * D_MODEL >> 2];
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    uint32_t out[NUM_HEAD * D_SEQ*D_Q >> 2];

    uint32_t * weightVec[3*NUM_HEAD];

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

    std::vector<std::string> names = {"query/weight", "query/bias", "key/weight", "key/bias", "value/weight", "value/bias"};


    TransformerBlock selfatten(names, D_SEQ, D_MODEL, D_Q, NUM_HEAD, weightVec);
    selfatten.compute(D_SEQ, tensor_in, out );
}

int main() {
    test();
    return 0;
}