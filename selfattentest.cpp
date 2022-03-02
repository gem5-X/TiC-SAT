#include"kernels/selfattention.h"
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
    std::vector<uint32_t *> weightVec;
    uint32_t query_kernel[D_Q* D_MODEL >> 2];
    fill_kernel(query_kernel, D_Q* D_MODEL >> 2);
    weightVec.push_back(query_kernel);
    uint32_t query_bias[D_Q >> 2];
    fill_kernel(query_bias, D_Q >> 2);

    uint32_t key_kernel[D_Q* D_MODEL >> 2];
    fill_kernel(key_kernel, D_Q* D_MODEL>> 2);
    weightVec.push_back(key_kernel);
    uint32_t key_bias[D_Q >> 2];
    fill_kernel(key_bias, D_Q>> 2);

    uint32_t value_kernel[D_Q* D_MODEL >> 2];
    fill_kernel(value_kernel, D_Q* D_MODEL >> 2);
    weightVec.push_back(value_kernel);
    uint32_t value_bias[D_Q >> 2];
    fill_kernel(value_bias, D_Q >> 2);

    std::vector<std::string> names = {"query/weight", "query/bias", "key/weight", "key/bias", "value/weight", "value/bias"};

    size_t batch_size = 1;
    size_t num_attention_heads = 1;
    size_t size_per_head = 3;
    size_t seq_length = D_SEQ;
    SingleHeadSelfAttn selfatten(names, D_SEQ, 1, D_MODEL, D_Q, weightVec);
    uint32_t tensor_in[D_SEQ * D_MODEL >> 2];
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    uint64_t mask[1];
    mask[0] = D_SEQ;

    uint32_t out[D_SEQ*D_Q >> 2];

    selfatten.compute(seq_length, tensor_in, out);
}

int main() {
    test();
    return 0;
}