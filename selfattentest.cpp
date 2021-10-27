#include"kernels/selfattention.h"
//#include"gtest/gtest.h"
#include "selfattentest.h"

using namespace std;
using namespace lh;


void fill_kernel(float* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++)
        kernel[i]=static_cast <float> ((rand() % 200  - 100) / 100.0);
}

void test(){
    float query_kernel[D_Q* D_MODEL];
    fill_kernel(query_kernel, D_Q* D_MODEL);
    float query_bias[D_Q];
    fill_kernel(query_bias, D_Q);

    float key_kernel[D_Q* D_MODEL];
    fill_kernel(key_kernel, D_Q* D_MODEL);
    float key_bias[D_Q];
    fill_kernel(key_bias, D_Q);

    float value_kernel[D_Q* D_MODEL];
    fill_kernel(value_kernel, D_Q* D_MODEL);
    float value_bias[D_Q];
    fill_kernel(value_bias, D_Q);
    
    Graph<float> graph;
    graph["query/weight"] = make_pair(vector<size_t>({D_MODEL, D_Q}), query_kernel);
    graph["query/bias"] = make_pair(vector<size_t>({D_Q}), query_bias);
    graph["key/weight"] = make_pair(vector<size_t>({D_MODEL, D_Q}), key_kernel);
    graph["key/bias"] = make_pair(vector<size_t>({D_Q}), key_bias);
    graph["value/weight"] = make_pair(vector<size_t>({D_MODEL, D_Q}), value_kernel);
    graph["value/bias"] = make_pair(vector<size_t>({D_Q}), value_bias);

    vector<string> names = {"query/weight", "query/bias", "key/weight", "key/bias", "value/weight", "value/bias"};

    size_t batch_size = 1;
    size_t num_attention_heads = 2;
    size_t size_per_head = 3;
    size_t seq_length = D_SEQ;
    MutiheadselfAttn<float> selfatten(names, graph, 1, D_SEQ, 1, D_Q);
    float tensor_in[D_SEQ * D_MODEL];
    fill_kernel(tensor_in, D_SEQ * D_MODEL);
    uint64_t mask[1];
    mask[0] = D_SEQ;

    float out[D_SEQ*D_Q];

    selfatten.compute(batch_size, seq_length, tensor_in, mask, out);
}

int main() {
    test();
    return 0;
}