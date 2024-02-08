//
// Created by alireza on 2/7/24.
//

#include "transformer.h"
#include "transformer_layers/sparse_rep.h"
#include "accelerator/sparseMatrixMultiplication.h"
#include "transformer_layers/dataFunctions.h"
#include "transformer_layers/dense.h"

int main(){
    std::string dir_name = "/home/alireza/CLionProjects/FvllMontiTransformer/data16";
    const uint32_t hidden_flag = 0xAAAAAAAA;
    uint32_t* tensor_in = new uint32_t [D_SEQ * D_MODEL >> 2];
    loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, 0, dir_name, nullptr);
    uint32_t* out = new uint32_t [D_SEQ*D_MODEL >> 2]();
    uint32_t* query_kernel = new uint32_t [D_Q* D_MODEL >> 2]();
    uint32_t* query_flag = new uint32_t [(D_Q* D_MODEL) / (32* KERNEL_DIM * KERNEL_DIM)]();
    int head_qkv_size = D_Q* D_MODEL >> 2;
    int head_flag_size = (D_Q* D_MODEL) / (32* KERNEL_DIM * KERNEL_DIM);

    loadWeight(0, 0, head_qkv_size, query_kernel, 10, dir_name, &hidden_flag);
    // Change the representation of the query kernel

    remove_zero_tiles(query_kernel, D_MODEL, D_Q >> 2);
    loadWeight(0, 10, head_flag_size, query_flag, 10, dir_name, nullptr);
    Dense* testDense;
    testDense = new Dense(D_Q, D_MODEL, query_kernel, query_flag, nullptr);
    testDense->compute(D_SEQ, tensor_in, out);

    //save the output to a file
    //saveWeight(-2, 0, D_SEQ * D_MODEL >> 2, out, 10, dir_name);

    //load output from file
    auto out_file = new uint32_t [D_SEQ*D_MODEL >> 2]();
    loadWeight(-2, 0, D_SEQ * D_MODEL >> 2, out_file, 10, dir_name, nullptr);

    //compare the output with the file
    for (int i = 0; i < D_SEQ*D_MODEL >> 2; i++){
        if (out[i] != out_file[i]){
            std::cout << "Error at index " << i << std::endl;
            std::cout << "Expected: " << std::hex << out_file[i] << std::endl;
            std::cout << "Got: " << std::hex << out[i] << std::endl;
        }
    }


    return 0;
}


