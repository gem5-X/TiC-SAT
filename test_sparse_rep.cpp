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
    Format format = Format::WITH_FLAG;
    const uint32_t hidden_flag = 0xAAAAAAAA;
    uint32_t* tensor_in = new uint32_t [D_SEQ * D_MODEL >> 2];
    loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, 0, dir_name, nullptr);
    uint32_t* out = new uint32_t [D_SEQ*D_MODEL >> 2]();
    uint32_t* query_kernel = new uint32_t [D_Q* D_MODEL >> 2]();
    uint32_t* query_flag = new uint32_t [(D_Q* D_MODEL) / (32* KERNEL_DIM * KERNEL_DIM)]();
    int head_qkv_size = D_Q* D_MODEL >> 2;
    int head_flag_size = (D_Q* D_MODEL) / (32* KERNEL_DIM * KERNEL_DIM);
    uint32_t* m1 = new uint32_t [D_Q / KERNEL_DIM]();
    uint32_t* m2 = new uint32_t [(D_Q * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();

    // Initialize the parameters for format::CSC or format::CSR
    int* col_ptr;
    int* row_ptr;
    uint32_t** values;

    loadWeight(0, 0, head_qkv_size, query_kernel, 10, dir_name, &hidden_flag);
    // Change the representation of the query kernel
    if (format == Format::WITH_FLAG)
        remove_zero_tiles(query_kernel, D_MODEL, D_Q >> 2);
    else if (format == Format::HIDDEN_KEY)
        interleave_hidden_flag_zero_free(query_kernel, D_MODEL, D_Q >> 2, hidden_flag);
    else if (format == Format::META_DATA)
        dense2metaData(&query_kernel, D_MODEL, D_Q >> 2, m1, m2);
    else if (format == Format::CSC){
        col_ptr = new int [D_Q / KERNEL_DIM + 1]();
        row_ptr = new int [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();
        values = new uint32_t* [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();

        int nnz = dense2csc(query_kernel, D_MODEL, D_Q >> 2, col_ptr, row_ptr, values);
    }
    else if (format == Format::CSR){
        row_ptr = new int [D_MODEL / KERNEL_DIM + 1]();
        col_ptr = new int [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();
        values = new uint32_t* [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();

        int nnz = dense2csr(query_kernel, D_MODEL, D_Q >> 2, col_ptr, row_ptr, values);
    }

    loadWeight(0, 10, head_flag_size, query_flag, 10, dir_name, nullptr);
    if (format == Format::META_DATA){
        auto testDense = new Dense(D_MODEL, D_Q, query_kernel, m1, (const uint32_t*)(m2), format);
        testDense->compute(D_SEQ, tensor_in, out);
    } else if (format == Format::CSC || format == Format::CSR){
        auto testDense = new Dense(D_MODEL, D_Q, query_kernel, col_ptr, row_ptr, values, format);
        testDense->compute(D_SEQ, tensor_in, out);
    }
    else{
        auto testDense = new Dense(D_MODEL, D_Q, query_kernel, query_flag, &hidden_flag, format);
        testDense->compute(D_SEQ, tensor_in, out);
    }

    //save the output to a file
//    saveWeight(-2, 0, D_SEQ * D_MODEL >> 2, out, 10, dir_name);
//    getchar();

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

