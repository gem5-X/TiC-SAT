#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include "transformer_layers/debuggerFunctions.h"
#include "transformer_layers/dataFunctions.h"
#include "transformer_layers/sparse_rep.h"

//#define KERNEL_DIM SA_SIZE
//#define MAX_COL (SA_SIZE/4)


void inference(int sparsityPercentage, Format sparseFormat){
    uint32_t hidden_flag ;
    hidden_flag = 0xAAAAAAAA;

    std::cout<<"First line" << std::endl;
#ifdef DEVELOP
    std::string dir_name = "/home/alireza/CLionProjects/FvllMontiTransformer/data16";
#else
    std::string dir_name = "/mnt/data";
#endif

    uint32_t* tensor_in = new uint32_t [D_SEQ * D_MODEL >> 2];
    #ifdef RELOAD_WEIGHT
        loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, 0, dir_name, nullptr);
    #else
        fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
        saveWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, 0, dir_name);

        std::ofstream outfile("flags_generated.h");
        outfile << "#pragma once" << std::endl;
        outfile << std::endl;
        outfile << "#include <cstdint>" << std::endl;
        outfile << std::endl;
        outfile << "static const uint32_t flags[] = {";
        outfile.close();
    #endif


    uint32_t* out = new uint32_t [D_SEQ*D_MODEL >> 2]();

    uint32_t * weightVec[3*NUM_HEAD+3];
    uint32_t * flagVec[3*NUM_HEAD+3];
    int* col_ptr[3*NUM_HEAD+3];
    int* row_ptr[3*NUM_HEAD+3];
    uint32_t** values[3*NUM_HEAD+3];

    int head_qkv_size = D_Q* D_MODEL >> 2;
    int head_flag_size = (D_Q* D_MODEL) / (32* KERNEL_DIM * MAX_COL);

    for (int n=0; n<NUM_HEAD; n++){
        volatile auto query_kernel = new uint32_t [D_Q* D_MODEL >> 2]();
        volatile auto query_flag = new uint32_t [(D_Q* D_MODEL) / (32* KERNEL_DIM * MAX_COL)]();

        volatile auto key_kernel = new uint32_t [ D_Q* D_MODEL >> 2]();
        volatile auto key_flag = new uint32_t [(D_Q* D_MODEL) / (32 * KERNEL_DIM * MAX_COL)]();

        volatile auto value_kernel = new uint32_t [ D_Q* D_MODEL >> 2]();
        volatile auto value_flag = new uint32_t [D_Q* D_MODEL / (32* KERNEL_DIM * MAX_COL)]();

#ifdef RELOAD_WEIGHT
        loadWeight(n, 0, head_qkv_size, query_kernel, sparsityPercentage, dir_name, &hidden_flag);
        loadWeight(n, 1, head_qkv_size, key_kernel, sparsityPercentage, dir_name, &hidden_flag);
        loadWeight(n, 2, head_qkv_size, value_kernel, sparsityPercentage, dir_name, &hidden_flag);
        loadWeight(n, 10, head_flag_size, query_flag, sparsityPercentage, dir_name, nullptr);
        loadWeight(n, 11, head_flag_size, key_flag, sparsityPercentage, dir_name, nullptr);
        loadWeight(n, 12, head_flag_size, value_flag, sparsityPercentage, dir_name, nullptr);
#else
        fill_sparse_weight(query_kernel, query_flag, D_MODEL, D_Q >> 2, sparsityPercentage);
        fill_sparse_weight(key_kernel, key_flag, D_MODEL, D_Q >> 2, sparsityPercentage);
        fill_sparse_weight(value_kernel, value_flag, D_MODEL, D_Q >> 2, sparsityPercentage);
        if (!std::filesystem::exists(dir_name)) {
            std::filesystem::create_directory(dir_name);
        }

        saveWeight(n, 0, head_qkv_size, query_kernel, sparsityPercentage, dir_name);
        saveWeight(n, 1, head_qkv_size, key_kernel, sparsityPercentage, dir_name);
        saveWeight(n, 2, head_qkv_size, value_kernel, sparsityPercentage, dir_name);

        saveWeight(n, 10, head_flag_size, query_flag, sparsityPercentage, dir_name);
        saveWeight(n, 11, head_flag_size, key_flag, sparsityPercentage, dir_name);
        saveWeight(n, 12, head_flag_size, value_flag, sparsityPercentage, dir_name);
        append_flags(query_flag, head_flag_size);
        append_flags(key_flag, head_flag_size);
        append_flags(value_flag, head_flag_size);
#endif


        weightVec[n*3] = query_kernel;
        flagVec[n*3] = query_flag;

        weightVec[n*3 + 1] = key_kernel;
        flagVec[n*3 + 1] = key_flag;

        weightVec[n*3 + 2] = value_kernel;
        flagVec[n*3+2] = value_flag;
    }

    volatile auto condense_kernel = new uint32_t [ NUM_HEAD * D_Q * D_MODEL >> 2]();
    volatile auto condense_flag = new uint32_t [ NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL)]();

    volatile auto ff0_kernel = new uint32_t [ D_MODEL* D_FF >> 2]();
    volatile auto ff0_flag = new uint32_t [ D_MODEL* D_FF / (32 * KERNEL_DIM * MAX_COL)]();

    volatile auto ff1_kernel = new uint32_t [ D_FF* D_MODEL >> 2]();
    volatile auto ff1_flag = new uint32_t [ D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL)]();

    #ifdef RELOAD_WEIGHT
        int n = -1; // n=-1 means that we are not saving/loading a head

        loadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, sparsityPercentage, dir_name, &hidden_flag);
        loadWeight(n, 1, D_MODEL* D_FF >> 2, ff0_kernel, sparsityPercentage, dir_name, &hidden_flag);
        loadWeight(n, 2, D_MODEL* D_FF >> 2, ff1_kernel, sparsityPercentage, dir_name, &hidden_flag);

        loadWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL), condense_flag, sparsityPercentage, dir_name,
                   nullptr);
        loadWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff0_flag, sparsityPercentage, dir_name, nullptr);
        loadWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff1_flag, sparsityPercentage, dir_name, nullptr);

    #else
        fill_sparse_weight(condense_kernel, condense_flag, D_MODEL, NUM_HEAD * D_Q >> 2, sparsityPercentage);
        fill_sparse_weight(ff0_kernel, ff0_flag, D_MODEL, D_FF >> 2, sparsityPercentage);
        fill_sparse_weight(ff1_kernel, ff1_flag, D_FF, D_MODEL >> 2, sparsityPercentage);

        int n = -1; // n=-1 means that we are not saving/loading a head
        saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, sparsityPercentage, dir_name);
        saveWeight(n, 1, D_MODEL* D_FF >> 2, ff0_kernel, sparsityPercentage, dir_name);
        saveWeight(n, 2, D_MODEL* D_FF >> 2, ff1_kernel, sparsityPercentage, dir_name);

        saveWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL), condense_flag, sparsityPercentage, dir_name);
        saveWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff0_flag, sparsityPercentage, dir_name);
        saveWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff1_flag, sparsityPercentage, dir_name);

        append_flags(condense_flag, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL));
        append_flags(ff0_flag, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL));
        append_flags(ff1_flag, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL));

        outfile.open("flags_generated.h", std::ios::app);
        outfile << "};" << std::endl;
        outfile.close();
    #endif

    weightVec[NUM_HEAD*3] = condense_kernel;
    flagVec[NUM_HEAD*3] = condense_flag;

    weightVec[NUM_HEAD*3+1] = ff0_kernel;
    flagVec[NUM_HEAD*3+1] = ff0_flag;

    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;
    flagVec[NUM_HEAD*3 + 2] = ff1_flag;

    if (sparseFormat == Format::WITH_FLAG){
        for (int i = 0; i < NUM_HEAD * 3; i++){
            // call remove_zero_tiles for each weight vector in query, key and value
            remove_zero_tiles(weightVec[i], (int) D_MODEL, (int) D_Q >> 2);
        }
        remove_zero_tiles(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2);
        remove_zero_tiles(weightVec[NUM_HEAD*3 + 1], D_MODEL, D_FF >> 2);
        remove_zero_tiles(weightVec[NUM_HEAD*3 + 2], D_FF, D_MODEL >> 2);
    }
    else if (sparseFormat == Format::HIDDEN_KEY){
        for (int i = 0; i < NUM_HEAD * 3; ++i) {
            interleave_hidden_flag_zero_free(weightVec[i], D_MODEL,D_Q >> 2, hidden_flag);
        }
        interleave_hidden_flag_zero_free(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2, hidden_flag);
        interleave_hidden_flag_zero_free(weightVec[NUM_HEAD*3+1], D_MODEL, D_FF >> 2, hidden_flag);
        interleave_hidden_flag_zero_free(weightVec[NUM_HEAD*3+2], D_FF, D_MODEL >> 2, hidden_flag);
    }
    else if(sparseFormat == Format::CSC){
        for (int i = 0; i < NUM_HEAD * 3; ++i) {
            col_ptr[i] = new int [D_Q / KERNEL_DIM + 1]();
            row_ptr[i] = new int [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();
            values[i] = new uint32_t* [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();
            dense2csc(weightVec[i], D_MODEL, D_Q >> 2, col_ptr[i], row_ptr[i], values[i]);
        }
        col_ptr[NUM_HEAD*3] = new int [D_MODEL / KERNEL_DIM + 1]();
        row_ptr[NUM_HEAD*3] = new int [(NUM_HEAD * D_Q * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
        values[NUM_HEAD*3] = new uint32_t* [(NUM_HEAD * D_Q * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
        dense2csc(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2, col_ptr[NUM_HEAD*3],
                  row_ptr[NUM_HEAD*3], values[NUM_HEAD*3]);

        col_ptr[NUM_HEAD*3+1] = new int [D_FF / KERNEL_DIM + 1]();
        row_ptr[NUM_HEAD*3+1] = new int [(D_MODEL * D_FF) / (KERNEL_DIM * KERNEL_DIM)]();
        values[NUM_HEAD*3+1] = new uint32_t* [(D_MODEL * D_FF) / (KERNEL_DIM * KERNEL_DIM)]();
        dense2csc(weightVec[NUM_HEAD*3+1], D_MODEL, D_FF >> 2, col_ptr[NUM_HEAD*3+1],
                  row_ptr[NUM_HEAD*3+1], values[NUM_HEAD*3+1]);

        col_ptr[NUM_HEAD*3+2] = new int [D_MODEL / KERNEL_DIM + 1]();
        row_ptr[NUM_HEAD*3+2] = new int [(D_FF * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
        values[NUM_HEAD*3+2] = new uint32_t* [(D_FF * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
        dense2csc(weightVec[NUM_HEAD*3+2], D_FF, D_MODEL >> 2, col_ptr[NUM_HEAD*3+2],
                  row_ptr[NUM_HEAD*3+2], values[NUM_HEAD*3+2]);
    }

    if (sparseFormat == Format::CSC || sparseFormat == Format::CSR){
        TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF,
                                   weightVec, col_ptr, row_ptr, values, sparseFormat);
        selfatten.compute(D_SEQ, tensor_in, out);
    }else{
        TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF,
                                   weightVec, flagVec, &hidden_flag, sparseFormat);
        selfatten.compute(D_SEQ, tensor_in, out);
    }
}

int main() {
    for (int sparsity = 10; sparsity <= 90; sparsity += 20) {
        inference(sparsity, Format::CSC);
    }
    return 0;
}