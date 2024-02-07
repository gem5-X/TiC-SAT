#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include "transformer_layers/debuggerFunctions.h"
#include "transformer_layers/dataFunctions.h"

#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/4)


void test(int sparsity_percentage){
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
        loadWeight(n, 0, head_qkv_size, query_kernel, sparsity_percentage, dir_name, &hidden_flag);
        loadWeight(n, 1, head_qkv_size, key_kernel, sparsity_percentage, dir_name, &hidden_flag);
        loadWeight(n, 2, head_qkv_size, value_kernel, sparsity_percentage, dir_name, &hidden_flag);
        loadWeight(n, 10, head_flag_size, query_flag, sparsity_percentage, dir_name, nullptr);
        loadWeight(n, 11, head_flag_size, key_flag, sparsity_percentage, dir_name, nullptr);
        loadWeight(n, 12, head_flag_size, value_flag, sparsity_percentage, dir_name, nullptr);
    #ifdef ZERO_FREE
        remove_zero_tiles(const_cast<uint32_t*&>(query_kernel), D_MODEL, D_Q >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(key_kernel), D_MODEL, D_Q >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(value_kernel), D_MODEL, D_Q >> 2);
    #endif
#else
        fill_sparse_weight(query_kernel, query_flag, D_MODEL, D_Q >> 2, sparsity_percentage);
        fill_sparse_weight(key_kernel, key_flag, D_MODEL, D_Q >> 2, sparsity_percentage);
        fill_sparse_weight(value_kernel, value_flag, D_MODEL, D_Q >> 2, sparsity_percentage);
        if (!std::filesystem::exists(dir_name)) {
            std::filesystem::create_directory(dir_name);
        }

        saveWeight(n, 0, head_qkv_size, query_kernel, sparsity_percentage, dir_name);
        saveWeight(n, 1, head_qkv_size, key_kernel, sparsity_percentage, dir_name);
        saveWeight(n, 2, head_qkv_size, value_kernel, sparsity_percentage, dir_name);

        saveWeight(n, 10, head_flag_size, query_flag, sparsity_percentage, dir_name);
        saveWeight(n, 11, head_flag_size, key_flag, sparsity_percentage, dir_name);
        saveWeight(n, 12, head_flag_size, value_flag, sparsity_percentage, dir_name);
        append_flags(query_flag, head_flag_size);
        append_flags(key_flag, head_flag_size);
        append_flags(value_flag, head_flag_size);
#endif

#ifdef HIDDEN_FLAG
interleave_hidden_flag_zero_free(const_cast<uint32_t*&>(query_kernel), D_MODEL, D_Q >> 2, hidden_flag);
interleave_hidden_flag_zero_free(const_cast<uint32_t*&>(key_kernel), D_MODEL, D_Q >> 2, hidden_flag);
interleave_hidden_flag_zero_free(const_cast<uint32_t*&>(value_kernel), D_MODEL, D_Q >> 2, hidden_flag);
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

        loadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, sparsity_percentage, dir_name, &hidden_flag);
        loadWeight(n, 1, D_MODEL* D_FF >> 2, ff0_kernel, sparsity_percentage, dir_name, &hidden_flag);
        loadWeight(n, 2, D_MODEL* D_FF >> 2, ff1_kernel, sparsity_percentage, dir_name, &hidden_flag);

        loadWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL), condense_flag, sparsity_percentage, dir_name,
                   nullptr);
        loadWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff0_flag, sparsity_percentage, dir_name, nullptr);
        loadWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff1_flag, sparsity_percentage, dir_name, nullptr);

    #ifdef ZERO_FREE
        remove_zero_tiles(const_cast<uint32_t*&>(condense_kernel), NUM_HEAD * D_Q, D_MODEL >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(ff0_kernel), D_MODEL, D_FF >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(ff1_kernel), D_FF, D_MODEL >> 2);
    #endif
    #else
        fill_sparse_weight(condense_kernel, condense_flag, D_MODEL, NUM_HEAD * D_Q >> 2, sparsity_percentage);
        fill_sparse_weight(ff0_kernel, ff0_flag, D_MODEL, D_FF >> 2, sparsity_percentage);
        fill_sparse_weight(ff1_kernel, ff1_flag, D_FF, D_MODEL >> 2, sparsity_percentage);

        int n = -1; // n=-1 means that we are not saving/loading a head
        saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, sparsity_percentage, dir_name);
        saveWeight(n, 1, D_MODEL* D_FF >> 2, ff0_kernel, sparsity_percentage, dir_name);
        saveWeight(n, 2, D_MODEL* D_FF >> 2, ff1_kernel, sparsity_percentage, dir_name);

        saveWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL), condense_flag, sparsity_percentage, dir_name);
        saveWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff0_flag, sparsity_percentage, dir_name);
        saveWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff1_flag, sparsity_percentage, dir_name);

        append_flags(condense_flag, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL));
        append_flags(ff0_flag, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL));
        append_flags(ff1_flag, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL));

        outfile.open("flags_generated.h", std::ios::app);
        outfile << "};" << std::endl;
        outfile.close();
    #endif

#ifdef HIDDEN_FLAG
        interleave_hidden_flag_zero_free(const_cast<uint32_t*&>(condense_kernel), NUM_HEAD * D_Q, D_MODEL >> 2, hidden_flag);
        interleave_hidden_flag_zero_free(const_cast<uint32_t*&>(ff0_kernel), D_MODEL, D_FF >> 2, hidden_flag);
        interleave_hidden_flag_zero_free(const_cast<uint32_t*&>(ff1_kernel), D_FF, D_MODEL >> 2, hidden_flag);
#endif

    weightVec[NUM_HEAD*3] = condense_kernel;
    flagVec[NUM_HEAD*3] = condense_flag;

    weightVec[NUM_HEAD*3+1] = ff0_kernel;
    flagVec[NUM_HEAD*3+1] = ff0_flag;

    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;
    flagVec[NUM_HEAD*3 + 2] = ff1_flag;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, flagVec, KERNEL_DIM, MAX_COL,
                               &hidden_flag);
    selfatten.compute(D_SEQ, tensor_in, out);
}

int main() {
    for (int sparsity_level = 30; sparsity_level <= 95; sparsity_level+=30){
        test(sparsity_level);
    }
    return 0;
}