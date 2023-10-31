#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include <fstream>
#ifndef RELOAD_WEIGHT
#include <filesystem>
#endif
#include "transformer_layers/debuggerFunctions.h"

#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/4)


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * j);
        }
        kernel[i]=result;
    }
}

void print_binary(uint32_t value) {
    for (int i = 31; i >= 0; i--) {
        std::cout << ((value >> i) & 1);
    }
    std::cout<< std::endl;
}

void fill_sparse_weight(uint32_t * kernel, uint32_t* sparse_flag, int n_row, int n_col, int sparsity){
    auto *flag_ptr = sparse_flag;
    uint32_t *kernel_ptr = kernel;
    int counter32 = 0;
    uint32_t flag32 = 0;
    for (int i=0; i<n_row/KERNEL_DIM; i++){
        for (int j=0; j<n_col/MAX_COL; j++){
            if (rand() % 100 >= sparsity){
                int tile_index = (i * (n_col/MAX_COL) + j) * KERNEL_DIM * MAX_COL;
                for (int ii=0; ii<KERNEL_DIM; ii++){
                    for (int jj=0; jj<MAX_COL; jj++){
                        uint32_t result = 0;
                        for (int k=0; k<4; k++){
                            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * k);
                        }
                        #ifdef REARRANGE
//                        kernel[tile_index + ii * MAX_COL + jj]=result;
                        *kernel_ptr=result;
                        kernel_ptr++;
                        #else
                        kernel[(i*KERNEL_DIM+ii)*n_col + (j*MAX_COL + jj)]=result;
                        #endif
                    }
                }
                counter32++;
            }
            else{
                for (int ii=0; ii<KERNEL_DIM*MAX_COL; ii++){
                    *kernel_ptr = 0x0;
                    kernel_ptr++;
                }
                flag32 |= 0x00000001;
                counter32++;
            }

            if (counter32 == 32){
                *flag_ptr = flag32;
                flag_ptr++;
                counter32=0;
                flag32 = 0;
            }
            else{
                flag32 = flag32 << 1;
            }
        }
    }
}


void remove_zero_tiles(uint32_t*& kernel, int n_row, int n_col) {
    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;
    int counter = 0;

    for (int i = 0; i < n_row / KERNEL_DIM; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (i * (n_col / MAX_COL) + j) * KERNEL_DIM * MAX_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < KERNEL_DIM; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                for (int ii = 0; ii < KERNEL_DIM; ii++) {
                    for (int jj = 0; jj < MAX_COL; jj++) {
                        *new_kernel ++ = kernel[tile_index + ii * MAX_COL + jj];
                    }
                }
            }
            else{
                counter ++;
            }
        }
    }
    kernel = new_kernel_ptr;
}

void load_kernel_from_file(std::vector<uint32_t> &kernel, int n_row, int n_col, const char *filename) {
    std::ifstream file(filename, std::ios::binary);
    kernel.resize(n_row * n_col);
    file.read(reinterpret_cast<char *>(kernel.data()), n_row * n_col * sizeof(uint32_t));
    file.close();
}

void saveWeight(int n_head, int qkv, int size, uint32_t *array, int sparsity_level, const std::string &dir_name) {
    // Write the kernel array to file
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + "_S" + std::to_string(sparsity_level) + ".bin";
    std::ofstream fout(filename);
    if (fout.is_open()) {
        for (int i = 0; i < size; i++) {
            fout << array[i] << " ";
        }
        fout.close();
    }
}

void loadWeight(int n_head, int qkv, int size, uint32_t * array,  int sparsity_level, const std::string &dir_name,
                const uint32_t* hidden_flag){
    bool hidden_flag_check = (hidden_flag != nullptr);
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + "_S" + std::to_string(sparsity_level) + ".bin";
    std::ifstream fin(filename);
    if (fin.is_open()) {
        for (int i = 0; i < size; i++) {
            fin >> array[i];
            if (hidden_flag_check)
                if (array[i] == *hidden_flag){
                    std::cout << *hidden_flag << " is found in the arrays!" << std::endl;
                    exit(404);
                }
        }
        fin.close();
    }
    else{
        std::cout << filename + " Not loaded" << std::endl;
    }

}

void append_flags(uint32_t* new_flags, int new_flags_count) {
    std::ofstream outfile("flags_generated.h", std::ios::app);
    for (int i = 0; i < new_flags_count; i++) {
        outfile << new_flags[i] << ", ";
    }
    outfile.close();
}


void test(int sparsity_percentage){
    uint32_t hidden_flag = 0xAAAAAAAA;
    std::cout<<"First line" << std::endl;
#ifdef REARRANGE
    std::cout<<"Rearranged" << std::endl;
#else
    std::cout<<"TiCSAT" << std::endl;
#endif

    std::string dir_name = "/home/alireza/CLionProjects/FvllMontiTransformer/data16";

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

#ifndef REARRANGE
        uint32_t tensorInRowWise[D_SEQ * D_MODEL >> 2];
        blockWise2RowWise(tensor_in, tensorInRowWise, D_SEQ, D_MODEL >> 2);
        tensor_in = tensorInRowWise;
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
        #ifdef REARRANGE
        remove_zero_tiles(const_cast<uint32_t*&>(query_kernel), D_MODEL, D_Q >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(key_kernel), D_MODEL, D_Q >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(value_kernel), D_MODEL, D_Q >> 2);
        #endif
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

#ifndef REARRANGE
    uint32_t* queryRowWise = new uint32_t [D_MODEL * D_Q >> 2];
    blockWise2RowWise(query_kernel, queryRowWise, D_MODEL, D_Q >> 2);
    query_kernel = queryRowWise;
    uint32_t* keyRowWise = new uint32_t [D_MODEL * D_Q >> 2];
    blockWise2RowWise(key_kernel, keyRowWise, D_MODEL, D_Q >> 2);
    key_kernel = keyRowWise;
    uint32_t* valueRowWise = new uint32_t [D_MODEL * D_Q >> 2];
    blockWise2RowWise(value_kernel, valueRowWise, D_MODEL, D_Q >> 2);
    value_kernel = valueRowWise;
#endif
//        uint32_t tensorInRowWise[D_SEQ * D_MODEL >> 2];
//        uint32_t queryWeightRowWise[D_MODEL * D_Q >> 2];
//        uint32_t queryRowWiseFromBlock[D_SEQ * D_Q >> 2];
//        auto queryResultRowWise = new uint32_t [D_SEQ* D_Q >> 2]();
//        auto queryBlockWise = new uint32_t [D_SEQ* D_Q >> 2]();
//
//        blockWise2RowWise(value_kernel, queryWeightRowWise, D_MODEL, D_Q >> 2);
//        blockWise2RowWise(tensor_in, tensorInRowWise, D_SEQ, D_MODEL >> 2);
//
////        conventionalCompute(D_SEQ, tensorInRowWise,queryRowWise, queryWeightRowWise, D_MODEL, D_Q);
//        smmCompute(D_SEQ, tensorInRowWise,queryResultRowWise, queryWeightRowWise, nullptr, D_MODEL, D_Q, false);
//        smmComputeRearranged(D_SEQ, tensor_in,queryBlockWise, value_kernel, nullptr, D_MODEL, D_Q, false);
//
//        blockWise2RowWise(queryBlockWise, queryRowWiseFromBlock, D_SEQ, D_Q >> 2);
//
//        std::cout<< "Input " << std::endl;
//        print_weight(tensor_in, D_SEQ, D_MODEL>>2);
//        std::cout<< "tensor in" << std::endl;
//        print_weight(tensorInRowWise, D_SEQ, D_MODEL>>2);
//        std::cout<< "Block Wise" << std::endl;
//        print_weight(queryBlockWise, D_SEQ, D_Q>>2);
//        std::cout<< "Row Wise" << std::endl;
//        print_weight(queryResultRowWise, D_SEQ, D_Q >> 2);
//
//        uint32_t error = 0;
//        for (int elem =0; elem < D_SEQ*D_Q >>2 ; elem++){
//            error += (queryResultRowWise[elem] - queryRowWiseFromBlock[elem]);
//        }
//        std::cout<< "ERROR : " << error << std::endl;

//        print_weight(query_kernel, (D_MODEL * D_Q >> 2) / (KERNEL_DIM* MAX_COL), KERNEL_DIM *MAX_COL);
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
    #ifdef REARRANGE
        remove_zero_tiles(const_cast<uint32_t*&>(condense_kernel), NUM_HEAD * D_Q, D_MODEL >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(ff0_kernel), D_MODEL, D_FF >> 2);
        remove_zero_tiles(const_cast<uint32_t*&>(ff1_kernel), D_FF, D_MODEL >> 2);
    #endif
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

    #ifndef REARRANGE
        uint32_t* condenseRowWise = new uint32_t [NUM_HEAD * D_Q * D_MODEL >> 2];
        blockWise2RowWise(condense_kernel, condenseRowWise, NUM_HEAD * D_Q, D_MODEL >> 2);
        condense_kernel = condenseRowWise;
        uint32_t* ff0RowWise = new uint32_t [D_MODEL * D_FF >> 2];
        blockWise2RowWise(ff0_kernel, ff0RowWise, D_MODEL, D_FF >> 2);
        ff0_kernel = ff0RowWise;
        uint32_t* ff1RowWise = new uint32_t [D_FF * D_MODEL >> 2];
        blockWise2RowWise(ff1_kernel, ff1RowWise, D_FF, D_MODEL >> 2);
        ff1_kernel = ff1RowWise;
    #endif

    weightVec[NUM_HEAD*3] = condense_kernel;
    flagVec[NUM_HEAD*3] = condense_flag;

    weightVec[NUM_HEAD*3+1] = ff0_kernel;
    flagVec[NUM_HEAD*3+1] = ff0_flag;

    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;
    flagVec[NUM_HEAD*3 + 2] = ff1_flag;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, flagVec, KERNEL_DIM, MAX_COL);
    selfatten.compute(D_SEQ, tensor_in, out);
}

int main() {
    for (int sparsity_level = 30; sparsity_level <= 95; sparsity_level+=30){
        test(sparsity_level);
    }
    return 0;
}