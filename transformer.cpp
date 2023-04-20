#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include <fstream>
#include <filesystem>

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

void loadWeight(int n_head, int qkv, int size, uint32_t * array,  int sparsity_level, const std::string &dir_name){
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + "_S" + std::to_string(sparsity_level) + ".bin";
    std::ifstream fin(filename);
    if (fin.is_open()) {
        for (int i = 0; i < size; i++) {
            fin >> array[i];
        }
        fin.close();
    }
    else{
        std::cout << filename + " Not loaded" << std::endl;
    }

}

void print_weight(uint32_t* kernel, int n_row, int n_col){
    for (int i=0; i<n_row; i++){
        for (int j=0; j<2; j++){
            printf("%08x\t", kernel[i*n_col + j]);
        }
        printf("\n");
    }
}


void test(int sparsity_percentage){
    std::cout<<"First line" << std::endl;
#ifdef REARRANGE
    std::cout<<"Rearranged" << std::endl;
#else
    std::cout<<"TiCSAT" << std::endl;
#endif

    std::string dir_name = "data";

    uint32_t tensor_in[D_SEQ * D_MODEL >> 2];
    #ifdef RELOAD_WEIGHT
        loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, 0, dir_name);
    #else
        fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
        saveWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, 0, dir_name);
    #endif


    uint32_t out[D_SEQ*D_MODEL >> 2];

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
        loadWeight(n, 0, head_qkv_size, query_kernel, sparsity_percentage, dir_name);
        loadWeight(n, 1, head_qkv_size, key_kernel, sparsity_percentage, dir_name);
        loadWeight(n, 2, head_qkv_size, value_kernel, sparsity_percentage, dir_name);
        loadWeight(n, 10, head_flag_size, query_flag, sparsity_percentage, dir_name);
        loadWeight(n, 11, head_flag_size, key_flag, sparsity_percentage, dir_name);
        loadWeight(n, 12, head_flag_size, value_flag, sparsity_percentage, dir_name);
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
#endif
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

        loadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, sparsity_percentage, dir_name);
        loadWeight(n, 1, D_MODEL* D_FF >> 2, ff0_kernel, sparsity_percentage, dir_name);
        loadWeight(n, 2, D_MODEL* D_FF >> 2, ff1_kernel, sparsity_percentage, dir_name);

        loadWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL), condense_flag, sparsity_percentage, dir_name);
        loadWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff0_flag, sparsity_percentage, dir_name);
        loadWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff1_flag, sparsity_percentage, dir_name);
    #else
        fill_sparse_weight(condense_kernel, condense_flag, D_MODEL, NUM_HEAD * D_Q >> 2, sparsity_percentage);
        fill_sparse_weight(ff0_kernel, ff0_flag,D_MODEL, D_FF >> 2, sparsity_percentage);
        fill_sparse_weight(ff1_kernel, ff1_flag, D_FF, D_MODEL >> 2, sparsity_percentage);

        int n = -1; // n=-1 means that we are not saving/loading a head
        saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, sparsity_percentage, dir_name);
        saveWeight(n, 1, D_MODEL* D_FF >> 2, ff0_kernel, sparsity_percentage, dir_name);
        saveWeight(n, 2, D_MODEL* D_FF >> 2, ff1_kernel, sparsity_percentage, dir_name);

        saveWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL), condense_flag, sparsity_percentage, dir_name);
        saveWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff0_flag, sparsity_percentage, dir_name);
        saveWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL), ff1_flag, sparsity_percentage, dir_name);
    #endif

    weightVec[NUM_HEAD*3] = condense_kernel;
    flagVec[NUM_HEAD*3] = condense_flag;

    weightVec[NUM_HEAD*3+1] = ff0_kernel;
    flagVec[NUM_HEAD*3+1] = ff0_flag;

    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;
    flagVec[NUM_HEAD*3 + 2] = ff1_flag;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, flagVec, KERNEL_DIM, MAX_COL);
    selfatten.compute(D_SEQ, tensor_in, out);
    print_weight(out, 5, D_MODEL/4);
}

int main() {
//    for (int sparsity_level = 0; sparsity_level <= 8; sparsity_level+=10){
    int sparsity_level = 0;
    test(sparsity_level);
//    }
    return 0;
}