#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"

#define MAX_COL 2
#define KERNEL_DIM 8


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * j);
        }
        kernel[i]=result;
    }
}

void fill_sparse_weight(uint32_t* kernel, int n_row, int n_col, int sparsity){
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
                        kernel[tile_index + ii * MAX_COL + jj]=result;
                    }
                }
            }
        }
    }
}

void print_weight(uint32_t* kernel, int n_row, int n_col){
    for (int i=0; i<n_row; i++){
        for (int j=0; j<n_col; j++){
            printf("%08x\t", kernel[i*n_col + j]);
        }
        printf("\n");
    }
}

void test(int sparsity_percentage){
    std::cout<<"First line" << std::endl;
    uint32_t tensor_in[D_SEQ * D_MODEL >> 2];
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    uint32_t out[D_SEQ*D_MODEL >> 2];

    uint32_t * weightVec[3*NUM_HEAD+3];

    for (int n=0; n<NUM_HEAD; n++){
        auto query_kernel = new uint32_t [D_Q* D_MODEL >> 2]();
        fill_sparse_weight(query_kernel, D_MODEL, D_Q >> 2, sparsity_percentage);
//        print_weight(query_kernel, (D_MODEL * D_Q >> 2) / (KERNEL_DIM* MAX_COL), KERNEL_DIM *MAX_COL);
        weightVec[n*3] = query_kernel;

        auto key_kernel = new uint32_t [ D_Q* D_MODEL >> 2]();
        fill_sparse_weight(key_kernel, D_MODEL, D_Q >> 2, sparsity_percentage);
        weightVec[n*3 + 1] = key_kernel;

        auto value_kernel = new uint32_t [ D_Q* D_MODEL >> 2]();
        fill_sparse_weight(value_kernel, D_MODEL, D_Q >> 2, sparsity_percentage);
        weightVec[n*3 + 2] = value_kernel;
    }

    auto condense_kernel = new uint32_t [ NUM_HEAD * D_Q * D_MODEL >> 2]();
    fill_sparse_weight(condense_kernel, D_MODEL, NUM_HEAD * D_Q >> 2, sparsity_percentage);
    weightVec[NUM_HEAD*3] = condense_kernel;

    auto ff0_kernel = new uint32_t [ D_MODEL* D_FF >> 2]();
    fill_sparse_weight(ff0_kernel, D_MODEL, D_FF >> 2, sparsity_percentage);
    weightVec[NUM_HEAD*3+1] = ff0_kernel;

    auto ff1_kernel = new uint32_t [ D_FF* D_MODEL >> 2]();
    fill_sparse_weight(ff1_kernel, D_FF, D_MODEL >> 2, sparsity_percentage);
    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfatten.compute(D_SEQ, tensor_in, out);
}

int main() {
    for (int sparsity_level = 0; sparsity_level <= 1; sparsity_level+=5){
        test(sparsity_level);
    }
    return 0;
}