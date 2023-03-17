#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"

#define MAX_COL 4
#define KERNEL_DIM 16


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * j);
        }
        kernel[i]=result;
    }
}


void fill_sparse_weight(uint32_t* kernel, uint32_t* sparse_flag, int n_row, int n_col, int sparsity){
    auto *flag_ptr = sparse_flag;
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
                        *kernel=result;
                        kernel++;
                        #else
                        kernel[(i*KERNEL_DIM+ii)*n_col + (j*MAX_COL + jj)]=result;
                        #endif
                    }
                }
                counter32++;
            }
            else{
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

//void print_weight(uint32_t* kernel, int n_row, int n_col){
//    for (int i=0; i<n_row; i++){
//        for (int j=0; j<n_col; j++){
//            printf("%08x\t", kernel[i*n_col + j]);
//        }
//        printf("\n");
//    }
//}

void test(int sparsity_percentage){
    std::cout<<"First line" << std::endl;
#ifdef REARRANGE
    std::cout<<"Rearranged" << std::endl;
#else
    std::cout<<"TiCSAT" << std::endl;
#endif

    uint32_t tensor_in[D_SEQ * D_MODEL >> 2];
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    uint32_t out[D_SEQ*D_MODEL >> 2];

    uint32_t * weightVec[3*NUM_HEAD+3];
    uint32_t * flagVec[3*NUM_HEAD+3];

    for (int n=0; n<NUM_HEAD; n++){
        auto query_kernel = new uint32_t [D_Q* D_MODEL >> 2]();
        auto query_flag = new uint32_t [(D_Q* D_MODEL) / (32* KERNEL_DIM * MAX_COL)]();
        fill_sparse_weight(query_kernel, query_flag, D_MODEL, D_Q >> 2, sparsity_percentage);
//        print_weight(query_kernel, (D_MODEL * D_Q >> 2) / (KERNEL_DIM* MAX_COL), KERNEL_DIM *MAX_COL);
        weightVec[n*3] = query_kernel;
        flagVec[n*3] = query_flag;

        auto key_kernel = new uint32_t [ D_Q* D_MODEL >> 2]();
        auto key_flag = new uint32_t [(D_Q* D_MODEL) / (32 * KERNEL_DIM * MAX_COL)]();

        fill_sparse_weight(key_kernel, key_flag, D_MODEL, D_Q >> 2, sparsity_percentage);
        weightVec[n*3 + 1] = key_kernel;
        flagVec[n*3 + 1] = key_flag;

        auto value_kernel = new uint32_t [ D_Q* D_MODEL >> 2]();
        auto value_flag = new uint32_t [D_Q* D_MODEL / (32* KERNEL_DIM * MAX_COL)]();
        fill_sparse_weight(value_kernel, value_flag, D_MODEL, D_Q >> 2, sparsity_percentage);
        weightVec[n*3 + 2] = value_kernel;
        flagVec[n*3+2] = value_flag;
    }

    auto condense_kernel = new uint32_t [ NUM_HEAD * D_Q * D_MODEL >> 2]();
    auto condense_flag = new uint32_t [ NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_COL)]();
    fill_sparse_weight(condense_kernel, condense_flag, D_MODEL, NUM_HEAD * D_Q >> 2, sparsity_percentage);
    weightVec[NUM_HEAD*3] = condense_kernel;
    flagVec[NUM_HEAD*3] = condense_flag;

    auto ff0_kernel = new uint32_t [ D_MODEL* D_FF >> 2]();
    auto ff0_flag = new uint32_t [ D_MODEL* D_FF / (32 * KERNEL_DIM * MAX_COL)]();
    fill_sparse_weight(ff0_kernel, ff0_flag,D_MODEL, D_FF >> 2, sparsity_percentage);
    weightVec[NUM_HEAD*3+1] = ff0_kernel;
    flagVec[NUM_HEAD*3+1] = ff0_flag;

    auto ff1_kernel = new uint32_t [ D_FF* D_MODEL >> 2]();
    auto ff1_flag = new uint32_t [ D_MODEL* D_FF / (32* KERNEL_DIM * MAX_COL)]();
    fill_sparse_weight(ff1_kernel, ff1_flag, D_FF, D_MODEL >> 2, sparsity_percentage);
    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;
    flagVec[NUM_HEAD*3 + 2] = ff1_flag;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, flagVec, KERNEL_DIM, MAX_COL);
    selfatten.compute(D_SEQ, tensor_in, out);
}

int main() {
    for (int sparsity_level = 90; sparsity_level >= 0; sparsity_level-=10){
        test(sparsity_level);
    }
    return 0;
}