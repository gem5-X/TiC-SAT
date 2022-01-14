//
// Created by alireza on 1/13/22.
//

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"

//void fill_kernel(int* kernel, int kernel_size){
//    for(int i=0; i<kernel_size; i++)
//        kernel[i]=(((rand() % (1<<NUM_BITS))  - (1<<(NUM_BITS-1))));
//}
//
//void printMatrix(int* kernel, int row, int col, std::string name){
//    std::cout<<name<<" Matrix:" <<std::endl;
//    for(int i=0;i<row;i++)
//    {
//        for(int j=0;j<col;j++)
//            std::cout<<kernel[i*col+j]<<"\t\t";
//        std::cout<<std::endl;
//    }
//
//}

void test(){
    uint32_t weights[] = {0x01020103, 0x01020103, 0x01020103, 0x01020103};
//    uint32_t inputArray[] = {0x00010203, 0x01020304, 0x00010001, 0x01010101};
    uint32_t inputArray[] = {0x00000001, 0x00000102, 0x00030402, 0x01010101, 0x04020300, 0x01000000, 0x02000000, 0, 0, 0, 0};
    SystolicMatrixMultiplication systolicMM;
    for (int i=0; i< W_DIM; i++){
        systolicMM.loadWeights(i, weights[i]);
    }
    systolicMM.printWeights();
    for(uint32_t in : inputArray){
        std::cout<< systolicMM.streamInOut(in) << std::endl;
    }

//    int input_mat[D_SEQ * D_MODEL];  // "D_SEQ" is the number of rows and "D_MODEL" is the number of columns.
//    fill_kernel(input_mat, D_SEQ* D_MODEL);
//    #ifdef PRINT_MAT
//    printMatrix(input_mat, D_SEQ, D_MODEL, "Input");
//    #endif
//
//    int weight_kernel[D_MODEL * D_Q]; // "D_MODEL" is the number of rows and "D_Q" is the number of columns.
//    fill_kernel(weight_kernel, D_MODEL * D_Q);
//    #ifdef PRINT_MAT
//    printMatrix(weight_kernel, D_MODEL, D_Q, "Weight");
//    #endif
//
//    int output_mat[D_SEQ * D_Q];
//
//    lh::MatMul<int> matMul;
//    matMul.compute(D_SEQ, input_mat, output_mat, weight_kernel, D_MODEL, D_Q);
//    #ifdef PRINT_MAT
//    printMatrix(output_mat, D_SEQ, D_Q, "Output");
//    #endif
}

int main() {
    test();
    return 0;
}

