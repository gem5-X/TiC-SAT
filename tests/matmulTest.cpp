//
// Created by alireza on 11/25/21.
//
#include <cstdlib>
#include <iostream>
#include "../kernels/matmul.h"

#define D_SEQ 512
#define D_MODEL 512
#define D_Q 64

#define NUM_BITS 8
//#define PRINT_MAT

void fill_kernel(int* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++)
        kernel[i]=(((rand() % (1<<NUM_BITS))  - (1<<(NUM_BITS-1))));
}

void printMatrix(int* kernel, int row, int col, std::string name){
    std::cout<<name<<" Matrix:" <<std::endl;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
            std::cout<<kernel[i*col+j]<<"\t\t";
        std::cout<<std::endl;
    }

}

void test(){
    int input_mat[D_SEQ * D_MODEL];  // "D_SEQ" is the number of rows and "D_MODEL" is the number of columns.
    fill_kernel(input_mat, D_SEQ* D_MODEL);
    #ifdef PRINT_MAT
    printMatrix(input_mat, D_SEQ, D_MODEL, "Input");
    #endif

    int weight_kernel[D_MODEL * D_Q]; // "D_MODEL" is the number of rows and "D_Q" is the number of columns.
    fill_kernel(weight_kernel, D_MODEL * D_Q);
    #ifdef PRINT_MAT
    printMatrix(weight_kernel, D_MODEL, D_Q, "Weight");
    #endif

    int output_mat[D_SEQ * D_Q];

    lh::MatMul<int> matMul;
    matMul.compute(D_SEQ, input_mat, output_mat, weight_kernel, D_MODEL, D_Q);
    #ifdef PRINT_MAT
    printMatrix(output_mat, D_SEQ, D_Q, "Output");
    #endif
}

int main() {
    test();
    return 0;
}
