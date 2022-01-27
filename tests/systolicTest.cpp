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
    uint32_t weights[] = {0x3030303, 0x2020300, 0x3020002, 0x1020102, 0x20301, 0x10301, 0x1010003, 0x2010003, 0x10103,
                          0x1030101, 0x1000203, 0x3000201, 0x1030003, 0x2010301, 0x2010303, 0x0};
    uint32_t inputArray[] = {0x2000a00, 0x6020409, 0xc0a030a, 0x40e020e, 0x6050d0a, 0x90b0402, 0x30503, 0xb06050b,
                             0x102080c, 0xb080c0a, 0x6020405, 0x7030508, 0x30a0805, 0x1080f01, 0xd070705, 0xa0b0a06
                             , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    SystolicMatrixMultiplication systolicMM;
    for (int i=0; i< W_DIM * MAX_COL; i++){
        systolicMM.loadWeights(i/MAX_COL, i%MAX_COL,  weights[i]);
    }
    systolicMM.printWeights();
//    std::cout<<std::endl;

    for (int i=0; i< 2 * (W_DIM * MAX_COL + W_DIM -1); i++){
        std::cout<<  systolicMM.streamInOut(i%MAX_COL, inputArray[i])<<std::endl;
//        systolicMM.streamInOut(i%MAX_COL, inputArray[i]);
//        std::cout<<std::endl;
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

