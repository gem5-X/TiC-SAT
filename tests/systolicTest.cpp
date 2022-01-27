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
    uint32_t weights[] = {0x2010303, 0x2010303, 0x10103, 0x3030301, 0x30100, 0x102, 0x2000202, 0x30203, 0x1000002,
                          0x2000301, 0x102, 0x3030301, 0x2010201, 0x20201, 0x20200, 0x3020200, 0x301, 0x3030303,
                          0x10300, 0x2010201, 0x10302, 0x20101, 0x2030202, 0x2030201, 0x3020201, 0x3030003, 0x2020100,
                          0x1000300, 0x2000000, 0x20302, 0x1000102, 0x2000300, 0x203, 0x3030102, 0x2010300, 0x10100,
                          0x1010301, 0x303, 0x2020003, 0x3000302, 0x3020303, 0x10101, 0x3020000, 0x1030001, 0x1020002,
                          0x2030003, 0x10002, 0x3020001, 0x3030202, 0x3000102, 0x1000301, 0x2020103, 0x3000100, 0x20300,
                          0x1020202, 0x2030201, 0x10200, 0x3010201, 0x200, 0x30103, 0x3000102, 0x3030102, 0x2010302,
                          0x3000203};
    uint32_t inputArray[W_DIM* W_DIM] = {0x2000200, 0x6020401, 0x4020302, 0x4060206, 0x6050502, 0x1030402, 0x30503,
                                         0x3060503, 0x1020004, 0x3000402, 0x6020405, 0x7030500, 0x3020005, 0x1000701,
                                         0x5070705, 0x2030206, 0x7030307, 0x6060300, 0x7020002, 0x5060506, 0x4060305,
                                         0x50701, 0x5010403, 0x2010403, 0x50507, 0x5030101, 0x1000207, 0x3040205,
                                         0x1070403, 0x6010301, 0x2010703, 0x4000404, 0x3060403, 0x6050500, 0x3030301,
                                         0x40501, 0x1020002, 0x2020102, 0x4070006, 0x7020104, 0x7010000, 0x1030704,
                                         0x7000304, 0x7000501, 0x5050206, 0x30407, 0x7010000, 0x7060701, 0x6020506,
                                         0x1060707, 0x2000606, 0x5020106, 0x2050607, 0x7000506, 0x4050406, 0x4060300,
                                         0x1030300, 0x20302, 0x3040100, 0x1040300, 0x6000500, 0x5040400, 0x7040704,
                                         0x50601};
    SystolicMatrixMultiplication systolicMM;
    for (int i=0; i< W_DIM * MAX_COL; i++){
        systolicMM.loadWeights(i/MAX_COL, i%MAX_COL,  weights[i]);
    }
//    systolicMM.printWeights();
//    std::cout<<std::endl;

    for (int i=0; i< 2 *(W_DIM * MAX_COL + 2 * W_DIM -1); i++){
        std::cout<<std::hex<<systolicMM.streamInOut(i%MAX_COL, inputArray[i]) << std::endl;
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

