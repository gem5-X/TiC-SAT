//
// Created by alireza on 1/13/22.
//

#ifndef FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H
#define FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H

#define KERNEL_DIM 8
#define W_DATA 4
#define MAX_COL 2
#define Nx 10
#define M 24

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class SystolicMatrixMultiplication {
private:
    uint8_t weights[KERNEL_DIM * KERNEL_DIM]{};
    uint8_t outputMemory[KERNEL_DIM * (KERNEL_DIM + 1)]{};
    uint8_t inputMemory[KERNEL_DIM * KERNEL_DIM]{};
    uint8_t inWaitingMemory[KERNEL_DIM * KERNEL_DIM]{};
    uint8_t outWaitingMemory[KERNEL_DIM * KERNEL_DIM]{};
public:
    SystolicMatrixMultiplication();
    void loadWeights(int row, int col, uint32_t  val);
    void printWeights();
    uint32_t streamInOut(int col, uint32_t val);
};

#endif //FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H
