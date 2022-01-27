//
// Created by alireza on 1/13/22.
//

#ifndef FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H
#define FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H

#define W_DIM 4
#define W_DATA 4
#define MAX_COL 1

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class SystolicMatrixMultiplication {
private:
    uint8_t weights[W_DIM * W_DIM]{};
    uint8_t outputMemory[W_DIM * (W_DIM+1)]{};
    uint8_t inputMemory[W_DIM * W_DIM]{};
    uint8_t waitingMemory[W_DIM * W_DIM]{};
public:
    SystolicMatrixMultiplication();
    void loadWeights(int row, int col, uint32_t  val);
    void printWeights();
    uint32_t streamInOut(int col, uint32_t val);
};

#endif //FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H
