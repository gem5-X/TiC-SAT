//
// Created by alireza on 1/13/22.
//

#ifndef FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H
#define FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H

#define W_DIM 4

class SystolicMatrixMultiplication {
private:
    uint8_t weights[W_DIM * W_DIM]{};
    uint8_t outputMemory[W_DIM * (W_DIM+1)]{};
    uint8_t inputMemory[W_DIM * W_DIM]{};
public:
    SystolicMatrixMultiplication();
    void loadWeights(int row, uint32_t  val);
    void printWeights();
    uint32_t streamInOut(uint32_t val);
};

#endif //FVLLMONTIMATRIXMULTIPLICATION_SYSTOLIC_M2M_H
