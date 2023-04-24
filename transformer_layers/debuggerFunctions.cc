//
// Created by alireza on 4/24/23.
//

#include "debuggerFunctions.h"

void print_weight(uint32_t* kernel, int n_row, int n_col){
    for (int i=0; i< n_row; i++){
        for (int j=0; j<n_col; j++){
            printf("0x%08x,\t", kernel[i*n_col + j]);
        }
        printf("\n");
    }
}

void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col){
    uint32_t* initialRowWise = rowWise;
    for (int col=0; col<n_col/2; col++){
        rowWise = initialRowWise + col * 2;
        for (int row=0; row < n_row; row++){
            for (int i=0; i<2; i++){
                *(rowWise + i) = *(blockWise+ i);
            }
            blockWise += 2;
            rowWise += n_col;
        }
    }
}
