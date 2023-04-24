//
// Created by alireza on 4/24/23.
//

#ifndef FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#define FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#include "util.h"


void print_weight(uint32_t* kernel, int n_row, int n_col);
void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col);


#endif //FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
