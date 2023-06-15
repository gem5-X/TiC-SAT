//
// Created by alireza on 4/24/23.
//

#ifndef FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#define FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#include "util.h"
#include <iostream>
#include <fstream>
#include <cstdint>


void print_weight(uint32_t* kernel, int n_row, int n_col);
void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col);
void rowWise2BlockWise(const uint32_t* rowWise, uint32_t* blockWise, int n_row, int n_col);
void write_weight_to_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col);
void read_weight_from_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col);

#endif //FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
