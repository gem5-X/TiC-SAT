//
// Created by alireza on 2/7/24.
//

#ifndef FVLLMONTITRANSFORMER_DATAFUNCTIONS_H
#define FVLLMONTITRANSFORMER_DATAFUNCTIONS_H

#include <cstdint>
#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#ifndef RELOAD_WEIGHT
#include <filesystem>
#endif


void fill_kernel(uint32_t* kernel, int kernel_size);
void print_binary(uint32_t value);
void fill_sparse_weight(uint32_t * kernel, uint32_t* sparse_flag, int n_row, int n_col, int sparsity);
void remove_zero_tiles(uint32_t*& kernel, int n_row, int n_col);
void load_kernel_from_file(std::vector<uint32_t> &kernel, int n_row, int n_col, const char *filename) ;
void saveWeight(int n_head, int qkv, int size, uint32_t *array, int sparsity_level, const std::string &dir_name);
void loadWeight(int n_head, int qkv, int size, uint32_t * array,  int sparsity_level, const std::string &dir_name,
                const uint32_t* hidden_flag);
void append_flags(uint32_t* new_flags, int new_flags_count);

#endif //FVLLMONTITRANSFORMER_DATAFUNCTIONS_H
