//
// Created by alireza on 3/3/22.
//

#ifndef FVLLMONTITRANSFORMER_SMM_GEM_H
#define FVLLMONTITRANSFORMER_SMM_GEM_H
#include "gemmini.h"

void conventionalCompute(size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         size_t input_size_, size_t output_size_);

void tiledCompute(size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         size_t input_size_, size_t output_size_);

void tiledL1Compute(size_t seq_len, const int8_t * input, int8_t * output, int8_t *weight,
                    size_t input_size_, size_t output_size_, acc_scale_t scale, acc_scale_t bert_scale);


void smmCompute(size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                size_t input_size_, size_t output_size_);

void print_arr(uint32_t* array, int n, int p);

#endif //FVLLMONTITRANSFORMER_SMM_GEM_H
