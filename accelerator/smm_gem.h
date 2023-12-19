//
// Created by alireza on 3/3/22.
//

#ifndef FVLLMONTITRANSFORMER_SMM_GEM_H
#define FVLLMONTITRANSFORMER_SMM_GEM_H

#include <cstddef>
#include <cstdint>

void conventionalCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);

void tiledCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);

void tiledL1Compute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                  std::size_t input_size_, std::size_t output_size_);

void simdCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                    std::size_t input_size_, std::size_t output_size_);


void smmCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void smmComputeRearranged(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t *weights,
                          uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse,
                          const uint32_t* hidden_flag);

void simdCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                          uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void simdComputeRearranged(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                 uint32_t *flag, std::size_t input_size_, std::size_t output_size_, bool sparse);

void smmComputeEigen(std::size_t seq_len, const int8_t *input, int8_t *output, int8_t *weights,
                     std::size_t input_size_, std::size_t output_size_);


void print_arr(uint32_t* array, int n, int p);
//uint64_t smmParamWrite(uint64_t rm, uint64_t rn, uint64_t ra, int tid);
//uint64_t smmQueue(uint64_t rm, uint64_t rn, int tid) ;
//uint64_t smmStream(uint64_t rn, int tid);

#endif //FVLLMONTITRANSFORMER_SMM_GEM_H
