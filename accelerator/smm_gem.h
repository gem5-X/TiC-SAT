//
// Created by alireza on 3/3/22.
//

#ifndef FVLLMONTITRANSFORMER_SMM_GEM_H
#define FVLLMONTITRANSFORMER_SMM_GEM_H

#include <cstddef>
#include <cstdint>

#include "../transformer_layers/util.h"

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
#ifdef DEVELOP

// Store the compressed weight value rn at the flattened position of the systolic-array rm.
//
// rm is used as the single index to access the flattened uint8_t[] weight array. The
// four weights will be stored at uint8_t[rm], uint8_t[rm+1], uint8_t[rm+2], uint8_t[rm+3].
//
// The weights array has size uint8_t[SA_SIZE*SA_SIZE] and is accessed as weights[row * SA_SIZE + col]
//
// Return value can be ignored (non-zero tile)
bool smmParamWrite(int rm, uint32_t rn, int tid=0);
uint32_t smmQueue(int rm, uint32_t rn, int tid=0);
uint32_t smmStream(int rm, uint32_t rn, int tid=0);
uint64_t smmReadFlag(uint64_t val, uint64_t tid=0);


void printWeights(int tid);

#else

uint64_t smmParamWrite(uint64_t rm, uint64_t rn, uint64_t tid);
uint64_t smmQueue(uint64_t rm, uint64_t rn, uint64_t tid);
uint64_t smmStream(uint64_t rn, uint64_t tid);
uint64_t smmReadFlag(uint64_t val, uint64_t tid);


void printWeights(int tid);

#endif

void addActIn32(uint32_t &memory, uint32_t &systolicResult);

#endif //FVLLMONTITRANSFORMER_SMM_GEM_H
