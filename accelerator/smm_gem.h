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

void smmComputeRWMA(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                    std::size_t input_size_, std::size_t output_size_);

void smmComputeBWMA(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t *weights,
                    std::size_t input_size_, std::size_t output_size_);

void simdComputeRWMA(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                     std::size_t input_size_, std::size_t output_size_);

void simdComputeBWMA(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                     std::size_t input_size_, std::size_t output_size_);


#endif //FVLLMONTITRANSFORMER_SMM_GEM_H
