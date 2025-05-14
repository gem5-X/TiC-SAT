/*
 * Copyright (c) 2020 EPFL
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Alireza Amirshahi
 *          Rafael Medina Morillas
 *          Pedro Palacios Almendros
 */
// Constructor.
#include <cstdint>
#include <iostream>
#include "systolic_m2m.h"
#include <cassert>

weight_t SystolicMatrixMultiplication::getNthWeight(uint32_t x, uint8_t n) {
  assert(n >= 0);
  assert(n < W_PER_BUS);

  return (x >> (WEIGHT_BITS * (W_PER_BUS -n-1))) & WEIGHT_MASK;
}

activation_t SystolicMatrixMultiplication::getNthActivation(uint32_t x, uint8_t n) {
  assert(n >= 0);
  assert(n < ACT_PER_BUS);

  return (x >> (ACTIVATION_BITS * (ACT_PER_BUS -n-1))) & ACTIVATION_MASK;
}

bool SystolicMatrixMultiplication::loadWeights(int idx, uint32_t val) {

  //int idx= row * KERNEL_DIM + col * W_PER_BUS;
  for (int i=0; i < W_PER_BUS; i++){
    auto currVal = (weight_t)getNthWeight(val, i);
    if (idx + i < KERNEL_DIM * KERNEL_DIM) {
      weights[idx + i] = currVal;
    }
  }

  if (val!=0)
    non_zero_tile = true;
  return non_zero_tile;
}

void SystolicMatrixMultiplication::storeInput(int idx, uint32_t val) {
  assert(KERNEL_DIM % 4 == 0);

  auto storeInputSystolicArray= [&](int in_row, int in_col, activation_t val) {
    assert(in_row >= 0);
    assert(in_row < KERNEL_DIM);

    assert(in_col >= 0);
    assert(in_col < KERNEL_DIM);

    mem2d(inWaitingMemory, KERNEL_DIM, in_row, in_col) = val; // off-diagonal of the waiting memory
  };

  // TODO: 1. First store in the systolic array input the elements from the extra_input_buffer
  // TODO: 2. Then, store in the systolic array input all remaining bytes in val
  // TODO: 3. Then, store all remaining bytes in val into the extra_input_buffer

  // TODO: Can it happen that extra_input_buffer may need to trigger a shift? How should we handle that?s

  // Split the input to an array
  for (int i=0; i < ACT_PER_BUS; i++){
    auto currVal = (activation_t)getNthActivation(val, i);
    int row_index = idx + i;

    //    std::cout << "Loaded at i=" << (KERNEL_DIM - row_index - 1) << ", value=" << std::hex << +uint8_t(currVal) << std::dec << std::endl;
    int in_row = row_index % SA_SIZE;
    int in_col = (KERNEL_DIM - row_index - 1 + SA_SIZE) % SA_SIZE;

    storeInputSystolicArray(in_row, in_col, currVal);
  }
  // print status of waiting input memory
  // std::cout << "Waiting input memory: " << std::endl;
  // for (int i = 0; i < KERNEL_DIM; i++) {
  //   for (int j = 0; j < KERNEL_DIM; j++) {
  //     std::cout << std::hex << int(mem2d(inWaitingMemory, KERNEL_DIM, i, j)) << "\t";
  //   }
  //   std::cout << std::endl;
  // }
}

uint32_t SystolicMatrixMultiplication::retrieveOutput(int idx) {
  assert(KERNEL_DIM % ACT_PER_BUS == 0);

  // Return the output
  uint32_t result = 0;
  int result_idx = (idx + ACT_PER_BUS) % SA_SIZE;
  //  std::cout << "Result idx queue: " << result_idx << std::endl;

  for (int i = 0; i < ACT_PER_BUS; i++) {
    // Construct the i-th byte of the output.

    int out_row = (KERNEL_DIM - result_idx - i - 1 + SA_SIZE) % SA_SIZE;
    int out_col = (result_idx + i) % SA_SIZE;

    assert(out_row >= 0);
    assert(out_row < KERNEL_DIM);

    assert(out_col >= 0);
    assert(out_col < KERNEL_DIM);

    result |= mem2d(outWaitingMemory, KERNEL_DIM, out_row, out_col) << (ACTIVATION_BITS * (ACT_PER_BUS - i - 1));
  }

  return result;
}

uint32_t SystolicMatrixMultiplication::inputQueue(int idx, uint32_t val) {
  storeInput(idx, val);
  return retrieveOutput(idx);
}

void SystolicMatrixMultiplication::printWeights() {
  // Save formatting state.
  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  //std::cout << std::hex << (uint32_t) inputMemory[0] << std::endl;
  std::cout << "SA weights: [";
  for (int j=0; j < KERNEL_DIM * KERNEL_DIM; j++) {
    std::cout << std::hex << (size_t)(weights[j]);

    if (j != KERNEL_DIM * KERNEL_DIM - 1) {
      if (j % KERNEL_DIM == KERNEL_DIM - 1) {
        std::cout << "; ";
      } else {
        std::cout << ", ";
      }
    }
  }
  std::cout << ']' << std::endl;

  std::cout.copyfmt(oldState);
}

uint32_t SystolicMatrixMultiplication::readFlag(uint32_t val) {
    return KERNEL_DIM;
}

uint32_t SystolicMatrixMultiplication::streamInOut(int idx, uint32_t val) {
  non_zero_tile = false;

  storeInput(idx, val);

  // Shift the waiting memory to the right for skewing
  for (int i = 0; i < KERNEL_DIM; i++) {
    mem2d(inputMemory, KERNEL_DIM, i, 0) = mem2d(inWaitingMemory, KERNEL_DIM, i, KERNEL_DIM - 1);
    for (int j= KERNEL_DIM - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
      mem2d(inWaitingMemory, KERNEL_DIM, i, j) = mem2d(inWaitingMemory, KERNEL_DIM, i, j - 1);
    }
  }

  // Multiply the input to the weight and accumulate to the output
  for (int i= KERNEL_DIM * KERNEL_DIM - 1; i >= 0 ; i--){
#if ACTIVATION_FP == 1  // Assuming weight is only FP if activation is FP
    arith_activation_t in, acc, out;
    in.bin = inputMemory[i];
    acc.bin = outputMemory[i];
#if WEIGHT_FP == 1
    arith_weight_t w;
    w.bin = weights[i];
    out.fp = in.fp * w.fp + acc.fp;
#else   // WEIGHT_FP == 0
    out.fp = in.fp * weights[i] + acc.fp;
#endif  // WEIGHT_FP
    outputMemory[i + KERNEL_DIM] = out.bin;
#else   // ACTIVATION_FP == 0
    outputMemory[i + KERNEL_DIM] = int(inputMemory[i] * weights[i]) + outputMemory[i];
#endif  // ACTIVATION_FP
  }

  // Shift the input memory to the right
  for (int i = 0; i < KERNEL_DIM; i++) {
    for (int j= KERNEL_DIM - 1; j > 0; j--){
      inputMemory[i * KERNEL_DIM + j] = inputMemory[i * KERNEL_DIM + j - 1];
    }
  }

  // Shift the outWaitingMemory because of the skew in the output
  for (int j = 0; j < KERNEL_DIM; j++) {
    for (int i= KERNEL_DIM - 1; i > 0; i--){ // TODO: shift only the right-hand triangle
      mem2d(outWaitingMemory, KERNEL_DIM, i, j) = mem2d(outWaitingMemory, KERNEL_DIM, i - 1, j);
    }
    //        std::cout << std::hex << mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) << std::endl;
    mem2d(outWaitingMemory, KERNEL_DIM, 0, j) = (u_activation_t)(mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) & ACTIVATION_MASK);
  }
  // print the status of the waiting output memory
  // std::cout << "Waiting output memory: " << std::endl;
  // for (int i = 0; i < KERNEL_DIM; i++) {
  //   for (int j = 0; j < KERNEL_DIM; j++) {
  //     std::cout << std::hex << int(mem2d(outWaitingMemory, KERNEL_DIM, i, j)) << "\t";
  //   }
  //   std::cout << std::endl;
  // }

  return retrieveOutput(idx);
}
