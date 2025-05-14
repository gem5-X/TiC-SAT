/*
 * Copyright (c) 2025 EPFL
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

#ifndef __SYSTOLIC_M2M_H__
#define __SYSTOLIC_M2M_H__

#include <cstddef>
#include <cstdint>
#include "../transformer_layers/util.h"

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class SystolicMatrixMultiplication {
  private:
    // System this ACM belongs to.
    weight_t weights[KERNEL_DIM * KERNEL_DIM]{};

    int32_t outputMemory[KERNEL_DIM * (KERNEL_DIM + 1)]{};

    activation_t inputMemory[KERNEL_DIM * KERNEL_DIM]{};

    activation_t inWaitingMemory[KERNEL_DIM * KERNEL_DIM]{};

    u_activation_t outWaitingMemory[KERNEL_DIM * KERNEL_DIM]{};

    // Used when SA_SIZE % 4 != 0. When data is streamed and cannot be fully inserted into the systolic array
    //  due to not being a multiple of 4, the extra bytes are stored in this buffer
    activation_t extra_input_buffer[4];

    // Next position in the extra_input_buffer to place the new bytes.
    uint8_t extra_input_buffer_next_position = 0;

    bool non_zero_tile = false;

  private:
    // Returns the n-th value of x. The 0-th value is the most significant one (used to unpack  uint32_t into weights and activations).
    weight_t getNthWeight(uint32_t x, uint8_t n);
    activation_t getNthActivation(uint32_t x, uint8_t n);

    void storeInput(int idx, uint32_t val);
    uint32_t retrieveOutput(int idx);

  public:
    bool loadWeights(int idx, uint32_t  val);
    uint32_t inputQueue(int idx, uint32_t  val);
    uint32_t readFlag(uint32_t val);
    void printWeights();
    uint32_t streamInOut(int idx, uint32_t val);
 };

#endif // __SYSTOLIC_M2M_H__
