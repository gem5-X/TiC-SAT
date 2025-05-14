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

#include "arch/arm/system.hh"
#include "dev/io_device.hh"
#include "debug/SMM.hh"
#include "mem/packet.hh"
#include "mem/packet_access.hh"
#include "params/SystolicMatrixMultiplication.hh"

#include <vector>

#define KERNEL_DIM      4   // Dimension of systolic array tile
#define ACTIVATION_BITS 32   // Number of bits for activation
#define WEIGHT_BITS     8   // Number of bits for weight
#define ACTIVATION_FP   1   // Wether the activation is floating point or not
#define WEIGHT_FP       0   // Wether the weight is floating point or not

#define CEILING_DIV(x,y)    (((x) + (y) - 1) / (y))
#define BUS_WIDTH           32                                      // Width of the bus interfacing CPU and SA
#define ACT_PER_BUS         (BUS_WIDTH / ACTIVATION_BITS)           // Number of activations per bus
#define W_PER_BUS           (BUS_WIDTH / WEIGHT_BITS)               // Number of weights per bus
#define ACTIVATION_MASK     ((1UL << ACTIVATION_BITS) - 1)          // Bit-mask for activation
#define WEIGHT_MASK         ((1UL << WEIGHT_BITS) - 1)              // Bit-mask for weight
#define MAX_ACT_COL         CEILING_DIV(KERNEL_DIM, ACT_PER_BUS)    // Number of 32-bit words to hold all activations in a column
#define MAX_W_COL           CEILING_DIV(KERNEL_DIM, W_PER_BUS)      // Number of 32-bit words to hold all weights in a column

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

#if ACTIVATION_BITS == 8
    typedef int8_t activation_t;
    typedef uint8_t u_activation_t;
#elif ACTIVATION_BITS == 16
    typedef int16_t activation_t;
    typedef uint16_t u_activation_t;
#elif ACTIVATION_BITS == 32
    typedef int32_t activation_t;
    typedef uint32_t u_activation_t;
#if ACTIVATION_FP == 1
    typedef union
    {
        float   fp;
        int32_t bin;
    } arith_activation_t;
#endif
#endif

#if WEIGHT_BITS == 8
    typedef int8_t weight_t;
#elif WEIGHT_BITS == 16
    typedef int16_t weight_t;
#elif WEIGHT_BITS == 32
    typedef int32_t weight_t;
#if WEIGHT_FP == 1 && ACTIVATION_FP == 1    // Assume that if weights are FP32, activations are also FP32
    typedef union
    {
        float   fp;
        int32_t bin;
    } arith_weight_t;
#endif
#endif

class ArmSystem;
class BaseCPU;

struct SATile {
    SATile():
    weights(new weight_t[KERNEL_DIM * KERNEL_DIM]),
    inputMemory(new activation_t[KERNEL_DIM * KERNEL_DIM]),
    outputMemory(new int32_t[KERNEL_DIM * (KERNEL_DIM+1)]),
    inWaitingMemory(new activation_t[KERNEL_DIM * KERNEL_DIM]),
    outWaitingMemory(new u_activation_t[KERNEL_DIM * KERNEL_DIM])
    {
        for (int i = 0; i < KERNEL_DIM*KERNEL_DIM; i++) {
            inputMemory[i] = 0;
            weights[i] = 0;
            inWaitingMemory[i] = 0;
            outWaitingMemory[i] = 0;
            outputMemory[i] = 0;
        }

        for (int i = 0; i < KERNEL_DIM; i++) {
            outputMemory[(KERNEL_DIM* KERNEL_DIM)+i] = 0;
        }
    }
    
    weight_t * weights;
    activation_t * inputMemory;
    int32_t * outputMemory;
    
    activation_t * inWaitingMemory;
    u_activation_t * outWaitingMemory;
    bool non_zero_tile = false;
};

class SystolicMatrixMultiplication : public BasicPioDevice {
  private:
      
      std::vector<BaseCPU *> cpus;
      
      std::vector<SATile *> tiles;
      
    // System this ACM belongs to.
    ArmSystem * system;
    
    
    
  public:
      typedef SystolicMatrixMultiplicationParams Params;
      const Params * params() const {
          return dynamic_cast<const Params *>(_params);
      }
    SystolicMatrixMultiplication(const Params * p);
    ~SystolicMatrixMultiplication();
    void init() override;
    
    bool loadWeights(int tid, int idx, uint32_t  val);
    uint32_t inputQueue(int tid, int col, uint32_t  val);
    void printWeights();
    uint32_t readFlag(int tid, uint32_t val);
    uint32_t streamInOut(int tid, uint32_t val);
    

    // Required by SimObject.
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;
    void serialize(CheckpointOut &cp) const override;
    void unserialize(CheckpointIn &cp) override;
    
    AddrRangeList getAddrRanges() const override;
    
    
};

#endif // __SYSTOLIC_M2M_H__
