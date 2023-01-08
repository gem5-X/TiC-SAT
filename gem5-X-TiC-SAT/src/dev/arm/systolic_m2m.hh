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

#define KERNEL_DIM 8
#define W_DATA 4
#define MAX_COL 2

#define mem2d(data,data_len,row,col)   data[((row)*(data_len))+(col)]

class ArmSystem;
class BaseCPU;

class SystolicMatrixMultiplication : public BasicPioDevice {
  private:
    // System this ACM belongs to.
    ArmSystem * system;
    
    int8_t weights[KERNEL_DIM * KERNEL_DIM]{};

    int32_t outputMemory[KERNEL_DIM * (KERNEL_DIM + 1)]{};

    int8_t inputMemory[KERNEL_DIM * KERNEL_DIM]{};

    int8_t inWaitingMemory[KERNEL_DIM * KERNEL_DIM]{};

    uint8_t outWaitingMemory[KERNEL_DIM * KERNEL_DIM]{};
    
  public:
    SystolicMatrixMultiplication(const SystolicMatrixMultiplicationParams * p);
    void loadWeights(int row, int col, uint32_t  val);
    uint32_t inputQueue(int col, uint32_t  val);
    void printWeights();
    uint32_t streamInOut(uint32_t val);
    void init() override;

    // Required by SimObject.
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;
    void serialize(CheckpointOut &cp) const override;
    void unserialize(CheckpointIn &cp) override;
};

#endif // __SYSTOLIC_M2M_H__
