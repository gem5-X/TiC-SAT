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

#include "dev/arm/systolic_m2m.hh"

// Constructor.
SystolicMatrixMultiplication::SystolicMatrixMultiplication(const SystolicMatrixMultiplicationParams * p) :
	BasicPioDevice(p, p->pio_size),
	system(dynamic_cast<ArmSystem *>(p->system))
{
	warn("SMM core instantiated.");
    
    this->pioAddr = p->pio_addr;
    this->pioSize = p->pio_size;
    
    for (auto cpu : p->cpus) {
        cpus.push_back(cpu);
        tiles.push_back(new SATile());
    }

}

// Destructor
SystolicMatrixMultiplication::~SystolicMatrixMultiplication()
{
    for (auto tile : tiles){
        delete[] tile;
    }
}


// SMM initialization.
void
SystolicMatrixMultiplication::init()
{
	BasicPioDevice::init();
	system->setSystolicMatrixMultiplication(this);
}

bool SystolicMatrixMultiplication::loadWeights(int tid, int idx, uint32_t val) {

    //int idx= row * KERNEL_DIM + col * W_PER_BUS;
    for (int i=0; i < W_PER_BUS; i++){
        auto currVal = (weight_t)((val >> (WEIGHT_BITS * (W_PER_BUS -i-1))) & WEIGHT_MASK);
        tiles[tid]->weights[idx + i] = currVal;
    }

    if (val!=0)
        tiles[tid]->non_zero_tile = true;
    return tiles[tid]->non_zero_tile;
}

uint32_t SystolicMatrixMultiplication::inputQueue(int tid, int col, uint32_t val) {
    // Split the input to an array
    for (int i=0; i < ACT_PER_BUS; i++){ 
        auto currVal = (activation_t)((val >> (ACTIVATION_BITS * (ACT_PER_BUS - i -1))) & ACTIVATION_MASK);
        int row_index = (col*ACT_PER_BUS+i);
        mem2d(tiles[tid]->inWaitingMemory, KERNEL_DIM, row_index, KERNEL_DIM - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }

    // Return the output
    uint32_t result = 0;
    int result_idx = ((col+1)%MAX_ACT_COL) * ACT_PER_BUS;
    for (int i = 0; i < ACT_PER_BUS; i++) {
        result |=  mem2d(tiles[tid]->outWaitingMemory, KERNEL_DIM, KERNEL_DIM - result_idx - i - 1, result_idx + i ) << (ACTIVATION_BITS * (ACT_PER_BUS - i - 1));
    }

    return result;
}

void SystolicMatrixMultiplication::printWeights() {
    //std::cout << std::hex << (uint32_t) inputMemory[0] << std::endl;
    for (SATile * tile : tiles){
        for (int i=0; i< KERNEL_DIM * KERNEL_DIM; i++)
            std::cout << std::hex << tile->weights[i] << ", ";
        std::cout << std::endl;
    }
}

uint32_t SystolicMatrixMultiplication::readFlag(int tid, uint32_t val) {
    return KERNEL_DIM;
}

uint32_t SystolicMatrixMultiplication::streamInOut(int tid, uint32_t val) {
    tiles[tid]->non_zero_tile = false;
	int col = MAX_ACT_COL - 1;
    // Split the input to an array
    for (int i=0; i < ACT_PER_BUS; i++){
        auto currVal = (activation_t)((val >> (ACTIVATION_BITS * (ACT_PER_BUS - i -1))) & ACTIVATION_MASK);
        int row_index = (col*ACT_PER_BUS+i);
        mem2d(tiles[tid]->inWaitingMemory, KERNEL_DIM, row_index, KERNEL_DIM - row_index - 1) = currVal; // off-diagonal of the waiting memory
    }

    // Shift the waiting memory to the right for skewing
    for (int i = 0; i < KERNEL_DIM; i++) {
        mem2d(tiles[tid]->inputMemory, KERNEL_DIM, i, 0) = mem2d(tiles[tid]->inWaitingMemory, KERNEL_DIM, i, KERNEL_DIM - 1);
        for (int j= KERNEL_DIM - 1; j > 0; j--){ // TODO: shift only the right-hand triangle
            mem2d(tiles[tid]->inWaitingMemory, KERNEL_DIM, i, j) = mem2d(tiles[tid]->inWaitingMemory, KERNEL_DIM, i, j - 1);
        }
    }

    // Multiply the input to the weight and accumulate to the output
    for (int i= KERNEL_DIM * KERNEL_DIM - 1; i >= 0 ; i--){
#if ACTIVATION_FP == 1  // Assuming weight is only FP if activation is FP
        arith_activation_t in, acc, out;
        in.bin = tiles[tid]->inputMemory[i];
        acc.bin = tiles[tid]->outputMemory[i];
#if WEIGHT_FP == 1
        arith_weight_t w;
        w.bin = tiles[tid]->weights[i];
        out.fp = in.fp * w.fp + acc.fp;
#else   // WEIGHT_FP == 0
        out.fp = in.fp * tiles[tid]->weights[i] + acc.fp;
#endif  // WEIGHT_FP
        tiles[tid]->outputMemory[i + KERNEL_DIM] = out.bin;
#else   // ACTIVATION_FP == 0
        tiles[tid]->outputMemory[i + KERNEL_DIM] = int(tiles[tid]->inputMemory[i] * tiles[tid]->weights[i]) + tiles[tid]->outputMemory[i];
#endif  // ACTIVATION_FP
    }

    // Shift the input memory to the right
    for (int i = 0; i < KERNEL_DIM; i++) {
        for (int j= KERNEL_DIM - 1; j > 0; j--){
            tiles[tid]->inputMemory[i * KERNEL_DIM + j] = tiles[tid]->inputMemory[i * KERNEL_DIM + j - 1];
        }
    }

    // Shift the outWaitingMemory because of the skew in the output
    for (int j = 0; j < KERNEL_DIM; j++) {
        for (int i= KERNEL_DIM - 1; i > 0; i--){ // TODO: shift only the right-hand triangle
            mem2d(tiles[tid]->outWaitingMemory, KERNEL_DIM, i, j) = mem2d(tiles[tid]->outWaitingMemory, KERNEL_DIM, i - 1, j);
        }
//        std::cout << std::hex << mem2d(outputMemory, KERNEL_DIM, KERNEL_DIM, j) << std::endl;
        mem2d(tiles[tid]->outWaitingMemory, KERNEL_DIM, 0, j) = (u_activation_t)(mem2d(tiles[tid]->outputMemory, KERNEL_DIM, KERNEL_DIM, j) & ACTIVATION_MASK);
    }


    // Return the output
    uint32_t result = 0;
    for (int i = 0; i < ACT_PER_BUS; i++) {
        result |=  mem2d(tiles[tid]->outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ) << (ACTIVATION_BITS * (ACT_PER_BUS - i - 1));
//        std::cout << std::hex << (int) mem2d(outWaitingMemory, KERNEL_DIM, KERNEL_DIM - i - 1, i ) << ",";
    }
    return result;

}

// Read to ACM based on packet interation.
Tick
SystolicMatrixMultiplication::read(PacketPtr pkt)
{
	warn("Packet-based read access to ACM core not yet implemented.");
	return 0;
}

// Write to ACM based on packet interation.
Tick
SystolicMatrixMultiplication::write(PacketPtr pkt)
{
	warn("Packet-based write access to ACM core not yet implemented.");
	return 0;
}

// Serialize ACM.
void
SystolicMatrixMultiplication::serialize(CheckpointOut &cp) const
{
	warn("ACM serialization not yet implemented.");
    for (auto tile : tiles) {
        if (tile)
            delete[] tile;
    }
}

// Unserialize ACM.
void
SystolicMatrixMultiplication::unserialize(CheckpointIn &cp)
{
	warn("ACM deserialization not yet implemented.");
    const Params *p = this->params();
    if (p->system) warn("SA system param added");
    
    system = dynamic_cast<ArmSystem *>(p->system);
    this->init();
    for (auto cpu : p->cpus) {
        if (cpu) warn("Adding CPU SA tile.");
        cpus.push_back(cpu);
        tiles.push_back(new SATile());
    }
    
}

AddrRangeList
SystolicMatrixMultiplication::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(RangeSize(pioAddr, pioSize));
    return ranges;
}

SystolicMatrixMultiplication *
SystolicMatrixMultiplicationParams::create()
{
	return new SystolicMatrixMultiplication(this);
}
