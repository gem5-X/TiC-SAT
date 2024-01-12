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

#include "dev/arm/flag_sparse_memory.hh"
uint32_t flags[1000]= {0};

// Constructor.
FlagSparseMemory::FlagSparseMemory(const FlagSparseMemoryParams * p) :
	BasicPioDevice(p, p->pio_size),
	system(dynamic_cast<ArmSystem *>(p->system))
{
	warn("FlagSparseMemory instantiated.");
    
    this->pioAddr = p->pio_addr;
    this->pioSize = p->pio_size;
}

// Destructor
FlagSparseMemory::~FlagSparseMemory()
{
    
}


// FalgMem initialization.
void
FlagSparseMemory::init()
{
	BasicPioDevice::init();
	system->setFlagSparseMemory(this);
}

uint32_t FlagSparseMemory::readFlag(int idx) {

    return flags[idx];
    
}

// Read to ACM based on packet interation.
Tick
FlagSparseMemory::read(PacketPtr pkt)
{
	warn("Packet-based read access to Flag core not yet implemented.");
	return 0;
}

// Write to ACM based on packet interation.
Tick
FlagSparseMemory::write(PacketPtr pkt)
{
	warn("Packet-based write access to Flag core not yet implemented.");
	return 0;
}

// Serialize ACM.
void
FlagSparseMemory::serialize(CheckpointOut &cp) const
{
	warn("Flag serialization not yet implemented.");
}

// Unserialize ACM.
void
FlagSparseMemory::unserialize(CheckpointIn &cp)
{
	warn("Flag deserialization not yet implemented.");
    const Params *p = this->params();
    if (p->system) warn("Flag system param added");
    
    system = dynamic_cast<ArmSystem *>(p->system);
    this->init();    
}

FlagSparseMemory *
FlagSparseMemoryParams::create()
{
	return new FlagSparseMemory(this);
}
