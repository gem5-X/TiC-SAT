//
// Created by alireza on 12/18/23.
//

#ifndef FVLLMONTITRANSFORMER_SPARSE_REP_H
#define FVLLMONTITRANSFORMER_SPARSE_REP_H
#include "util.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include "debuggerFunctions.h"


void dense2metaData(uint32_t**, int, int, uint32_t*, uint32_t*);
int dense2csc(uint32_t*, int, int, int*, int*, uint32_t**);

#endif //FVLLMONTITRANSFORMER_SPARSE_REP_H
