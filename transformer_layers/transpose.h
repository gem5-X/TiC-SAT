#pragma once

#include "util.h"

class Transpose {
public:
    static void transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) ;
};