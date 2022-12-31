#include "transpose.h"
#include <iostream>

void Transpose::transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) {
    uint32_t swap[4];
    std::size_t height_tile = height >> 2;
    std::size_t width_tile = width >> 2;
    for (int i=0; i < height_tile; i++){
        for (int j=0; j < width_tile; j++){
            for( int k=0; k< 4; k++){
                swap[k] = input[(i * 4 + k) * width_tile + j];
            }

            for ( int k=0; k< 4; k++){
                uint32_t result = 0;
                for (int s=0; s< 4; s++)
                    result |= ((swap [s] >> (24 - 8*k)) & 0xFF) << (24 - 8*s);
                output[(j * 4 + k) * height_tile + i] = result;
            }
        }
    }
}
