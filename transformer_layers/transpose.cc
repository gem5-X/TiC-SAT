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

void Transpose::transpose_rearranged(uint32_t* input, uint32_t* output, std::size_t width, std::size_t height,
                                     std::size_t kernelSize, std::size_t maxCol) {
    std::size_t tileRow = height / kernelSize;
    std::size_t tileCol = width / kernelSize;
    for (int i=0; i < tileRow; i++){
        for (int j=0; j < tileCol; j++){
            uint32_t * tileInputPtr = input + (j*tileRow + i) * (kernelSize * maxCol);
            uint32_t * tileOutputPtr = output + (i* tileCol + j) * (kernelSize * maxCol);
            for (int m=0; m< maxCol; m++){
                for (int indexIn4 =0; indexIn4<4; indexIn4++){
                    for ( int k=0; k< maxCol; k++){
                        uint32_t result = 0;
                        for (int s=0; s< 4; s++)
                            result |= ((*(tileInputPtr+(4*k+s) *maxCol + m)>> (24 - 8*indexIn4)) & 0xFF) << (24 - 8*s);
                        *(tileOutputPtr + m*4*maxCol + indexIn4* maxCol + k)= result;
                    }
                }
            }
        }
    }
}

void Transpose::multihead_transpose(const uint32_t* input, uint32_t* output, std::size_t seq_len,
                                    std::size_t head_hidden_size, std::size_t num_head) {
    const uint32_t * initial_input = input;
    for (int i=0; i < seq_len; i++){
        for (int n=0; n< num_head; n++){
            input = initial_input + i*head_hidden_size + n*seq_len*head_hidden_size;
            for (int j=0; j < head_hidden_size; j++){
                *output++ = *input++;
            }
        }
    }
}
