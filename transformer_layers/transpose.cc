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

void Transpose::transpose_rearranged(uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) {
    std::size_t tileRow = height / KERNEL_DIM;
    std::size_t tileCol = width / KERNEL_DIM;
    for (int i=0; i < tileRow; i++){
        for (int j=0; j < tileCol; j++){
            uint32_t * tileInputPtr = input + (j*tileRow + i) * (KERNEL_DIM * MAX_COL);
            uint32_t * tileOutputPtr = output + (i* tileCol + j) * (KERNEL_DIM * MAX_COL);
            for (int m=0; m< MAX_COL; m++){
                for (int indexIn4 =0; indexIn4<4; indexIn4++){
                    for ( int k=0; k< MAX_COL; k++){
                        uint32_t result = 0;
                        for (int s=0; s< 4; s++)
                            result |= ((*(tileInputPtr+(4*k+s) *MAX_COL + m)>> (24 - 8*indexIn4)) & 0xFF) << (24 - 8*s);
                        *(tileOutputPtr + m*4*MAX_COL + indexIn4* MAX_COL + k)= result;
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
