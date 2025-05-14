#include "transpose.h"
#include <iostream>

void Transpose::transpose(const uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) {
    uint32_t swap[ACT_PER_BUS];
    std::size_t height_tile = height / ACT_PER_BUS;
    std::size_t width_tile = width / ACT_PER_BUS;
    for (int i=0; i < height_tile; i++){
        for (int j=0; j < width_tile; j++){
            for( int k=0; k< ACT_PER_BUS; k++){
                swap[k] = input[(i * ACT_PER_BUS + k) * width_tile + j];
            }

            for ( int k=0; k< ACT_PER_BUS; k++){
                uint32_t result = 0;
                for (int s=0; s< ACT_PER_BUS; s++)
                    result |= ((swap [s] >> ((ACT_PER_BUS - 1 - k) * ACTIVATION_BITS)) & ACTIVATION_MASK)
                                 << ((ACT_PER_BUS - 1 - s) * ACTIVATION_BITS);
                output[(j * ACT_PER_BUS + k) * height_tile + i] = result;
            }
        }
    }
}

void Transpose::transpose_rearranged(uint32_t* input, uint32_t* output, std::size_t width, std::size_t height) {
    std::size_t tileRow = height / KERNEL_DIM;
    std::size_t tileCol = width / KERNEL_DIM;
    for (int i=0; i < tileRow; i++){
        for (int j=0; j < tileCol; j++){
            uint32_t * tileInputPtr = input + (j*tileRow + i) * (KERNEL_DIM * MAX_ACT_COL);
            uint32_t * tileOutputPtr = output + (i* tileCol + j) * (KERNEL_DIM * MAX_ACT_COL);
            for (int m=0; m< MAX_ACT_COL; m++){
                for (int indexInBus =0; indexInBus<ACT_PER_BUS; indexInBus++){
                    for ( int k=0; k< MAX_ACT_COL; k++){
                        uint32_t result = 0;
                        for (int s=0; s< ACT_PER_BUS; s++)
                            result |= ((*(tileInputPtr+(ACT_PER_BUS*k+s) *MAX_ACT_COL + m)>> ((ACT_PER_BUS - 1 - indexInBus)*ACTIVATION_BITS)) & ACTIVATION_MASK)
                                    << (ACT_PER_BUS - 1 - s)*ACTIVATION_BITS;
                        *(tileOutputPtr + m*ACT_PER_BUS*MAX_ACT_COL + indexInBus* MAX_ACT_COL + k)= result;
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
