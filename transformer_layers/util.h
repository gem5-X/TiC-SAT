#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>

#ifndef SA_SIZE // Force the user to define all the parameters
#define SA_SIZE         4
#define ACTIVATION_BITS 32   // Number of bits for activation
#define WEIGHT_BITS     8   // Number of bits for weight
#define ACTIVATION_FP   1   // Wether the activation is floating point or not
#define WEIGHT_FP       0   // Wether the weight is floating point or not
#endif

#define KERNEL_DIM  SA_SIZE // Dimension of systolic array tile

#define CEILING_DIV(x,y)    (((x) + (y) - 1) / (y))
#define BUS_WIDTH           32                                      // Width of the bus interfacing CPU and SA
#define ACT_PER_BUS         (BUS_WIDTH / ACTIVATION_BITS)           // Number of activations per bus
#define W_PER_BUS           (BUS_WIDTH / WEIGHT_BITS)               // Number of weights per bus
#define ACTIVATION_MASK     ((1UL << ACTIVATION_BITS) - 1)          // Bit-mask for activation
#define WEIGHT_MASK         ((1UL << WEIGHT_BITS) - 1)              // Bit-mask for weight
#define MAX_ACT_COL         CEILING_DIV(KERNEL_DIM, ACT_PER_BUS)    // Number of 32-bit words to hold all activations in a column
#define MAX_W_COL           CEILING_DIV(KERNEL_DIM, W_PER_BUS)      // Number of 32-bit words to hold all weights in a column

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
