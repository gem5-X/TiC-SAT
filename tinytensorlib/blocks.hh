/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file implements structs, functions, and other utilities for specific
 * block functionality common with very deep NNs. Blocks are usually comprised
 * of multiple layers and can have a repetition factor involved which
 * propogates the network. The repetition of blocks is refered by C library
 * vectors.
 *
 * Supported blocks so far include:
 * - Bottleneck (MobileNetV2)
 * - Residual (SSDResNet34)
 *
 */

#ifndef __BLOCKS_HH__
#define __BLOCKS_HH__

#include <vector>

using namespace std;

//////////////////////////////////
// MobileNetV2 Bottleneck Block //
//////////////////////////////////

typedef struct BOTTLENECK_BLOCK
{
    // Metadata.
    const int n; // Repetitions of this block.
    const int t; // Expansion factor.
    const int c; // Output channels.
    const int s; // Depthwise layer stride.

    // Layer vectors.
    vector<conv_layer_args> bn_conv1;
    vector<dwconv_layer_args> bn_dwconv;
    vector<conv_layer_args> bn_conv2;
    vector<end_residual_layer_args> bn_res;

    // Constructor.
    template<typename T>
    BOTTLENECK_BLOCK(int base_layer_num, int T_x, int bn_n, int bn_t, int bn_c,
        int bn_s,
#if defined (AIMC)
        bool u_a1, bool u_a2,
#endif
        T & in_lyr) : n(bn_n), t(bn_t), c(bn_c), s(bn_s)
    {
        int layer_num = base_layer_num;

        // Initialize first part of the block (n = 1).
        bn_conv1.push_back(conv_layer_args(layer_num++, T_x,
            SINGLE_BUFFER_TYPE, in_lyr, 1, 1, bn_c*bn_t, 1,
#if defined (AIMC)
            u_a1,
#endif
            0, NO_NORM_TYPE, RELU6_ACT_TYPE));

        // Add depth-wise and point-wise convolutions to block.
        bn_dwconv.push_back(dwconv_layer_args(layer_num++, T_x,
            SINGLE_BUFFER_TYPE, bn_conv1[0], 3, 3, bn_c*bn_t, bn_s, 1,
            NO_NORM_TYPE, RELU6_ACT_TYPE));

        bn_conv2.push_back(conv_layer_args(layer_num++, T_x,
            SINGLE_BUFFER_TYPE, bn_dwconv[0], 1, 1, bn_c, 1,
#if defined (AIMC)
            u_a2,
#endif
            0, NO_NORM_TYPE, RELU_ACT_TYPE));

        // Initialize internal argument structs for n > 1.
        for (int i = 1; i < bn_n; i++) {
            // If this is the first conv of the whole block connect to the last
            // conv.
            if (bn_s == 1 && i > 1) {
                bn_conv1.push_back(conv_layer_args(layer_num++, T_x,
                    SINGLE_BUFFER_TYPE, bn_res[i-2], 1, 1, bn_c*bn_t, 1,
#if defined (AIMC)
                    u_a1,
#endif
                    0, NO_NORM_TYPE, RELU6_ACT_TYPE));
            } else {
                bn_conv1.push_back(conv_layer_args(layer_num++, T_x,
                    SINGLE_BUFFER_TYPE, bn_conv2[i-1], 1, 1, bn_c*bn_t, 1,
#if defined (AIMC)
                    u_a1,
#endif
                    0, NO_NORM_TYPE, RELU6_ACT_TYPE));
            }

            // Add depth-wise and point-wise convolutions to block.
            bn_dwconv.push_back(dwconv_layer_args(layer_num++, T_x,
                SINGLE_BUFFER_TYPE, bn_conv1[i], 3, 3, bn_c*bn_t, 1, 1,
                NO_NORM_TYPE, RELU6_ACT_TYPE));

            bn_conv2.push_back(conv_layer_args(layer_num++, T_x,
                SINGLE_BUFFER_TYPE, bn_dwconv[i], 1, 1, bn_c, 1,
#if defined (AIMC)
                u_a2,
#endif
                0, NO_NORM_TYPE, RELU_ACT_TYPE));

            // If depth-wise stride is 1, add residual.
            if (bn_s == 1) {
                bn_res.push_back(end_residual_layer_args(layer_num++, T_x, 0,
                    false, false, SINGLE_BUFFER_TYPE, bn_conv2[i].output_h,
                    bn_conv2[i].output_w, bn_conv2[i].output_c, NO_NORM_TYPE,
                    NO_ACT_TYPE));
            }
        }
    }

    ~BOTTLENECK_BLOCK() {}
} bottleneck_block;

// Bottleneck block equivalent of "connectLayers" out the block.
template<typename T> inline void
connectBottleneckBlock(bottleneck_block & bn_blk, T & in_lyr)
{
    connectLayers(bn_blk.bn_conv1[0], in_lyr);
    return;
}

// Bottleneck block equivalent of "connectLayers" out the block.
inline void
connectBottleneckBlock(bottleneck_block & bn_blk, bottleneck_block & in_lyr)
{
    if (in_lyr.s == 1 && in_lyr.n > 1) {
        connectLayers(bn_blk.bn_conv1[0], in_lyr.bn_res[in_lyr.n-2]);
    } else {
        connectLayers(bn_blk.bn_conv1[0], in_lyr.bn_conv2[in_lyr.n-1]);
    }
    return;
}

// Bottleneck block equivalent of "connectLayers" within the block.
inline void
connectIntraBottleneckLayers(bottleneck_block & bn_blk)
{
    connectLayers(bn_blk.bn_dwconv[0], bn_blk.bn_conv1[0]);
    connectLayers(bn_blk.bn_conv2[0], bn_blk.bn_dwconv[0]);

    for (int i = 1; i < bn_blk.n; i++) {
        if (bn_blk.s == 1 && i > 1) {
            connectLayers(bn_blk.bn_conv1[i], bn_blk.bn_res[i-2]);
        } else {
            connectLayers(bn_blk.bn_conv1[i], bn_blk.bn_conv2[i-1]);
        }

        connectLayers(bn_blk.bn_dwconv[i], bn_blk.bn_conv1[i]);
        connectLayers(bn_blk.bn_conv2[i], bn_blk.bn_dwconv[i]);

        if (bn_blk.s == 1 && i == 1) {
            connectResidual(bn_blk.bn_res[i-1], bn_blk.bn_conv2[i-1]);
            connectLayers(bn_blk.bn_res[i-1], bn_blk.bn_conv2[i]);
        } else if (bn_blk.s == 1) {
            connectResidual(bn_blk.bn_res[i-1], bn_blk.bn_res[i-2]);
            connectLayers(bn_blk.bn_res[i-1], bn_blk.bn_conv2[i]);
        }
    }

    return;
}

inline void
cleanupBottleneckMemory(bottleneck_block & bn_blk)
{
    for (int i = 0; i < bn_blk.n; i++) {
        delete[] bn_blk.bn_conv1[i].input;
        delete[] bn_blk.bn_dwconv[i].input;
        delete[] bn_blk.bn_conv2[i].input;
        if (bn_blk.s == 1 && i > 0) {
            delete[] bn_blk.bn_res[i-1].input;
        }
    }

    return;
}

// Bottleneck block equivalent of "doLayer".
inline void
doBottleneckBlock(bottleneck_block & bn_blk)
{
    for (int i = 0; i < bn_blk.n; i++) {
        doLayer(bn_blk.bn_conv1[i]);
        doLayer(bn_blk.bn_dwconv[i]);
        doLayer(bn_blk.bn_conv2[i]);
        if (bn_blk.s == 1 && i > 0) {
            doLayer(bn_blk.bn_res[i-1]);
        }
    }

    return;
}

/////////////////////////////////
// SSDResNet34 Residual Block. //
/////////////////////////////////

typedef struct RESIDUAL_BLOCK
{
    // Metadata.

    // Layer vectors.

    // Constructor.
    RESIDUAL_BLOCK()
    {

    }

    ~RESIDUAL_BLOCK() {}
} residual_block;

// Residual block equivalent of "connectLayers" out the block.
template<typename T> inline void
connectResidualBlock(residual_block & bn_blk, T & in_lyr)
{
    return;
}

// Residual block equivalent of "connectLayers" out the block.
inline void
connectResidualBlock(residual_block & bn_blk, residual_block & in_lyr)
{
    return;
}

// Residual block equivalent of "connectLayers" within the block.
inline void
connectIntraResidualLayers(bottleneck_block & bn_blk)
{
    return;
}

inline void
cleanupResidualMemory(residual_block & bn_blk)
{
    return;
}

// Residual block equivalent of "doLayer".
inline void
doResidualBlock(residual_block & bn_blk)
{
    return;
}

#endif // __BLOCKS_HH__
