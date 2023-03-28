/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file contains layer argument data structures for specific AIMC tile
 * -enabled layers.
 *
 */

#ifndef __AIMC_LAYER_HH__
#define __AIMC_LAYER_HH__

#include <string>

#include "layer.hh"

using namespace std;

/////////////////////////////
// Layer argument structs. //
/////////////////////////////

// Base analog layers for common metadata.=, mostly.
struct ANALOG_BASE_LAYER_ARG
{
    // Metadata.
    const int aimc_h;       // Height of allocated AIMC tiles.
    const int aimc_w;       // Width of allocated AIMC tiles.

    // Constructor/destructor.
    ANALOG_BASE_LAYER_ARG(const int a_h, const int a_w) :
        aimc_h(a_h), aimc_w(a_w) {}

    ~ANALOG_BASE_LAYER_ARG() {}
};

struct ANALOG_CONV_BASE_LAYER_ARG : public ANALOG_BASE_LAYER_ARG
{
    const int kernel_c;     // Kernel channels.
    const int kernel_h;     // Kernel height.
    const int kernel_w;     // Kernel width.
    const int kernel_size;  // Flattened kernel size.
    const int n_filters;    // Number of filters/kernels.
    const int stride;       // Stride over input.
    const int padding;      // Are we padding the input?
    PaddingType padding_type;

    // Constructor/destructor.
    ANALOG_CONV_BASE_LAYER_ARG(const int k_c, const int k_h, const int k_w,
        const int n_f, const int stri, const int pad, const int a_h,
        const int a_w) :
        ANALOG_BASE_LAYER_ARG(a_h, a_w), kernel_c(k_c), kernel_h(k_h),
        kernel_w(k_w), kernel_size(k_c * k_h * k_w), n_filters(n_f),
        stride(stri), padding(pad)
    {
        padding_type = (pad) ? PADDING_SAME : PADDING_VALID;   
    }

    ~ANALOG_CONV_BASE_LAYER_ARG() {}
};

// Case 1: AIMC tile covers the entirety of the im2col transformed weights.
// Therefore, assuming the weights are pre-mapped to the AIMC tile, nothing
// needs to be written and we can exclude weights entirely.
struct ANALOG_CASE1_CONV_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_3D_IN,
    LAYER_ARG_3D_OUT, ANALOG_CONV_BASE_LAYER_ARG
{
    // Constructor/Destructor.
    ANALOG_CASE1_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_h, const int in_w,
        const int in_c, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, n_f, ((in_h - k_h + 2 * pad) / stri + 1),
            ((in_w - k_w + 2 * pad) / stri + 1), norm, act),
        ANALOG_CONV_BASE_LAYER_ARG(in_c, k_h, k_w, n_f, stri, pad, a_h, a_w)
    {
        args_type = CONV_ARGS_TYPE;
        name = "ANA1Conv";
    }

    template<typename T>
    ANALOG_CASE1_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w);

    ~ANALOG_CASE1_CONV_LAYER_ARGS() {}
};

// Case 2: AIMC tile covers the width of the im2col transformed weights, but
// not the height. Therefore, assuming the weights are pre-mapped to the AIMC
// tile, we can assume the weights matrix to actually be a residual sub-matrix.
struct ANALOG_CASE2_CONV_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_3D_IN,
    LAYER_ARG_3D_OUT, ANALOG_CONV_BASE_LAYER_ARG
{
    // Single sub-matrix of weights.
    const int kernels_h;
    const int kernels_w;
    TB_Matrix2D kernels;

    // Constructor/Destructor.
    ANALOG_CASE2_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_h, const int in_w,
        const int in_c, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, n_f, ((in_h - k_h + 2 * pad) / stri + 1),
            ((in_w - k_w + 2 * pad) / stri + 1), norm, act),
        ANALOG_CONV_BASE_LAYER_ARG(in_c, k_h, k_w, n_f, stri, pad, a_h, a_w),
        kernels_h(k_h * k_w * in_c - a_h), kernels_w(n_f)
    {
        args_type = CONV_ARGS_TYPE;
        name = "ANA2Conv";

        // Initialize kernels.
        kernels = TB_Matrix2D(k_h * k_w * in_c - a_h, n_f);
        kernels.setRandom();
    }

    template<typename T>
    ANALOG_CASE2_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w);

    ~ANALOG_CASE2_CONV_LAYER_ARGS() {}
};

// Case 3: AIMC tile covers the height of the im2col transformed weights, but
// not the width. Therefore, assuming the weights are pre-mapped to the AIMC
// tile, we can assume the weights matrix to actually be a residual sub-matrix.
struct ANALOG_CASE3_CONV_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_3D_IN,
    LAYER_ARG_3D_OUT, ANALOG_CONV_BASE_LAYER_ARG
{
    // Single sub-matrix of weights.
    const int kernels_h;
    const int kernels_w;
    TB_Matrix2D kernels;

    // Constructor/Destructor.
    ANALOG_CASE3_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_h, const int in_w,
        const int in_c, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, n_f, ((in_h - k_h + 2 * pad) / stri + 1),
            ((in_w - k_w + 2 * pad) / stri + 1), norm, act),
        ANALOG_CONV_BASE_LAYER_ARG(in_c, k_h, k_w, n_f, stri, pad, a_h, a_w),
        kernels_h(k_h * k_w * in_c), kernels_w(n_f - a_w)
    {
        args_type = CONV_ARGS_TYPE;
        name = "ANA3Conv";

        // Initialize kernels.
        kernels = TB_Matrix2D(k_h * k_w * in_c, n_f - a_w);
        kernels.setRandom();
    }

    template<typename T>
    ANALOG_CASE3_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w);

    ~ANALOG_CASE3_CONV_LAYER_ARGS() {}
};

// Case 4: AIMC tile covers neither the height nor weight of the entireim2col
// transformed weights.  Therefore, assuming the weights are pre-mapped to the
// AIMC tile, we can assume the weights matrix to actually be two residual sub
// -matrices; a bottom 'corner' underneath the tile coverage, and a right
// 'side'.
struct ANALOG_CASE4_CONV_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_3D_IN,
    LAYER_ARG_3D_OUT, ANALOG_CONV_BASE_LAYER_ARG
{
    // Sub-matrices of weights.
    const int kernels_left_bottom_h;
    const int kernels_left_bottom_w;
    const int kernels_right_h;
    const int kernels_right_w;
    TB_Matrix2D kernels_left_bottom;
    TB_Matrix2D kernels_right;

    // Constructor/Destructor.
    ANALOG_CASE4_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_h, const int in_w,
        const int in_c, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, n_f, ((in_h - k_h + 2 * pad) / stri + 1),
            ((in_w - k_w + 2 * pad) / stri + 1), norm, act),
        ANALOG_CONV_BASE_LAYER_ARG(in_c, k_h, k_w, n_f, stri, pad, a_h, a_w),
        kernels_left_bottom_h(k_h * k_w * in_c - a_h),
        kernels_left_bottom_w(a_w), kernels_right_h(k_h * k_w * in_c),
        kernels_right_w(n_f - a_w)
    {
        args_type = CONV_ARGS_TYPE;
        name = "ANA4Conv";

        // Initialize weights.
        kernels_left_bottom = TB_Matrix2D(k_h * k_w * in_c - a_h, a_w);
        kernels_right = TB_Matrix2D(k_h * k_w * in_c, n_f - a_w);
        kernels_left_bottom.setRandom();
        kernels_right.setRandom();
    }

    template<typename T>
    ANALOG_CASE4_CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act,
        const int a_h, const int a_w);

    ~ANALOG_CASE4_CONV_LAYER_ARGS() {}
};

// Case 1: AIMC tile covers the entirety of the im2col transformed weights.
// Therefore, assuming the weights are pre-mapped to the AIMC tile, nothing
// needs to be written and we can exclude weights entirely.
struct ANALOG_CASE1_FC_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_1D_IN,
    LAYER_ARG_1D_OUT, ANALOG_BASE_LAYER_ARG
{
    // Constructor/destructor.
    ANALOG_CASE1_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_s, const int out_s,
        norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_1D_IN(inf, first, in_s),
        LAYER_ARG_1D_OUT(inf, last, out_s, norm, act),
        ANALOG_BASE_LAYER_ARG(a_h, a_w)
    {
        args_type = FC_ARGS_TYPE;
        name = "ANA1Dense";
    }

    template<typename T>
    ANALOG_CASE1_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int out_s, norm_ops_t norm,
        act_ops_t act, int a_h, int a_w);

    ~ANALOG_CASE1_FC_LAYER_ARGS() {}
};

// Case 2: AIMC tile covers the width of the im2col transformed weights, but
// not the height. Therefore, assuming the weights are pre-mapped to the AIMC
// tile, we can assume the weights matrix to actually be a residual sub-matrix.
struct ANALOG_CASE2_FC_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_1D_IN,
    LAYER_ARG_1D_OUT, ANALOG_BASE_LAYER_ARG
{
    // Single sub-matrix of weights.
    const int weights_h;
    const int weights_w;
    TB_Matrix2D weights;

    // Constructor/destructor.
    ANALOG_CASE2_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_s, const int out_s,
        norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_1D_IN(inf, first, in_s),
        LAYER_ARG_1D_OUT(inf, last, out_s, norm, act),
        ANALOG_BASE_LAYER_ARG(a_h, a_w), weights_h(in_s - a_h), weights_w(out_s)
    {
        args_type = FC_ARGS_TYPE;
        name = "ANA2Dense";

        // Initialize weights.
        weights = TB_Matrix2D(in_s - a_h, out_s);
        weights.setRandom();
    }

    template<typename T>
    ANALOG_CASE2_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int out_s, norm_ops_t norm,
        act_ops_t act, int a_h, int a_w);

    ~ANALOG_CASE2_FC_LAYER_ARGS() {}
};

// Case 3: AIMC tile covers the height of the im2col transformed weights, but
// not the width. Therefore, assuming the weights are pre-mapped to the AIMC
// tile, we can assume the weights matrix to actually be a residual sub-matrix.
struct ANALOG_CASE3_FC_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_1D_IN,
    LAYER_ARG_1D_OUT, ANALOG_BASE_LAYER_ARG
{
    // Single sub-matrix of weights.
    const int weights_h;
    const int weights_w;
    TB_Matrix2D weights;

    // Constructor/destructor.
    ANALOG_CASE3_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_s, const int out_s,
        norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_1D_IN(inf, first, in_s),
        LAYER_ARG_1D_OUT(inf, last, out_s, norm, act),
        ANALOG_BASE_LAYER_ARG(a_h, a_w), weights_h(in_s), weights_w(out_s - a_w)
    {
        args_type = FC_ARGS_TYPE;
        name = "ANA3Dense";

        // Initialize weights.
        weights = TB_Matrix2D(in_s, out_s - a_w);
        weights.setRandom();
    }

    template<typename T>
    ANALOG_CASE3_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int out_s, norm_ops_t norm,
        act_ops_t act, int a_h, int a_w);

    ~ANALOG_CASE3_FC_LAYER_ARGS() {}
};

// Case 4: AIMC tile covers neither the height nor weight of the entireim2col
// transformed weights.  Therefore, assuming the weights are pre-mapped to the
// AIMC tile, we can assume the weights matrix to actually be two residual sub
// -matrices; a bottom 'corner' underneath the tile coverage, and a right
// 'side'.
struct ANALOG_CASE4_FC_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_1D_IN,
    LAYER_ARG_1D_OUT, ANALOG_BASE_LAYER_ARG
{
    // Sub-matrices of weights.
    const int weights_left_bottom_h;
    const int weights_left_bottom_w;
    const int weights_right_h;
    const int weights_right_w;
    TB_Matrix2D weights_left_bottom;
    TB_Matrix2D weights_right;

    // Constructor/destructor.
    ANALOG_CASE4_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t, const int in_s, const int out_s,
        norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_1D_IN(inf, first, in_s),
        LAYER_ARG_1D_OUT(inf, last, out_s, norm, act),
        ANALOG_BASE_LAYER_ARG(a_h, a_w), weights_left_bottom_h(in_s - a_h),
        weights_left_bottom_w(a_w), weights_right_h(in_s),
        weights_right_w(out_s - a_w)
    {
        args_type = FC_ARGS_TYPE;
        name = "ANA4Dense";

        // Initialize weights.
        weights_left_bottom = TB_Matrix2D(in_s - a_h, a_w);
        weights_right = TB_Matrix2D(in_s, out_s - a_w);
        weights_left_bottom.setRandom();
        weights_right.setRandom();
    }

    template<typename T>
    ANALOG_CASE4_FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int out_s, norm_ops_t norm,
        act_ops_t act, int a_h, int a_w);

    ~ANALOG_CASE4_FC_LAYER_ARGS() {}
};

///////////////////////////////
// Alternative constructors. //
///////////////////////////////

template<typename T>
ANALOG_CASE1_CONV_LAYER_ARGS::ANALOG_CASE1_CONV_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t,
    T & in_lyr, const int k_h, const int k_w, const int n_f, const int stri,
    const int pad, norm_ops_t norm, act_ops_t act, const int a_h,
    const int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f,
        ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    ANALOG_CONV_BASE_LAYER_ARG(in_lyr.output_c, k_h, k_w, n_f, stri, pad, a_h,
        a_w)
{
    args_type = CONV_ARGS_TYPE;
    name = "ANA1Conv";
}

template<typename T>
ANALOG_CASE2_CONV_LAYER_ARGS::ANALOG_CASE2_CONV_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int k_h,
    const int k_w, const int n_f, const int stri, const int pad,
    norm_ops_t norm, act_ops_t act, const int a_h, const int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f,
        ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    ANALOG_CONV_BASE_LAYER_ARG(in_lyr.output_c, k_h, k_w, n_f, stri, pad, a_h,
        a_w),
    kernels_h(k_h * k_w * in_lyr.output_c - a_h), kernels_w(n_f)
{
    args_type = CONV_ARGS_TYPE;
    name = "ANA2Conv";

    // Initialize kernels.
    kernels = TB_Matrix2D(k_h * k_w * in_lyr.output_c - a_h, n_f);
    kernels.setRandom();
}

template<typename T>
ANALOG_CASE3_CONV_LAYER_ARGS::ANALOG_CASE3_CONV_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int k_h,
    const int k_w, const int n_f, const int stri, const int pad,
    norm_ops_t norm, act_ops_t act, const int a_h, const int a_w) : 
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f,
        ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    ANALOG_CONV_BASE_LAYER_ARG(in_lyr.output_c, k_h, k_w, n_f, stri, pad, a_h,
        a_w),
    kernels_h(k_h * k_w * in_lyr.output_c), kernels_w(n_f - a_w)
{
    args_type = CONV_ARGS_TYPE;
    name = "ANA3Conv";

    // Initialize kernels.
    kernels = TB_Matrix2D(k_h * k_w * in_lyr.output_c, n_f - a_w);
    kernels.setRandom();
}

template<typename T>
ANALOG_CASE4_CONV_LAYER_ARGS::ANALOG_CASE4_CONV_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int k_h,
    const int k_w, const int n_f, const int stri, const int pad,
    norm_ops_t norm, act_ops_t act, const int a_h, const int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f,
        ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    ANALOG_CONV_BASE_LAYER_ARG(in_lyr.output_c, k_h, k_w, n_f, stri, pad, a_h,
        a_w),
    kernels_left_bottom_h(k_h * k_w * in_lyr.output_c - a_h),
    kernels_left_bottom_w(a_w), kernels_right_h(k_h * k_w * in_lyr.output_c),
    kernels_right_w(n_f - a_w)
{
    args_type = CONV_ARGS_TYPE;
    name = "ANA4Conv";

    // Initialize weights.
    kernels_left_bottom = TB_Matrix2D(k_h * k_w * in_lyr.output_c - a_h, a_w);
    kernels_right = TB_Matrix2D(k_h * k_w * in_lyr.output_c, n_f - a_w);
    kernels_left_bottom.setRandom();
    kernels_right.setRandom();
}

template<typename T>
ANALOG_CASE1_FC_LAYER_ARGS::ANALOG_CASE1_FC_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int out_s,
    norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_1D_IN(inf, false, in_lyr.output_size),
    LAYER_ARG_1D_OUT(inf, false, out_s, norm, act),
    ANALOG_BASE_LAYER_ARG(a_h, a_w)
{
    args_type = FC_ARGS_TYPE;
    name = "ANA1Dense";
}

template<typename T>
ANALOG_CASE2_FC_LAYER_ARGS::ANALOG_CASE2_FC_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int out_s,
    norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_1D_IN(inf, false, in_lyr.output_size),
    LAYER_ARG_1D_OUT(inf, false, out_s, norm, act),
    ANALOG_BASE_LAYER_ARG(a_h, a_w), weights_h(in_lyr.output_size - a_h),
    weights_w(out_s)
{
    args_type = FC_ARGS_TYPE;
    name = "ANA2Dense";

    // Initialize weights.
    weights = TB_Matrix2D(in_lyr.output_size - a_h, out_s);
    weights.setRandom();
}

template<typename T>
ANALOG_CASE3_FC_LAYER_ARGS::ANALOG_CASE3_FC_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int out_s,
    norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_1D_IN(inf, false, in_lyr.output_size),
    LAYER_ARG_1D_OUT(inf, false, out_s, norm, act),
    ANALOG_BASE_LAYER_ARG(a_h, a_w), weights_h(in_lyr.output_size),
    weights_w(out_s - a_w)
{
    args_type = FC_ARGS_TYPE;
    name = "ANA3Dense";

    // Initialize weights.
    weights = TB_Matrix2D(in_lyr.output_size, out_s - a_w);
    weights.setRandom();
}

template<typename T>
ANALOG_CASE4_FC_LAYER_ARGS::ANALOG_CASE4_FC_LAYER_ARGS(const int lyr,
    const int inf, const int thrd, buffer_t b_t, T & in_lyr, const int out_s,
    norm_ops_t norm, act_ops_t act, int a_h, int a_w) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_1D_IN(inf, false, in_lyr.output_size),
    LAYER_ARG_1D_OUT(inf, false, out_s, norm, act),
    ANALOG_BASE_LAYER_ARG(a_h, a_w),
    weights_left_bottom_h(in_lyr.output_size - a_h), weights_left_bottom_w(a_w),
    weights_right_h(in_lyr.output_size), weights_right_w(out_s - a_w)
{
    args_type = FC_ARGS_TYPE;
    name = "ANA4Dense";

    // Initialize weights.
    weights_left_bottom = TB_Matrix2D(in_lyr.output_size - a_h, a_w);
    weights_right = TB_Matrix2D(in_lyr.output_size, out_s - a_w);
    weights_left_bottom.setRandom();
    weights_right.setRandom();
}

///////////////
// Typedefs. //
///////////////

typedef ANALOG_CASE1_CONV_LAYER_ARGS analog_case1_conv_layer_args;
typedef ANALOG_CASE2_CONV_LAYER_ARGS analog_case2_conv_layer_args;
typedef ANALOG_CASE3_CONV_LAYER_ARGS analog_case3_conv_layer_args;
typedef ANALOG_CASE4_CONV_LAYER_ARGS analog_case4_conv_layer_args;
typedef ANALOG_CASE1_FC_LAYER_ARGS analog_case1_fc_layer_args;
typedef ANALOG_CASE2_FC_LAYER_ARGS analog_case2_fc_layer_args;
typedef ANALOG_CASE3_FC_LAYER_ARGS analog_case3_fc_layer_args;
typedef ANALOG_CASE4_FC_LAYER_ARGS analog_case4_fc_layer_args;

#endif // __AIMC_LAYER_HH__
