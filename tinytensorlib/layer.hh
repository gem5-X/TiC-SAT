/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file contains layer argument data structures.
 *
 */

#ifndef __LAYER_HH__
#define __LAYER_HH__

#include <string>
#include "tinytensorlib.hh"

using namespace std;

///////////////////////////
// Forward declarations. //
///////////////////////////

struct LAYER_ARG_BASE;
struct CONV_LAYER_ARGS;
struct FC_LAYER_ARGS;
struct POOL_LAYER_ARGS;
struct FLATTEN_LAYER_ARGS;
struct END_RESIDUAL_LAYER_ARGS;
struct DWCONV_LAYER_ARGS;

/////////////////////////////
// Operation enumerations. //
/////////////////////////////

enum args_t
{
    BASE_ARGS_TYPE,
    CONV_ARGS_TYPE,
    FC_ARGS_TYPE,
    POOL_ARGS_TYPE,
    FLATTEN_ARGS_TYPE,
    END_RESIDUAL_ARGS_TYPE,
    DWCONV_ARGS_TYPE
};

enum pool_ops_t
{
    NO_POOL_TYPE,
    MAX_POOL_TYPE,
    AVG_POOL_TYPE
};

enum norm_ops_t
{
    NO_NORM_TYPE,
    BATCH_NORM_TYPE,
    LRN_NORM_TYPE
};

enum act_ops_t
{
    NO_ACT_TYPE,
    RELU_ACT_TYPE,
    SIGMOID_ACT_TYPE,
    SOFTMAX_ACT_TYPE,
    RELU6_ACT_TYPE
};

enum buffer_t
{
    SINGLE_BUFFER_TYPE,
    PING_PONG_BUFFER_TYPE
};

///////////////////////
// Helper functions. //
///////////////////////

// Helper functions to allocate and initialize all input or output data
// structures.  Inputs generate with random data while output generate with
// zeroed data.
inline void
generateIODataStructures(TB_Vector * target, const int T_x, const int in_s,
    bool setRandom)
{
    for (int i = 0; i < T_x; i++) {
        target[i] = TB_Vector(in_s);
        if (setRandom) {
            target[i].setRandom();
        } else {
            target[i].setZero();
        }
    }

    return;
}

inline void
generateIODataStructures(TB_Matrix3D * target, const int T_x, const int in_c,
    const int in_h, const int in_w, bool setRandom)
{
    for (int i = 0; i < T_x; i++) {
        target[i] = TB_Matrix3D(in_c, in_h, in_w);
        if (setRandom) {
            target[i].setRandom();
        } else {
            target[i].setZero();
        }
    }

    return;
}

/////////////////////////////
// Layer argument structs. //
/////////////////////////////

// Base struct for layer arguments, multithreading variables.
typedef struct LAYER_ARG_BASE
{
    // Metadata.
    args_t args_type;
    const buffer_t buffer_type;
    const int layer_n;
    const int T_x; // Number of inferences.
    const int thread_n;
    const bool isFirstLayer;
    const bool isLastLayer;
    int cond_var_prv_lyr_idx;
    int cond_var_nxt_lyr_idx;
    int cnt_mutex_prv_lyr_idx;
    int cnt_mutex_nxt_lyr_idx;
    string name;
    uint8_t pong_idx; // Select ping/pong if using this kind of buffer.

    // Constructor.
    LAYER_ARG_BASE(const int lyr, const int inf, const int thrd,
        const bool first, const bool last, const buffer_t b_t) :
        args_type(BASE_ARGS_TYPE), buffer_type(b_t), layer_n(lyr), T_x(inf),
        thread_n(thrd), isFirstLayer(first), isLastLayer(last),
        cond_var_prv_lyr_idx(0), cond_var_nxt_lyr_idx(0),
        cnt_mutex_prv_lyr_idx(0), cnt_mutex_nxt_lyr_idx(0), name(""),
        pong_idx(0)
    {}

    ~LAYER_ARG_BASE() {}
} layer_arg_base;

// Struct to hold 1-D (vector) input, associated metadata.
struct LAYER_ARG_1D_IN
{
    // Metadata, data structures.
    const int input_size;
    TB_Vector * input;    // Used to point to the input being used.

    // Constructor -- note inference count is held in base layer arg struct.
    LAYER_ARG_1D_IN(const int inf, bool first, const int in_s) :
        input_size(in_s)
    {
        // Initialize inputs if first layer.
        if (first) {
            input = new TB_Vector[inf];
            generateIODataStructures(input, inf, in_s, true);
        }
    }

    ~LAYER_ARG_1D_IN() {}
};

// Struct to hold 1-D (vector) output, associated metadata.
struct LAYER_ARG_1D_OUT
{
    // Metadata and data structures.
    const int output_size;
    TB_Vector * output;   // Used to point to the output being used.
    const norm_ops_t normalization;
    const act_ops_t activation;

    // Constructor -- note inference count is held in base layer arg struct.
    LAYER_ARG_1D_OUT(const int inf, bool last, const int out_s,
        const norm_ops_t norm, const act_ops_t act) :
        output_size(out_s), normalization(norm), activation(act)
    {
        // Initialize output if last layer.
        if (last) {
            output = new TB_Vector[inf];
            generateIODataStructures(output, inf, out_s, false);
        }
    }

    ~LAYER_ARG_1D_OUT() {}
};

// Struct to hold 3-D (matrix) input and associated metadata.
struct LAYER_ARG_3D_IN
{
    // Metadata.
    const int input_c;
    const int input_h;
    const int input_w;
    const int input_size;

    // Data structures, with ping-pong index to prevent buffer blocking.
    TB_Matrix3D * input;  // Used to point to the input being used.

    // Constructor -- note inference count is held in base layer arg struct.
    LAYER_ARG_3D_IN(const int inf, bool first, const int in_c,
        const int in_h, const int in_w) :
        input_c(in_c), input_h(in_h), input_w(in_w),
        input_size(in_c * in_h * in_w)
    {
        // Initialize input if first layer.
        if (first) {
            input = new TB_Matrix3D[inf];
            generateIODataStructures(input, inf, in_c, in_h, in_w, true);
        }
    }

    ~LAYER_ARG_3D_IN() {}
};

// Struct to hold 3-D (matrix) output and associated metadata.
struct LAYER_ARG_3D_OUT
{
    // Metadata.
    const int output_c;
    const int output_h;
    const int output_w;
    const int output_size;

    // Data structures, with ping-pong index to prevent buffer blocking.
    TB_Matrix3D * output; // Used to point to the output being used.

    // If the layer will use normalization or activation functions.
    const norm_ops_t normalization;
    const act_ops_t activation;

    // Constructor -- note inference count is held in base layer arg struct.
    LAYER_ARG_3D_OUT(const int inf, bool last, const int out_c, const int out_h,
        const int out_w, const norm_ops_t norm, const act_ops_t act) :
        output_c(out_c), output_h(out_h), output_w(out_w),
        output_size(out_c * out_h * out_w), normalization(norm), activation(act)
    {
        // Initialize inputs/output if first/last layer.
        if (last) {
            output = new TB_Matrix3D[inf];
            generateIODataStructures(output, inf, out_c, out_h, out_w, false);
        }
    }

    ~LAYER_ARG_3D_OUT() {}
};

struct CONV_LAYER_ARGS : 
    public LAYER_ARG_BASE, LAYER_ARG_3D_IN, LAYER_ARG_3D_OUT
{
    // Convolution variables.
    const int kernel_h;     // Kernel height.
    const int kernel_w;     // Kernel width.
    const int kernel_c;     // Kernel channels.
    const int kernel_size;  // Flattened kernel size.
    const int n_filters;    // Number of filters/kernels.
    const int stride;       // Stride over input.
    const int padding;      // Are we padding the input?
#if defined (SA)
    bool use_sa;    // Are we using analog tiles?
    int sa_size;       // Height of allocated AIMC tiles.
#endif
    PaddingType padding_type;

    // im2col transformed kernels.
    TB_Matrix2D weights;

    // Constructor.
    CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd, bool first,
        bool last, buffer_t b_t, const int in_h, const int in_w,
        const int in_c, const int k_h, const int k_w, const int n_f,
        const int stri,
#if defined (SA)
        bool u_a, int a_h,
#endif
        const int pad, norm_ops_t norm, act_ops_t act) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, n_f, ((in_h - k_h + 2 * pad) / stri + 1),
            ((in_w - k_w + 2 * pad) / stri + 1), norm, act),
        kernel_h(k_h), kernel_w(k_w), kernel_c(in_c),
        kernel_size(k_w * k_h * in_c), n_filters(n_f), stride(stri),
        padding(pad)
#if defined (SA)
        , use_sa(u_a), sa_size(a_h)
#endif
    {
        args_type = CONV_ARGS_TYPE;
        name = "Conv";
        padding_type = (pad) ? PADDING_SAME : PADDING_VALID;

        // Initialize weights.
        weights = TB_Matrix2D(k_h * k_w * in_c, n_f);
        weights.setRandom();
    }

    template<typename T>
    CONV_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t,
        T & in_lyr, const int k_h, const int k_w,
        const int n_f, const int stri,
#if defined (AIMC)
        bool u_a,
#endif
        const int pad, norm_ops_t norm, act_ops_t act);

    template<typename T>
    CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd, buffer_t b_t,
        T & in_lyr, const int k_h, const int k_w,
        const int n_f, const int stri,
#if defined (AIMC)
        bool u_a,
#endif
        const int pad, norm_ops_t norm, act_ops_t act);

    // Destructor.
    ~CONV_LAYER_ARGS() {}
};

struct FC_LAYER_ARGS : public LAYER_ARG_BASE, LAYER_ARG_1D_IN, LAYER_ARG_1D_OUT
{
    // Fully connected variables.
    const int weights_h;    // Weights matrix height.
    const int weights_w;    // Weights matrix width.
#if defined (AIMC)
    bool use_sa;    // Are we using analog tiles?
    int sa_size;       // Height of allocated AIMC tiles.
#endif
    TB_Matrix2D weights;    // Weights matrix.

    // Constructor and destructor.
    FC_LAYER_ARGS(const int lyr, const int inf, const int thrd, bool first,
        bool last,  buffer_t b_t, const int in_s,
#if defined (AIMC)
        bool u_a, int a_h,
#endif
        const int out_s, norm_ops_t norm, act_ops_t act) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_1D_IN(inf, first, in_s),
        LAYER_ARG_1D_OUT(inf, last, out_s, norm, act),
        weights_h(in_s), weights_w(out_s)
#if defined (AIMC)
        , use_sa(u_a), sa_size(a_h)
#endif
    {
        args_type = FC_ARGS_TYPE;
        name = "Dense";

        // Initialize weights.
        weights = TB_Matrix2D(in_s, out_s);
        weights.setRandom();
    }

    template<typename T>
    FC_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t,
        T & in_lyr,
#if defined (AIMC)
        bool u_a,
#endif
        const int out_s, norm_ops_t norm, act_ops_t act);    

    template<typename T>
    FC_LAYER_ARGS(const int lyr, const int inf, const int thrd, buffer_t b_t,
        T & in_lyr,
#if defined (AIMC)
        bool u_a,
#endif
        const int out_s, norm_ops_t norm, act_ops_t act);

    ~FC_LAYER_ARGS() {}
};

struct POOL_LAYER_ARGS :
    public LAYER_ARG_BASE, LAYER_ARG_3D_IN, LAYER_ARG_3D_OUT
{
    // Fully connected variables.
    const int pool_h;       // Pool height.
    const int pool_w;       // Pool width.
    const int pool_f;       // Pooling factor.
    const int stride;       // Pool stride.
    pool_ops_t pool_type;      // What kind of pooling is this layer?

    // Constructor and destructor.
    POOL_LAYER_ARGS(const int lyr, const int inf, const int thrd, bool first,
        bool last,  buffer_t b_t,
        const int in_h, const int in_w, const int in_c,
        const int p_h, const int p_w, const int p_f, const int stri,
        pool_ops_t p_type, norm_ops_t norm, act_ops_t act) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, in_c, in_h / p_h, in_w / p_w, norm, act),
        pool_h(p_h), pool_w(p_w), pool_f(p_f), stride(stri), pool_type(p_type)
    {
        args_type = POOL_ARGS_TYPE;
        name = "Pool";
    }

    template<typename T>
    POOL_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t, T & in_lyr,
        const int p_d, const int p_f, const int stri, pool_ops_t p_type,
        norm_ops_t norm, act_ops_t act);

    template<typename T>
    POOL_LAYER_ARGS(const int lyr, const int inf, const int thrd, buffer_t b_t,
        T & in_lyr, const int p_d, const int p_f, const int stri,
        pool_ops_t p_type, norm_ops_t norm, act_ops_t act);

    ~POOL_LAYER_ARGS() {}
};

struct FLATTEN_LAYER_ARGS :
    public LAYER_ARG_BASE, LAYER_ARG_3D_IN, LAYER_ARG_1D_OUT
{
    // Constructor and destructor.
    FLATTEN_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last,  buffer_t b_t,
        const int in_h, const int in_w, const int in_c,
        norm_ops_t norm, act_ops_t act) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_1D_OUT(inf, last, in_c * in_h * in_w, norm, act)
    {
        args_type = FLATTEN_ARGS_TYPE;
        name = "Flatten";
    }

    template<typename T>
    FLATTEN_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t, T & in_lyr);

    template<typename T>
    FLATTEN_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr);

    ~FLATTEN_LAYER_ARGS() {}
};

// Addition layer for residual block.
struct END_RESIDUAL_LAYER_ARGS :
    public LAYER_ARG_BASE, LAYER_ARG_3D_IN, LAYER_ARG_3D_OUT
{
    // Used to point to input being combined.
    TB_Matrix3D * residual;

    // Constructor and destructor.
    END_RESIDUAL_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        bool first, bool last, buffer_t b_t,
        const int in_h, const int in_w, const int in_c,
        norm_ops_t norm, act_ops_t act) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, in_c, in_h, in_w, norm, act)
    {
        args_type = END_RESIDUAL_ARGS_TYPE;
        name = "Residual";

        // Initialize residual if first layer (not that this makes much sense).
        if (first) {
            residual = new TB_Matrix3D[inf];
            generateIODataStructures(residual, inf, in_c, in_h, in_w, true);
        }
    }

    template<typename T>
    END_RESIDUAL_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t,
        T & in_lyr, norm_ops_t norm, act_ops_t act);

    template<typename T>
    END_RESIDUAL_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, norm_ops_t norm, act_ops_t act);

    ~END_RESIDUAL_LAYER_ARGS() {}
};

struct DWCONV_LAYER_ARGS :
    public LAYER_ARG_BASE, LAYER_ARG_3D_IN, LAYER_ARG_3D_OUT
{
    // Convolution variables.
    const int kernel_h;     // Kernel height.
    const int kernel_w;     // Kernel width.
    const int kernel_c;     // Kernel channels.
    const int kernel_size;  // Flattened kernel size.
    const int n_filters;    // Number of filters/kernels.
    const int stride;       // Stride over input.
    const int padding;      // Are we padding the input?
    PaddingType padding_type;

    // Kernels organized by channel, dim, dim.
    TB_Matrix3D weights;

    // Constructor.
    DWCONV_LAYER_ARGS(const int lyr, const int inf, const int thrd, bool first,
        bool last,  buffer_t b_t, const int in_h, const int in_w,
        const int in_c, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act) :
        LAYER_ARG_BASE(lyr, inf, thrd, first, last, b_t),
        LAYER_ARG_3D_IN(inf, first, in_c, in_h, in_w),
        LAYER_ARG_3D_OUT(inf, last, n_f, ((in_h - k_h + 2 * pad) / stri + 1),
            ((in_w - k_w + 2 * pad) / stri + 1), norm, act),
        kernel_h(k_h), kernel_w(k_w), kernel_c(in_c),
        kernel_size(k_w * k_h * in_c), n_filters(n_f), stride(stri),
        padding(pad)
    {
        args_type = DWCONV_ARGS_TYPE;
        name = "DWConv";
        padding_type = (pad) ? PADDING_SAME : PADDING_VALID;

        // Initialize weights.
        weights = TB_Matrix3D(n_f, k_h, k_w);
        weights.setRandom();
    }

    template<typename T>
    DWCONV_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t, T & in_lyr,
        const int k_h, const int k_w, const int n_f, const int stri,
        const int pad, norm_ops_t norm, act_ops_t act);

    template<typename T>
    DWCONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
        buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
        const int stri, const int pad, norm_ops_t norm, act_ops_t act);

    // Destructor.
    ~DWCONV_LAYER_ARGS() {}
};

///////////////////////////////
// Alternative constructors. //
///////////////////////////////

template<typename T>
CONV_LAYER_ARGS::CONV_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t,
    T & in_lyr, const int k_h, const int k_w, const int n_f, const int stri,
#if defined (AIMC)
    bool u_a,
#endif
    const int pad, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, 0, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f, ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    kernel_h(k_h), kernel_w(k_w), kernel_c(in_lyr.output_c),
    kernel_size(k_w * k_h * in_lyr.output_c), n_filters(n_f), stride(stri),
    padding(pad)
#if defined (AIMC)
    , use_aimc(u_a), aimc_h(-1), aimc_w(-1)
#endif
{
    args_type = CONV_ARGS_TYPE;
    name = "Conv";
    padding_type = (pad) ? PADDING_SAME : PADDING_VALID;

    // Initialize weights.
    weights = TB_Matrix2D(k_h * k_w * in_lyr.output_c, n_f);
    weights.setRandom();
}

template<typename T>
CONV_LAYER_ARGS::CONV_LAYER_ARGS(const int lyr, const int inf, const int thrd,
    buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
    const int stri,
#if defined (AIMC)
    bool u_a,
#endif
    const int pad, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f, ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    kernel_h(k_h), kernel_w(k_w), kernel_c(in_lyr.output_c),
    kernel_size(k_w * k_h * in_lyr.output_c), n_filters(n_f), stride(stri),
    padding(pad)
#if defined (AIMC)
    , use_aimc(u_a), aimc_h(-1), aimc_w(-1)
#endif
{
    args_type = CONV_ARGS_TYPE;
    name = "Conv";
    padding_type = (pad) ? PADDING_SAME : PADDING_VALID;

    // Initialize weights.
    weights = TB_Matrix2D(k_h * k_w * in_lyr.output_c, n_f);
    weights.setRandom();
}

template<typename T>
FC_LAYER_ARGS::FC_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t,
    T & in_lyr,
#if defined (AIMC)
    bool u_a,
#endif
    const int out_s, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, 0, false, false, b_t),
    LAYER_ARG_1D_IN(inf, false, in_lyr.output_size),
    LAYER_ARG_1D_OUT(inf, false, out_s, norm, act),
    weights_h(in_lyr.output_size), weights_w(out_s)
#if defined (AIMC)
    , use_aimc(u_a), aimc_h(-1), aimc_w(-1)
#endif
{
    args_type = FC_ARGS_TYPE;
    name = "Dense";

    // Initialize weights.
    weights = TB_Matrix2D(in_lyr.output_size, out_s);
    weights.setRandom();
}

template<typename T>
FC_LAYER_ARGS::FC_LAYER_ARGS(const int lyr, const int inf, const int thrd,
    buffer_t b_t, T & in_lyr,
#if defined (AIMC)
    bool u_a,
#endif
    const int out_s, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_1D_IN(inf, false, in_lyr.output_size),
    LAYER_ARG_1D_OUT(inf, false, out_s, norm, act),
    weights_h(in_lyr.output_size),
    weights_w(out_s)
#if defined (AIMC)
    , use_aimc(u_a), aimc_h(-1), aimc_w(-1)
#endif
{
    args_type = FC_ARGS_TYPE;
    name = "Dense";

    // Initialize weights.
    weights = TB_Matrix2D(in_lyr.output_size, out_s);
    weights.setRandom();
}

template<typename T>
POOL_LAYER_ARGS::POOL_LAYER_ARGS(const int lyr, const int inf, buffer_t b_t,
    T & in_lyr, const int p_d, const int p_f, const int stri,
    pool_ops_t p_type, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, 0, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, in_lyr.output_c, in_lyr.output_h / p_d,
        in_lyr.output_w / p_d, norm, act),
    pool_h(p_d), pool_w(p_d), pool_f(p_f), stride(stri), pool_type(p_type)
{
    args_type = POOL_ARGS_TYPE;
    name = "Pool";
}

template<typename T>
POOL_LAYER_ARGS::POOL_LAYER_ARGS(const int lyr, const int inf, const int thrd,
    buffer_t b_t, T & in_lyr, const int p_d, const int p_f, const int stri,
    pool_ops_t p_type, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, in_lyr.output_c, in_lyr.output_h / p_d,
        in_lyr.output_w / p_d, norm, act),
    pool_h(p_d), pool_w(p_d), pool_f(p_f), stride(stri), pool_type(p_type)
{
    args_type = POOL_ARGS_TYPE;
    name = "Pool";
}

template<typename T>
FLATTEN_LAYER_ARGS::FLATTEN_LAYER_ARGS(const int lyr, const int inf,
    buffer_t b_t, T & in_lyr) :
    LAYER_ARG_BASE(lyr, inf, 0, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_1D_OUT(inf, false,
        in_lyr.output_c * in_lyr.output_h * in_lyr.output_w, NO_NORM_TYPE,
        NO_ACT_TYPE)
{
    args_type = FLATTEN_ARGS_TYPE;
    name = "Flatten";
}

template<typename T>
FLATTEN_LAYER_ARGS::FLATTEN_LAYER_ARGS(const int lyr, const int inf,
    const int thrd, buffer_t b_t, T & in_lyr) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_1D_OUT(inf, false,
        in_lyr.output_c * in_lyr.output_h * in_lyr.output_w, NO_NORM_TYPE,
        NO_ACT_TYPE)
{
    args_type = FLATTEN_ARGS_TYPE;
    name = "Flatten";
}

template<typename T>
END_RESIDUAL_LAYER_ARGS::END_RESIDUAL_LAYER_ARGS(const int lyr, const int inf,
    buffer_t b_t, T & in_lyr, norm_ops_t norm,
    act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, 0, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w, norm, act)
{
    args_type = END_RESIDUAL_ARGS_TYPE;
    name = "Residual";
}

template<typename T>
END_RESIDUAL_LAYER_ARGS::END_RESIDUAL_LAYER_ARGS(const int lyr, const int inf,
    const int thrd, buffer_t b_t, T & in_lyr,
    norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w, norm, act)
{
    args_type = END_RESIDUAL_ARGS_TYPE;
    name = "Residual";
}

template<typename T>
DWCONV_LAYER_ARGS::DWCONV_LAYER_ARGS(const int lyr, const int inf,
    buffer_t b_t, T & in_lyr, const int k_h, const int k_w, const int n_f,
    const int stri, const int pad, norm_ops_t norm, act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, 0, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f,
        ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    kernel_h(k_h), kernel_w(k_w), kernel_c(in_lyr.output_c),
    kernel_size(k_w * k_h * in_lyr.output_c), n_filters(n_f), stride(stri),
    padding(pad)
{
    args_type = DWCONV_ARGS_TYPE;
    name = "DWConv";
    padding_type = (pad) ? PADDING_SAME : PADDING_VALID;

    // Initialize weights.
    weights = TB_Matrix3D(n_f, k_h, k_w);
    weights.setRandom();
}

template<typename T>
DWCONV_LAYER_ARGS::DWCONV_LAYER_ARGS(const int lyr, const int inf,
    const int thrd, buffer_t b_t, T & in_lyr, const int k_h, const int k_w,
    const int n_f, const int stri, const int pad, norm_ops_t norm,
    act_ops_t act) :
    LAYER_ARG_BASE(lyr, inf, thrd, false, false, b_t),
    LAYER_ARG_3D_IN(inf, false, in_lyr.output_c, in_lyr.output_h,
        in_lyr.output_w),
    LAYER_ARG_3D_OUT(inf, false, n_f,
        ((in_lyr.output_h - k_h + 2 * pad) / stri + 1),
        ((in_lyr.output_w - k_w + 2 * pad) / stri + 1), norm, act),
    kernel_h(k_h), kernel_w(k_w), kernel_c(in_lyr.output_c),
    kernel_size(k_w * k_h * in_lyr.output_c), n_filters(n_f), stride(stri),
    padding(pad)
{
    args_type = DWCONV_ARGS_TYPE;
    name = "DWConv";
    padding_type = (pad) ? PADDING_SAME : PADDING_VALID;

    // Initialize weights.
    weights = TB_Matrix3D(n_f, k_h, k_w);
    weights.setRandom();
}

///////////////
// Typedefs. //
///////////////

typedef CONV_LAYER_ARGS conv_layer_args;
typedef FC_LAYER_ARGS fc_layer_args;
typedef POOL_LAYER_ARGS pool_layer_args;
typedef FLATTEN_LAYER_ARGS flatten_layer_args;
typedef END_RESIDUAL_LAYER_ARGS end_residual_layer_args;
typedef DWCONV_LAYER_ARGS dwconv_layer_args;

#endif // __LAYER_HH__
