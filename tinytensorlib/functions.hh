/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file contains layer utilities, operations, and other functions.
 * - Main operations: convolution, fully-connected
 * - Pooling operations: Max Pool
 * - Normalization operations: LRN
 * - Activation operations: ReLU, Softmax, Sigmoid
 *
 */

#ifndef __FUNCTIONS_HH__
#define __FUNCTIONS_HH__

#include "layer.hh"
#include "../accelerator/smm_gem.h"
//#define AIMC

/////////////////////////////////////
// Main operation implementations. //
/////////////////////////////////////

// Conv2D implementation.  Extracts patches of input and then performs a MMM
// operation via "contract" or im2col MVMs.
inline void
Convolution2D(conv_layer_args *args, TB_Matrix3D &input,
              TB_Matrix2D &kernels, TB_Matrix3D &output) {
#if defined (SA)
    // Are we doing a whole convolution or only a partial convolution?
    // Set up dimensions array for MMM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
            {Eigen::IndexPair<int>(1, 0)};

    // Extract image patches.
    TB_Matrix2D patches = input.extract_image_patches(args->kernel_h, args->kernel_w, args->stride,
                                                      args->stride, 1, 1, args->padding_type)
            .reshape(Eigen::array<DenseIndex, 2>({args->output_h * args->output_w,
                                                  args->kernel_size}));


    if (args->use_sa) {
        // Case 1: kernel dimensions are multiples of the systolic array dimensions
        if ((args->kernel_size % args->sa_size == 0) &&
            args->n_filters % args->sa_size == 0) {

            std::cout << "Case 1 Conv " << patches.dimensions() << " X " << kernels.dimensions() << std::endl;
            smmComputeEigen(patches.dimension(0),
                            patches.data(),
                            output.data(),
                            kernels.data(),
                            patches.dimension(1),
                            kernels.dimension(1));

        }
            // Case 2: kernel size is not a multiple of SA_SIZE
        else if (args->n_filters % args->sa_size == 0) {
            smmComputeEigen(patches.dimension(0),
                            patches.data(),
                            output.data(),
                            kernels.data(),
                            patches.dimension(1),
                            kernels.dimension(1));

            const int multiple_sa_size = args->kernel_size - (args->kernel_size % args->sa_size);
            std::cout << "Case 2 Conv " << patches.dimensions() << " X " << kernels.dimensions() <<
                      " Size " << multiple_sa_size << std::endl;

            // Perform digital MVM.
            output += patches.slice(Eigen::array<Eigen::Index, 2>{0, multiple_sa_size},
                                    Eigen::array<Eigen::Index, 2>{args->output_h * args->output_w,
                                                                  args->kernel_size - multiple_sa_size})
                    .contract(kernels.slice(
                            Eigen::array<Eigen::Index, 2>{multiple_sa_size, 0},
                            Eigen::array<Eigen::Index, 2>{args->kernel_size - multiple_sa_size,
                                                          args->n_filters}), product_dims)
                    .reshape(output.dimensions());

        }
            // Case 3: n filter is not a multiple of SA_SIZE
        else if (args->kernel_size % args->sa_size == 0) {
            smmComputeEigen(patches.dimension(0),
                            patches.data(),
                            output.data(),
                            kernels.data(),
                            patches.dimension(1),
                            kernels.dimension(1));

            const int multiple_sa_size = args->n_filters - (args->n_filters % args->sa_size);

            std::cout << "Case 3 Conv " << patches.dimensions() << " X " << kernels.dimensions() <<
                      " Size " << multiple_sa_size << std::endl;


            // Perform digital MVM.
            output.slice(
                    Eigen::array<Eigen::Index, 3>{multiple_sa_size, 0, 0},
                    Eigen::array<Eigen::Index, 3>{args->n_filters - multiple_sa_size,
                                                  args->output_h, args->output_w}
            ) += patches.contract(kernels.slice(
                            Eigen::array<Eigen::Index, 2>{0, multiple_sa_size},
                            Eigen::array<Eigen::Index, 2>{args->kernel_size,
                                                          args->n_filters - multiple_sa_size}), product_dims)
                    .reshape(Eigen::array<DenseIndex, 3>(
                            {args->n_filters - multiple_sa_size,
                             args->output_h, args->output_w}));
        }
            // Case 4: SA tile doesn't fit both kernel height and # of kernels.
        else {
            smmComputeEigen(patches.dimension(0),
                            patches.data(),
                            output.data(),
                            kernels.data(),
                            patches.dimension(1),
                            kernels.dimension(1));

            const int multiple_sa_size_nfilter = args->n_filters - (args->n_filters % args->sa_size);
            const int multiple_sa_size_kernel = args->kernel_size - (args->kernel_size % args->sa_size);
            std::cout << "Case 4 Conv " << patches.dimensions() << " X " << kernels.dimensions() <<
                      " Size " << multiple_sa_size_nfilter << " and " << multiple_sa_size_kernel << std::endl;


            // Perform right-hand partial digital MVM.
            output.slice(
                    Eigen::array<Eigen::Index, 3>{multiple_sa_size_nfilter, 0, 0},
                    Eigen::array<Eigen::Index, 3>{args->n_filters - multiple_sa_size_nfilter,
                                                  args->output_h, args->output_w}
            ) += patches
                    .contract(kernels
                                      .slice(
                                              Eigen::array<Eigen::Index, 2>{0, multiple_sa_size_nfilter},
                                              Eigen::array<Eigen::Index, 2>{args->kernel_size,
                                                                            args->n_filters -
                                                                            multiple_sa_size_nfilter}),
                              product_dims)
                    .reshape(Eigen::array<DenseIndex, 3>(
                            {args->n_filters - multiple_sa_size_nfilter,
                             args->output_h, args->output_w}));

            // Perform bottom-left partial digital MVM.
            output.slice(
                    Eigen::array<Eigen::Index, 3>{0, 0, 0},
                    Eigen::array<Eigen::Index, 3>{multiple_sa_size_nfilter,
                                                  args->output_h, args->output_w}
            ) += patches
                    .slice(Eigen::array<Eigen::Index, 2>{0, multiple_sa_size_kernel},
                           Eigen::array<Eigen::Index, 2>{
                                   args->output_h * args->output_w,
                                   args->kernel_size - multiple_sa_size_kernel})
                    .contract(kernels.slice(
                                      Eigen::array<Eigen::Index, 2>{multiple_sa_size_kernel, 0},
                                      Eigen::array<Eigen::Index, 2>{
                                              args->kernel_size - multiple_sa_size_kernel, multiple_sa_size_nfilter}),
                              product_dims)
                    .reshape(Eigen::array<DenseIndex, 3>(
                            {multiple_sa_size_nfilter, args->output_h, args->output_w}));

        }
    } else {
        // Do fully digital convolution.
        output = patches.contract(kernels, product_dims)
                .reshape(output.dimensions());
    }
#else
    // Set up dimensions array for MMM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
            {Eigen::IndexPair<int>(1, 0)};

    // Extract image patches.
    TB_Matrix2D patches = input.extract_image_patches(args->kernel_h,
        args->kernel_w, args->stride, args->stride, 1, 1, args->padding_type)
        .reshape(Eigen::array<DenseIndex, 2>({args->output_h * args->output_w,
            args->kernel_size}));
    std::cout << "TiC-SAT Conv " << patches.dimensions() << " X " << kernels.dimensions() << std::endl;
    smmComputeEigen(patches.dimension(0),
                    patches.data(),
                    output.data(),
                    kernels.data(),
                    patches.dimension(1),
                    kernels.dimension(1));
    // Do convolution.
//    output = patches.contract(kernels, product_dims)
//        .reshape(output.dimensions());
#endif

    return;
}

// Fully connected/MVM operation imeplementation.
// TODO: Optimize 3D accesses with vectorization as in previous FullyConnected
// method.
inline void
FullyConnected(fc_layer_args *args, TB_Vector &input,
               TB_Vector &output) {
#if defined (AIMC)
    // Are we doing a whole MVM or only a partial MVM?
    if (args->use_sa) {
        // Case 1: Weights fit entirely in AIMC tile.
        if ((args->aimc_h < 0 && args->aimc_w < 0) ||
            (args->aimc_h >= args->input_size &&
                args->aimc_w >= args->output_size)) {
            // Queue input.
            queueVector(args->input_size, input);

            // Perform MVM.
            aimcProcess();

            // Dequeue output.
            dequeueVector(args->output_size, output);
        }
        // Case 2: AIMC tile height < weights matrix height.
        else if (args->aimc_w >= args->output_size) {
            // Queue partial input.
            queueVector(args->aimc_h, input);

            // Perform MVM.
            aimcProcess();

            // Dequeue partial output.
            dequeueVector(args->output_size, output);

            // Set up dimensions and do partial MVM.
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
                {Eigen::IndexPair<int>(1, 0)};

            output += input
                .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
                .slice(Eigen::array<Eigen::Index, 2>{0, args->aimc_h},
                    Eigen::array<Eigen::Index, 2>{1, args->input_size - args->aimc_h})
                .contract(args->weights.slice(
                    Eigen::array<Eigen::Index, 2>{args->aimc_h, 0},
                    Eigen::array<Eigen::Index, 2>{args->input_size - args->aimc_h, args->output_size}
                ), product_dims)
                .reshape(output.dimensions());
        }
        // Case 3: AIMC tile width < weights matrix width.
        else if (args->aimc_h >= args->input_size) {
            // Queue partial input.
            queueVector(args->input_size, input);

            // Perform MVM.
            aimcProcess();

            // Dequeue partial output.
            dequeueVector(args->aimc_w, output);

            // Set up dimensions and do partial MVM.
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
                {Eigen::IndexPair<int>(1, 0)};

            output.slice(
                Eigen::array<Eigen::Index, 1>{args->aimc_w},
                Eigen::array<Eigen::Index, 1>{args->output_size - args->aimc_w}
            ) += input
                .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
                .contract(args->weights.slice(
                    Eigen::array<Eigen::Index, 2>{0, args->aimc_w},
                    Eigen::array<Eigen::Index, 2>{args->input_size, args->output_size - args->aimc_w}
                ), product_dims)
                .reshape(Eigen::array<DenseIndex, 1>({args->output_size - args->aimc_w}));
        }
        // Case 4: AIMC tile doesn't fit both kernel height and # of kernels.
        else {
            // Queue partial input.
            queueVector(args->aimc_h, input);

            // Perform MVM.
            aimcProcess();

            // Dequeue partial output.
            dequeueVector(args->aimc_w, output);

            // Set up dimensions and do partial MVM, bottom left first then
            // right portion.
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
                {Eigen::IndexPair<int>(1, 0)};

            // Perform right-portion digital partial MVM.
            output.slice(
                Eigen::array<Eigen::Index, 1>{args->aimc_w},
                Eigen::array<Eigen::Index, 1>{args->output_size - args->aimc_w}
            ) += input
                .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
                .contract(args->weights.slice(
                    Eigen::array<Eigen::Index, 2>{0, args->aimc_w},
                    Eigen::array<Eigen::Index, 2>{args->input_size, args->output_size - args->aimc_w}
                ), product_dims)
                .reshape(Eigen::array<DenseIndex, 1>({args->output_size - args->aimc_w}));

            // Perform left-bottom-portion digital partial MVM.
            output.slice(
                Eigen::array<Eigen::Index, 1>{0},
                Eigen::array<Eigen::Index, 1>{args->aimc_w}
            ) += input
                .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
                .slice(Eigen::array<Eigen::Index, 2>{0, args->aimc_h},
                    Eigen::array<Eigen::Index, 2>{1, args->input_size - args->aimc_h})
                .contract(args->weights.slice(
                    Eigen::array<Eigen::Index, 2>{args->aimc_h, 0},
                    Eigen::array<Eigen::Index, 2>{args->input_size - args->aimc_h, args->aimc_w}
                ), product_dims)
                .reshape(Eigen::array<DenseIndex, 1>({args->aimc_w}));
        }
    } else {
        // Set up dimensions array and input for MVM.
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
            {Eigen::IndexPair<int>(1, 0)};

        // Do fully digital MVM.
        TB_Matrix2D res = input.reshape(
            Eigen::array<Eigen::Index, 2>{1, args->weights_h});
        output = res.contract(args->weights, product_dims)
            .reshape(output.dimensions());
    }
#else
    // Set up dimensions array and input for MVM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
            {Eigen::IndexPair<int>(1, 0)};

    // Do fully digital MVM.
    TB_Matrix2D res = input.reshape(
            Eigen::array<Eigen::Index, 2>{1, args->weights_h});
    std::cout << "TiC-SAT FC " << res.dimensions() << " X " << (args->weights).dimensions() << std::endl;
    smmComputeEigen(res.dimension(0),
                    res.data(),
                    output.data(),
                    (args->weights).data(),
                    res.dimension(1),
                    (args->weights).dimension(1));

//    output = res.contract(args->weights, product_dims)
//        .reshape(output.dimensions());
#endif
    return;
}

// Pooling operation implementations.
inline void
Pooling(pool_layer_args *args, TB_Matrix3D &input,
        TB_Matrix3D &output, pool_ops_t pool_type) {
    int i, j, ii, jj, ch, p_s;

    // Dimensions to perform pool on.
    // Dimension 0 is always channel which is not pooled.
    Eigen::array<int, 2> dims({1, 2});

    switch (pool_type) {
        // Defying all logic, this is actually faster than https://github.com/
        // tensorflow/tensorflow/blob/v1.13.1/tensorflow/core/kernels/
        // eigen_pooling.h#L131
        case MAX_POOL_TYPE: {
            for (ch = 0; ch < args->output_c; ch++) {
                for (i = 0, ii = 0; i < args->output_h; i += args->stride,
                        ii++) {
                    for (j = 0, jj = 0; j < args->output_w; j += args->stride,
                            jj++) {
                        TB_Vector tmp = input.slice( // Offsets, Extents
                                Eigen::array<Eigen::Index, 3>{ch, i, j},
                                Eigen::array<Eigen::Index, 3>{1, args->pool_h,
                                                              args->pool_w}
                        ).maximum(dims);
                        output(ch, ii, jj) = tmp(0);
                    }
                }
            }
            break;
        }
        case AVG_POOL_TYPE: {
            // Note: is susceptible to overflow errors.
            p_s = args->pool_h * args->pool_w;

            for (ch = 0; ch < args->output_c; ch++) {
                for (i = 0, ii = 0; i < args->output_h; i += args->stride,
                        ii++) {
                    for (j = 0, jj = 0; j < args->output_w; j += args->stride,
                            jj++) {
                        TB_Vector tmp = input.slice( // Offsets, Extents
                                Eigen::array<Eigen::Index, 3>{ch, i, j},
                                Eigen::array<Eigen::Index, 3>{1, args->pool_h,
                                                              args->pool_w}
                        ).sum(dims);
                        output(ch, ii, jj) = tmp(0) / p_s;
                    }
                }
            }
            break;
        }
        default: {
            break;
        }
    }

    return;
}

// Flatten operation implementation.
inline void
Flatten(TB_Matrix3D &m, TB_Vector &v) {
    v = m.reshape(v.dimensions());

    return;
}

// The combination portion of the residual block implementation.
inline void
EndResidual(end_residual_layer_args *args, TB_Matrix3D &input,
            TB_Matrix3D &residual, TB_Matrix3D &output) {
    output = input + residual;
    return;
}

// DWConv2D brute-force implementation.
inline void
DepthwiseConvolution2D(dwconv_layer_args *args, TB_Matrix3D &input,
                       TB_Matrix3D &kernels, TB_Matrix3D &output) {
    // Lovingly borrowed from https://iq.opengenus.org/depthwise-convolution/.
    // Note: Using extract_image_patches is NOT faster than brute force here.
    for (int out_h = 0; out_h < args->output_h; out_h++) {
        for (int out_w = 0; out_w < args->output_w; out_w++) {
            for (int channel = 0; channel < args->input_c; channel++) {
                int tmp = 0;
                for (int j = 0; j < args->kernel_w; j++) {
                    for (int i = 0; i < args->kernel_h; i++) {
                        tmp += kernels(channel, i, j) *
                               input(channel, out_h + i, out_w + j);
                    }
                }
                output(channel, out_h, out_w) = tmp;
            }
        }
    }

    return;
}

// Normalization operation implementations.
inline void
Normalization(TB_Vector &v, norm_ops_t norm_type) {
    switch (norm_type) {
        case NO_NORM_TYPE: {
            break;
        }
        case BATCH_NORM_TYPE: {
            break;
        }
        case LRN_NORM_TYPE: {
            break;
        }
        default: {
            break;
        }
    }

    return;
}

/* 
 * This implements inter-channel LRN in [1], where k = 2, n = 5,
 * Alpha = 1e-5, Beta = 0.75, N = number of kernels.
 *
 * [1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification
 * with deep convolutional neural networks,” in NIPS, 2012, pp. 1106–1114.
 */
inline void
Normalization(TB_Matrix3D &m, norm_ops_t norm_type) {
    int chas = m.dimensions()[0];
    int rows = m.dimensions()[1];
    int cols = m.dimensions()[2];

    switch (norm_type) {
        case NO_NORM_TYPE: {
            break;
        }
        case BATCH_NORM_TYPE: {
            break;
        }
        case LRN_NORM_TYPE: {
            const int size = chas * rows * cols;
            const double delta = 5.0 / 2.0; // n / 2
            const int upper = chas - 1;

            TB_Vector v = m.reshape(Eigen::array<DenseIndex, 1>({size}));

            for (int i = 0; i < size; i++) {
                double sum = 0;
                int min = (int) fmax(0, i - delta);
                int max = (int) fmin(upper, i + delta);

                // Grab sum.
                for (int j = min; j < max; j++) {
                    sum += v(j) * v(i);
                }

                // y[i] = x[i] / (k + alpha * sum)^beta
                v(i) = v(i) / pow(2 + 1e-5 * sum, 0.75);
            }

            m = v.reshape(m.dimensions());

            break;
        }
        default: {
            break;
        }
    }

    return;
}

// Activation function implementations.
inline void
Activation(TB_Vector &v, act_ops_t act_type) {
    int size = v.dimensions()[0];

    switch (act_type) {
        case RELU_ACT_TYPE: {
            for (int i = 0; i < size; i++) {
                if (v(i) < 0) {
                    v(i) = 0;
                }
            }
            break;
        }
        case SIGMOID_ACT_TYPE: {
            break;
        }
        case SOFTMAX_ACT_TYPE: {
            // Not necessarily the most efficient, although correct,
            // implementation.
            // TODO: Make more efficient.
            Tensor<float, 0> expsum = v.cast<float>().exp().sum();
            expsum(0) = 1.0 / expsum(0);
            for (int i = 0; i < size; i++) {
                v(i) = exp(v(i)) * expsum(0);
            }
            break;
        }
        case RELU6_ACT_TYPE: {
            for (int i = 0; i < size; i++) {
                if (v(i) < 0) {
                    v(i) = 0;
                } else if (v(i) > 6) {
                    v(i) = 6;
                }
            }
            break;
        }
        default: {
            break;
        }
    }

    return;
}

inline void
Activation(TB_Matrix3D &m, act_ops_t act_type) {
    int chas = m.dimensions()[0];
    int rows = m.dimensions()[1];
    int cols = m.dimensions()[2];

    switch (act_type) {
        case RELU_ACT_TYPE: {
            for (int ch = 0; ch < chas; ch++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        if (m(ch, i, j) < 0) {
                            m(ch, i, j) = 0;
                        }
                    }
                }
            }
            break;
        }
        case SIGMOID_ACT_TYPE: {
            break;
        }
        case SOFTMAX_ACT_TYPE: {
            // Not necessarily the most efficient, although correct,
            // implementation.
            // TODO: Make more efficient.
            Tensor<float, 0> expsum = m.cast<float>().exp().sum();
            expsum(0) = 1.0 / expsum(0);
            for (int ch = 0; ch < chas; ch++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        m(ch, i, j) = exp(m(ch, i, j)) * expsum(0);
                    }
                }
            }
            break;
        }
        case RELU6_ACT_TYPE: {
            for (int ch = 0; ch < chas; ch++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        if (m(ch, i, j) < 0) {
                            m(ch, i, j) = 0;
                        } else if (m(ch, i, j) > 6) {
                            m(ch, i, j) = 6;
                        }
                    }
                }
            }
            break;
        }
        default: {
            break;
        }
    }

    return;
}

/////////////////////
// Master methods. //
/////////////////////

inline void
doLayer(conv_layer_args &args, int in_idx = 0, int out_idx = 0) {
    Convolution2D(&args, args.input[in_idx], args.weights,
                  args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(fc_layer_args &args, int in_idx = 0, int out_idx = 0) {
    FullyConnected(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(pool_layer_args &args, int in_idx = 0, int out_idx = 0) {
    Pooling(&args, args.input[in_idx], args.output[out_idx], args.pool_type);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(flatten_layer_args &args, int in_idx = 0, int out_idx = 0) {
    Flatten(args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(end_residual_layer_args &args, int in_idx = 0, int out_idx = 0) {
    EndResidual(&args, args.input[in_idx], args.residual[in_idx],
                args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(dwconv_layer_args &args, int in_idx = 0, int out_idx = 0) {
    DepthwiseConvolution2D(&args, args.input[in_idx], args.weights,
                           args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

#endif // __FUNCTIONS_HH__
