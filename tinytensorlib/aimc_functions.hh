/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file contains analog tile-enabled layer utilities, operations, and
 * other methods for main operations:
 * - Convolution (Conv2D)
 * - Fully-Connected (Dense)
 *
 */

#ifndef __AIMC_FUNCTIONS_HH__
#define __AIMC_FUNCTIONS_HH__

#include "aimc_layer.hh"

/////////////////////////////////////
// Main operation implementations. //
/////////////////////////////////////

// Conv2D implementation for case 1: AIMC tile height/width >= kernel height/
// number of kernels.
inline void
Convolution2D(analog_case1_conv_layer_args * args, TB_Matrix3D & input,
    TB_Matrix3D & output)
{
    // Iterate over pixels.
    for (int p_i = 0; p_i < args->output_h; p_i++) {
        for (int p_j = 0; p_j < args->output_w; p_j++) {
            // Queue patch, MVM, get pixel output.
            queuePatchExperimental(input, p_i, p_j, args->input_c,
                args->kernel_h, args->kernel_w, args->stride,
                args->thread_n);
            aimcProcess();
            dequeuePixel(output, p_i, p_j, args->output_c);
        }
    }

    return;
}

// Conv2D implementation for case 2: AIMC tile height < kernel height and AIMC
// tile width >= number of kernels.
inline void
Convolution2D(analog_case2_conv_layer_args * args, TB_Matrix3D & input,
    TB_Matrix3D & output)
{
    for (int p_i = 0, patch = 0; p_i < args->output_h; p_i++) {
        for (int p_j = 0; p_j < args->output_w; p_j++, patch++) {
            // Queue patch, MVM, get pixel output.
            queuePatchExperimental(input, p_i, p_j, args->input_c,
                args->kernel_h, args->kernel_w, args->stride,
                args->aimc_h, args->thread_n);
            aimcProcess();
            dequeuePixel(output, p_i, p_j, args->output_c);
        }
    }

    // Set up dimensions array for MMM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        {Eigen::IndexPair<int>(1, 0)};

    // Extract image patches.
    TB_Matrix2D patches = input.extract_image_patches(args->kernel_h,
            args->kernel_w, args->stride, args->stride, 1, 1,
            args->padding_type)
        .reshape(Eigen::array<DenseIndex, 2>(
            {args->output_h * args->output_w, args->kernel_size}))
        .slice(Eigen::array<Eigen::Index, 2>{0, args->aimc_h},
            Eigen::array<Eigen::Index, 2>{args->output_h * args->output_w,
                args->kernels_h});

    // Perform digital MVM.
    output += patches.contract(args->kernels, product_dims)
        .reshape(output.dimensions());

    return;
}

// Conv2D implementation for case 3: AIMC tile height >= kernel height and AIMC
// tile width < number of kernels.
inline void
Convolution2D(analog_case3_conv_layer_args * args, TB_Matrix3D & input,
    TB_Matrix3D & output)
{
    for (int p_i = 0; p_i < args->output_h; p_i++) {
        for (int p_j = 0; p_j < args->output_w; p_j++) {
            // Queue patch, MVM, get pixel output.
            queuePatchExperimental(input, p_i, p_j, args->input_c,
                args->kernel_h, args->kernel_w, args->stride,
                args->thread_n);
            aimcProcess();
            dequeuePixel(output, p_i, p_j, args->aimc_w);
        }
    }

    // Set up dimensions array for MMM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        {Eigen::IndexPair<int>(1, 0)};

    // Extract image patches.
    TB_Matrix2D patches = input.extract_image_patches(args->kernel_h,
        args->kernel_w, args->stride, args->stride, 1, 1,
        args->padding_type)
        .reshape(Eigen::array<DenseIndex, 2>(
            {args->output_h * args->output_w, args->kernel_size}));

    // Perform digital MVM.
    output.slice(Eigen::array<Eigen::Index, 3>{args->aimc_w, 0, 0},
        Eigen::array<Eigen::Index, 3>{args->kernels_w, args->output_h,
            args->output_w}
    ) += patches.contract(args->kernels, product_dims)
        .reshape(Eigen::array<DenseIndex, 3>({args->kernels_w, args->output_h,
            args->output_w}));

    return;
}

// Conv2D implementation for case 4: AIMC tile height/width < kernel height/
// number of kernels.
inline void
Convolution2D(analog_case4_conv_layer_args * args, TB_Matrix3D & input,
    TB_Matrix3D & output)
{
    for (int p_i = 0; p_i < args->output_h; p_i++) {
        for (int p_j = 0; p_j < args->output_w; p_j++) {
            // Queue patch, MVM, get pixel output.
            queuePatchExperimental(input, p_i, p_j, args->input_c,
                   args->kernel_h, args->kernel_w, args->stride,
                   args->aimc_h, args->thread_n);
            aimcProcess();
            dequeuePixel(output, p_i, p_j, args->aimc_w);
        }
    }

    // Set up dimensions array for MMM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        {Eigen::IndexPair<int>(1, 0)};

    // Extract image patches.
    TB_Matrix2D patches = input.extract_image_patches(args->kernel_h,
            args->kernel_w, args->stride, args->stride, 1, 1,
            args->padding_type)
        .reshape(Eigen::array<DenseIndex, 2>(
            {args->output_h * args->output_w, args->kernel_size}));

    // Perform right-hand partial digital MVM.
    output.slice(
        Eigen::array<Eigen::Index, 3>{args->aimc_w, 0, 0},
        Eigen::array<Eigen::Index, 3>{args->kernels_right_w, args->output_h,
            args->output_w}
    ) += patches
        .contract(args->kernels_right, product_dims)
        .reshape(Eigen::array<DenseIndex, 3>({args->kernels_right_w,
            args->output_h, args->output_w}));

    // Perform bottom-left partial digital MVM.
    output.slice(
        Eigen::array<Eigen::Index, 3>{0, 0, 0},
        Eigen::array<Eigen::Index, 3>{args->aimc_w, args->output_h,
            args->output_w}
    ) += patches
        .slice(Eigen::array<Eigen::Index, 2>{0, args->aimc_h},
            Eigen::array<Eigen::Index, 2>{
                args->output_h * args->output_w,
                args->kernels_left_bottom_h})
        .contract(args->kernels_left_bottom, product_dims)
        .reshape(Eigen::array<DenseIndex, 3>(
            {args->kernels_left_bottom_w, args->output_h, args->output_w}));

    return;
}

// Dense layer implementation for case 1: AIMC tile height/width >= in/out
// size.
inline void
FullyConnected(analog_case1_fc_layer_args * args, TB_Vector & input,
    TB_Vector & output)
{
    // Queue input, MVM, dequeue output.
    queueVector(args->input_size, input);
    aimcProcess();
    dequeueVector(args->output_size, output);

    return;
}

// Dense layer implementation for case 2: AIMC tile height < in size and AIMC
// tile width >= out size.
inline void
FullyConnected(analog_case2_fc_layer_args * args, TB_Vector & input,
    TB_Vector & output)
{
    // Queue partial input, MVM, partial dequeue.
    queueVector(args->aimc_h, input);
    aimcProcess();
    dequeueVector(args->output_size, output);

    // Set up dimensions and do partial MVM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        {Eigen::IndexPair<int>(1, 0)};

    output += input
        .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
        .slice(Eigen::array<Eigen::Index, 2>{0, args->aimc_h},
            Eigen::array<Eigen::Index, 2>{1, args->weights_h})
        .contract(args->weights, product_dims)
        .reshape(output.dimensions());

    return;
}

// Dense layer implementation for case 3: AIMC tile height >= in size and AIMC
// tile width < out size.
inline void
FullyConnected(analog_case3_fc_layer_args * args, TB_Vector & input,
    TB_Vector & output)
{
    // Queue partial input, perform MVM, dequeue partial output.
    queueVector(args->input_size, input);
    aimcProcess();
    dequeueVector(args->aimc_w, output);

    // Set up dimensions and do partial MVM.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        {Eigen::IndexPair<int>(1, 0)};

    output.slice(
        Eigen::array<Eigen::Index, 1>{args->aimc_w},
        Eigen::array<Eigen::Index, 1>{args->weights_w}
    ) += input
        .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
        .contract(args->weights, product_dims)
        .reshape(Eigen::array<DenseIndex, 1>({args->weights_w}));

    return;
}

// Dense layer implementation for case 4: AIMC tile height/width < in/out size.
inline void
FullyConnected(analog_case4_fc_layer_args * args, TB_Vector & input,
    TB_Vector & output)
{
    // Queue partial input, perform MVM, dequeue partial output.
    queueVector(args->aimc_h, input);
    aimcProcess();
    dequeueVector(args->aimc_w, output);

    // Set up dimensions and do partial MVM, bottom left first then
    // right portion.
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        {Eigen::IndexPair<int>(1, 0)};

    // Perform right-portion digital partial MVM.
    output.slice(
        Eigen::array<Eigen::Index, 1>{args->aimc_w},
        Eigen::array<Eigen::Index, 1>{args->weights_right_w}
    ) += input
        .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
        .contract(args->weights_right, product_dims)
        .reshape(Eigen::array<DenseIndex, 1>({args->weights_right_w}));

    // Perform left-bottom-portion digital partial MVM.
    output.slice(
        Eigen::array<Eigen::Index, 1>{0},
        Eigen::array<Eigen::Index, 1>{args->aimc_w}
    ) += input
        .reshape(Eigen::array<Eigen::Index, 2>{1, args->input_size})
        .slice(Eigen::array<Eigen::Index, 2>{0, args->aimc_h},
            Eigen::array<Eigen::Index, 2>{1, args->weights_left_bottom_h})
        .contract(args->weights_left_bottom, product_dims)
        .reshape(Eigen::array<DenseIndex, 1>({args->aimc_w}));

    return;
}

/////////////////////
// Master methods. //
/////////////////////

inline void
doLayer(analog_case1_conv_layer_args & args, int in_idx=0, int out_idx=0)
{
    Convolution2D(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case2_conv_layer_args & args, int in_idx=0, int out_idx=0)
{
    Convolution2D(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case3_conv_layer_args & args, int in_idx=0, int out_idx=0)
{
    Convolution2D(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case4_conv_layer_args & args, int in_idx=0, int out_idx=0)
{
    Convolution2D(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case1_fc_layer_args & args, int in_idx=0, int out_idx=0)
{
    FullyConnected(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case2_fc_layer_args & args, int in_idx=0, int out_idx=0)
{
    FullyConnected(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case3_fc_layer_args & args, int in_idx=0, int out_idx=0)
{
    FullyConnected(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

inline void
doLayer(analog_case4_fc_layer_args & args, int in_idx=0, int out_idx=0)
{
    FullyConnected(&args, args.input[in_idx], args.output[out_idx]);
    Normalization(args.output[out_idx], args.normalization);
    Activation(args.output[out_idx], args.activation);
    return;
}

/////////////////////////
// Printing utilities. //
/////////////////////////

inline void
printLayerInfo(analog_case1_conv_layer_args * conv_args)
{
    cout << "Layer " << conv_args->layer_n << ": " << conv_args->name
        << " Type\t|";
    cout << " Named " << conv_args->name << "\t|";
    cout << " Input: " << conv_args->input[0].dimensions() << "\t|";
    cout << " Output: " << conv_args->output[0].dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << conv_args->aimc_h << ", " 
        << conv_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case2_conv_layer_args * conv_args)
{
    cout << "Layer " << conv_args->layer_n << ": " << conv_args->name
        << " Type\t|";
    cout << " Named " << conv_args->name << "\t|";
    cout << " Input: " << conv_args->input[0].dimensions() << "\t|";
    cout << " Output: " << conv_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << conv_args->kernels.dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << conv_args->aimc_h << ", " 
        << conv_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case3_conv_layer_args * conv_args)
{
    cout << "Layer " << conv_args->layer_n << ": " << conv_args->name
        << " Type\t|";
    cout << " Named " << conv_args->name << "\t|";
    cout << " Input: " << conv_args->input[0].dimensions() << "\t|";
    cout << " Output: " << conv_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << conv_args->kernels.dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << conv_args->aimc_h << ", " 
        << conv_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case4_conv_layer_args * conv_args)
{
    cout << "Layer " << conv_args->layer_n << ": " << conv_args->name
        << " Type\t|";
    cout << " Named " << conv_args->name << "\t|";
    cout << " Input: " << conv_args->input[0].dimensions() << "\t|";
    cout << " Output: " << conv_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << conv_args->kernels_left_bottom.dimensions() 
        << " / " << conv_args->kernels_right.dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << conv_args->aimc_h << ", " 
        << conv_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case1_fc_layer_args * fc_args)
{
    cout << "Layer " << fc_args->layer_n << ": " << fc_args->name
        << " Type\t|";
    cout << " Named " << fc_args->name << "\t|";
    cout << " Input: " << fc_args->input[0].dimensions() << "\t|";
    cout << " Output: " << fc_args->output[0].dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << fc_args->aimc_h << ", " 
        << fc_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case2_fc_layer_args * fc_args)
{
    cout << "Layer " << fc_args->layer_n << ": " << fc_args->name
        << " Type\t|";
    cout << " Named " << fc_args->name << "\t|";
    cout << " Input: " << fc_args->input[0].dimensions() << "\t|";
    cout << " Output: " << fc_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << fc_args->weights.dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << fc_args->aimc_h << ", " 
        << fc_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case3_fc_layer_args * fc_args)
{
    cout << "Layer " << fc_args->layer_n << ": " << fc_args->name
        << " Type\t|";
    cout << " Named " << fc_args->name << "\t|";
    cout << " Input: " << fc_args->input[0].dimensions() << "\t|";
    cout << " Output: " << fc_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << fc_args->weights.dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << fc_args->aimc_h << ", " 
        << fc_args->aimc_w << "]\n";
    return;
}

inline void
printLayerInfo(analog_case4_fc_layer_args * fc_args)
{
    cout << "Layer " << fc_args->layer_n << ": " << fc_args->name
        << " Type\t|";
    cout << " Named " << fc_args->name << "\t|";
    cout << " Input: " << fc_args->input[0].dimensions() << "\t|";
    cout << " Output: " << fc_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << fc_args->weights_left_bottom.dimensions() 
        << " / " << fc_args->weights_right.dimensions() << "\t|";
    cout << " AIMC Tile: " << "[" << fc_args->aimc_h << ", " 
        << fc_args->aimc_w << "]\n";
    return;
}

#endif // __AIMC_FUNCTIONS_HH__
