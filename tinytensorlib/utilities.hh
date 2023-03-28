/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file contains utilities for setting up buffers, inputs, etc.
 *
 */

#ifndef __UTILITIES_HH__
#define __UTILITIES_HH__

#include <iostream>

using namespace std;

#include "layer.hh"

#if defined (AIMC)
#include "aimc_layer.hh"
#endif // AIMC

/////////////////////////
// Printing utilities. //
/////////////////////////

inline void
printLayerInfo(conv_layer_args * conv_args)
{
    cout << "Layer " << conv_args->layer_n << ": Conv Type\t|";
    cout << " Named " << conv_args->name << "\t|";
    cout << " Input: " << conv_args->input[0].dimensions() << "\t|";
    cout << " Output: " << conv_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << conv_args->weights.dimensions();
    cout << endl;
    return;
}

inline void
printLayerInfo(fc_layer_args * fc_args)
{   
    cout << "Layer " << fc_args->layer_n << ": Dense Type\t|";
    cout << " Named " << fc_args->name << "\t|";
    cout << " Input: " << fc_args->input[0].dimensions() << "\t\t|";
    cout << " Output: " << fc_args->output[0].dimensions() << "\t\t|";
    cout << " Weights: " << fc_args->weights.dimensions();
    cout << endl;
    return;
}

inline void
printLayerInfo(pool_layer_args * pool_args)
{   
    cout << "Layer " << pool_args->layer_n << ": Pool Type\t|";
    cout << " Named " << pool_args->name << "\t|";
    cout << " Input: " << pool_args->input[0].dimensions() << "\t|";
    cout << " Output: " << pool_args->output[0].dimensions();
    cout << endl;
    return;
}

inline void
printLayerInfo(flatten_layer_args * flatten_args)
{   
    cout << "Layer " << flatten_args->layer_n << ": Flatten Type\t|";
    cout << " Named " << flatten_args->name << "\t|";
    cout << " Input: " << flatten_args->input[0].dimensions() << "\t|";
    cout << " Output: " << flatten_args->output[0].dimensions();
    cout << endl;
    return;
}

inline void
printLayerInfo(end_residual_layer_args * end_res_args)
{   
    cout << "Layer " << end_res_args->layer_n << ": Residual Type\t|";
    cout << " Named " << end_res_args->name << "\t|";
    cout << " Input: " << end_res_args->input[0].dimensions() << "\t|";
    cout << " Output: " << end_res_args->output[0].dimensions() << "\t|";
    cout << " Residual: " << end_res_args->residual[0].dimensions(); 
    cout << endl;
    return;
}

inline void
printLayerInfo(dwconv_layer_args * dwconv_args)
{
    cout << "Layer " << dwconv_args->layer_n << ": DWConv Type\t|";
    cout << " Named " << dwconv_args->name << "\t|";
    cout << " Input: " << dwconv_args->input[0].dimensions() << "\t|";
    cout << " Output: " << dwconv_args->output[0].dimensions() << "\t|";
    cout << " Weights: " << dwconv_args->weights.dimensions();
    cout << endl;
    return;
}

inline void
printVector(TB_Vector v, int limit=-1)
{
    int l = (limit < 0) ? v.dimensions()[0] : limit;

    cout << "Vector contents: ";
    for (int i = 0; i < l; i++) {
        cout << (int)v(i) << "\t";
    } cout << endl;

    return;
}

inline void
printMatrix(TB_Matrix2D m, int limit_x=-1, int limit_y=-1)
{
    int lx = (limit_x < 0) ? m.dimensions()[0] : limit_x;
    int ly = (limit_y < 0) ? m.dimensions()[0] : limit_y;

    cout << "Matrix contents:\n";
    for (int i = 0; i < ly; i++) {
        for (int j = 0; j < lx; j++) {
            cout << (int)m(i, j) << "\t";
        } cout << endl;
    } cout << endl;
}

inline void
printMatrix(TB_Matrix3D m, int channel, int limit_x=-1, int limit_y=-1)
{
    int lx = (limit_x < 0) ? m.dimensions()[0] : limit_x;
    int ly = (limit_y < 0) ? m.dimensions()[0] : limit_y;

    cout << "Matrix contents, channel " << channel << ":\n";
    for (int i = 0; i < ly; i++) {
        for (int j = 0; j < lx; j++) {
            cout << (int)m(channel, i, j) << "\t";
        } cout << endl;
    } cout << endl;
}

/////////////////////////////////
// Layer connection utilities. //
/////////////////////////////////

// Helper function to connect args_a's input to args_b's output pointers.
template<typename T, typename U> inline void
connectInputOutput(T & args_a, U & args_b)
{
    args_a.input = args_b.output;
    return;
}

template<typename T> inline void
connectResidual(end_residual_layer_args & args_a, T & args_b)
{   
    args_a.residual = args_b.output;
    return;
}

// Helper function to generate output vector buffer.
template<typename T> inline void
generateSingleOutputVector(T & args)
{
    args.output = new TB_Vector[1];
    args.output[0] = TB_Vector(args.output_size);
    args.output[0].setZero();   
    return;
}

// Helper function to generate output matrix buffer.
template<typename T> inline void
generateSingleOutputMatrix3D(T & args)
{
    args.output = new TB_Matrix3D[1];
    args.output[0] = TB_Matrix3D(args.output_c,
        args.output_h, args.output_w);
    std::cout << "Args " << args.output_c << "," << args.output_h <<  "," << args.output_w << std::endl;
    args.output[0].setZero(); 
    return;
}

// Helper function to generate ping-pong output vector buffer.
template<typename T> inline void
generatePingPongOutputVector(T & args)
{
    args.output = new TB_Vector[2];
    args.output[0] = TB_Vector(args.output_size);
    args.output[1] = TB_Vector(args.output_size);
    args.output[0].setZero(); 
    args.output[1].setZero();   
    return;
}

// Helper function to generate ping-pong output matrix buffer.
template<typename T> inline void
generatePingPongOutputMatrix3D(T & args)
{
    args.output = new TB_Matrix3D[2];
    args.output[0] = TB_Matrix3D(args.output_c,
        args.output_h, args.output_w);
    args.output[1] = TB_Matrix3D(args.output_c,
        args.output_h, args.output_w);
    args.output[0].setZero(); 
    args.output[1].setZero(); 
    return;
}

// Helper function to generate output point, matrix/vector.
template<typename T> inline void
generateSingleOutputDataStructure(T & args)
{
    generateSingleOutputMatrix3D(args);
    return;
}

// Helper functions to generate output vector.
template<> inline void
generateSingleOutputDataStructure<flatten_layer_args>(
    flatten_layer_args & args)
{
    generateSingleOutputVector(args);
    return;
}

template<> inline void
generateSingleOutputDataStructure<fc_layer_args>(fc_layer_args & args)
{
    generateSingleOutputVector(args);
    return;
}

#if defined (AIMC)
template<> inline void
generateSingleOutputDataStructure<analog_case1_fc_layer_args>(
    analog_case1_fc_layer_args & args)
{
    generateSingleOutputVector(args);
    return;
}

template<> inline void
generateSingleOutputDataStructure<analog_case2_fc_layer_args>(
    analog_case2_fc_layer_args & args)
{
    generateSingleOutputVector(args);
    return;
}

template<> inline void
generateSingleOutputDataStructure<analog_case3_fc_layer_args>(
    analog_case3_fc_layer_args & args)
{
    generateSingleOutputVector(args);
    return;
}

template<> inline void
generateSingleOutputDataStructure<analog_case4_fc_layer_args>(
    analog_case4_fc_layer_args & args)
{
    generateSingleOutputVector(args);
    return;
}
#endif // AIMC

// Helper function to generate output point, matrix/vector.
template<typename T> inline void
generatePingPongOutputDataStructure(T & args)
{
    generatePingPongOutputMatrix3D(args);
    return;
}

// Helper functions to generate output vector.
template<> inline void
generatePingPongOutputDataStructure<flatten_layer_args>(
    flatten_layer_args & args)
{
    generatePingPongOutputVector(args);
    return;
}

template<> inline void
generatePingPongOutputDataStructure<fc_layer_args>(fc_layer_args & args)
{
    generatePingPongOutputVector(args);
    return;
}

#if defined (AIMC)
template<> inline void
generatePingPongOutputDataStructure<analog_case1_fc_layer_args>(
    analog_case1_fc_layer_args & args)
{
    generatePingPongOutputVector(args);
    return;
}

template<> inline void
generatePingPongOutputDataStructure<analog_case2_fc_layer_args>(
    analog_case2_fc_layer_args & args)
{
    generatePingPongOutputVector(args);
    return;
}

template<> inline void
generatePingPongOutputDataStructure<analog_case3_fc_layer_args>(
    analog_case3_fc_layer_args & args)
{
    generatePingPongOutputVector(args);
    return;
}

template<> inline void
generatePingPongOutputDataStructure<analog_case4_fc_layer_args>(
    analog_case4_fc_layer_args & args)
{
    generatePingPongOutputVector(args);
    return;
}
#endif // AIMC

// Connect and initialize layer input/output structures based on buffer type.
template<typename T, typename U> inline void
connectLayers(T & args_in_cur, U & args_out_prev, bool printInfo=true)
{
    switch (args_in_cur.buffer_type) {
        case SINGLE_BUFFER_TYPE: {
            // Connect input data structure and add output data structure.
            connectInputOutput(args_in_cur, args_out_prev);
            generateSingleOutputDataStructure(args_in_cur);
            break;
        }
        // TODO: Implement ping-pong buffer connections.
        case PING_PONG_BUFFER_TYPE: {
            connectInputOutput(args_in_cur, args_out_prev);
            generatePingPongOutputDataStructure(args_in_cur);
            break;
        }
        default: {break;}
    }

    if (printInfo) {
        printLayerInfo(&args_in_cur);
    }

    return;
}

#endif // __UTILITIES_HH__
