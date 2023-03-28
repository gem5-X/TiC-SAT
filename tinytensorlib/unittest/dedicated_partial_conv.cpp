/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * Unit test for: Partial dense layer calculation using dedicated layer
 * definitions for mixed analog-digital MVMs.
 *
 */

#include <iostream>
#include <string>

#include "tinytensorlib.hh"

using namespace std;

// Number of inferences and layer definitions.
const int T_x = 3;
int layer_num = 0;

analog_case1_conv_layer_args ana_conv1 = analog_case1_conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    true,               // Is first layer.
    false,              // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    32,                 // Input height.
    32,                 // Input width.
    3,                  // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    64,                 // Number of filters.
    1,                  // Stride.
    0,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    1024,               // AIMC Tile height.
    1024                // AIMC Tile width.
);

analog_case2_conv_layer_args ana_conv2 = analog_case2_conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    SINGLE_BUFFER_TYPE, // Buffer type.
    ana_conv1,          // Input layer.
    3,                  // Kernel height.
    3,                  // Kernel width.
    192,                // Number of filters.
    1,                  // Stride.
    0,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    512,                // AIMC Tile height.
    1024                // AIMC Tile width.
);

analog_case3_conv_layer_args ana_conv3 = analog_case3_conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    SINGLE_BUFFER_TYPE, // Buffer type.
    ana_conv2,         // Input size.
    3,                  // Kernel height.
    3,                  // Kernel width.
    384,                // Number of filters.
    1,                  // Stride.
    0,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    NO_ACT_TYPE,        // Activation function.
    2048,               // AIMC Tile height.
    256                 // AIMC Tile width.
);

analog_case4_conv_layer_args ana_conv4 = analog_case4_conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    false,              // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    ana_conv3.output_h, // Input height.
    ana_conv3.output_w, // Input width.
    ana_conv3.output_c, // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    256,                // Number of filters.
    1,                  // Stride.
    0,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    2048,               // AIMC Tile height.
    128                 // AIMC Tile width.
);

int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;
    cout << "Initializing...\n";

    generateSingleOutputDataStructure(ana_conv1);
    cout << "Generated ana_conv1 outputs.\n";
    printLayerInfo(&ana_conv1);

    connectLayers(ana_conv2, ana_conv1);
    cout << "Connected ana_conv2.\n";
    connectLayers(ana_conv3, ana_conv2);
    cout << "Connected ana_conv3.\n";

    ana_conv4.input = ana_conv3.output;
    cout << "Connected ana_conv4.\n";
    printLayerInfo(&ana_conv4);

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        doLayer(ana_conv1);
        doLayer(ana_conv2);
        doLayer(ana_conv3);
        doLayer(ana_conv4);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");
    printVector(ana_conv4.output[T_x-1], 10);

    delete[] ana_conv1.input;
    delete[] ana_conv2.input;
    delete[] ana_conv3.input;
    delete[] ana_conv4.input;
    delete[] ana_conv4.output;

    return 0;
}
