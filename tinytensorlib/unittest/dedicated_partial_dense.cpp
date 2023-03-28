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

analog_case1_fc_layer_args ana_dense1 = analog_case1_fc_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    true,               // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    1024,               // Input size.
    1024,               // Output size.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    1024,               // AIMC Tile height.
    1024                // AIMC Tile width.
);

analog_case2_fc_layer_args ana_dense2 = analog_case2_fc_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    SINGLE_BUFFER_TYPE, // Buffer type.
    ana_dense1,         // Input layer.
    1024,               // Output size.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    750,                // AIMC Tile height.
    1024                // AIMC Tile width.
);

analog_case3_fc_layer_args ana_dense3 = analog_case3_fc_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    SINGLE_BUFFER_TYPE, // Buffer type.
    ana_dense2,         // Input size.
    1024,               // Output size.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    1024,               // AIMC Tile height.
    561                 // AIMC Tile width.
);

analog_case4_fc_layer_args ana_dense4 = analog_case4_fc_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    false,              // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    1024,               // Input size.
    1024,               // Output size.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE,      // Activation function.
    231,                // AIMC Tile height.
    561                 // AIMC Tile width.
);


int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;
    cout << "Initializing...\n";

    generateSingleOutputDataStructure(ana_dense1);
    printLayerInfo(&ana_dense1);

    connectLayers(ana_dense2, ana_dense1);
    connectLayers(ana_dense3, ana_dense2);

    ana_dense4.input = ana_dense3.output;
    printLayerInfo(&ana_dense4);

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        doLayer(ana_dense1);
        doLayer(ana_dense2);
        doLayer(ana_dense3);
        doLayer(ana_dense4);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");
    printVector(ana_dense4.output[T_x-1], 10);

    delete[] ana_dense1.input;
    delete[] ana_dense2.input;
    delete[] ana_dense3.input;
    delete[] ana_dense4.input;
    delete[] ana_dense4.output;

    return 0;
}
