/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * Unit test for: Partial conv layer calculation using mixed analog-
 * digital MVMs.
 *
 */

#include <iostream>
#include <string>

#include "tinytensorlib.hh"

using namespace std;

// Number of inferences and layer definitions.
const int T_x = 3;
int layer_num = 0;

conv_layer_args conv1 = conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    true,               // Is first layer.
    false,              // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    13,                 // Input height.
    13,                 // Input width.
    256,                // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    384,                // Number of filters.
    1,                  // Stride.
#if defined (AIMC)
    true,               // Are we using AIMC tiles?
    -1,                 // Allocated tile height? (-1 = infinite)
    -1,                 // Allocated tile width? (-1 = infinite)
#endif
    1,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    NO_ACT_TYPE         // Activation function.
);

conv_layer_args conv2 = conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    false,              // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    conv1.output_h,     // Input height.
    conv1.output_w,     // Input width.
    conv1.output_c,     // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    384,                // Number of filters.
    1,                  // Stride.
#if defined (AIMC)
    true,               // Are we using AIMC tiles?
    -1,                 // Allocated tile height? (-1 = infinite)
    -1,                 // Allocated tile width? (-1 = infinite)
#endif
    1,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    NO_ACT_TYPE         // Activation function.
);


int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;
    cout << "Initializing...\n";

    conv1.output = new TB_Matrix3D[1];
    conv1.output[0] = TB_Matrix3D(
        conv1.output_c, conv1.output_h, conv1.output_w);
    conv1.output[0].setZero();
    conv2.input = conv1.output;

    printLayerInfo(&conv1);
    printLayerInfo(&conv2);

    // Accept command line arguments for resizing AIMC tiles.
#if defined (AIMC)
    // conv1 tile height.
    if (argc > 2) {
        conv1.aimc_h = stoi(argv[1]);
        conv1.aimc_w = stoi(argv[2]);
    }

    if (argc > 4) {
        conv2.aimc_h = stoi(argv[3]);
        conv2.aimc_w = stoi(argv[4]);
    }
    cout << "Conv 1 AIMC tile dims: " << conv1.aimc_h << "x" << conv1.aimc_w << endl;
    cout << "Conv 2 AIMC tile dims: " << conv2.aimc_h << "x" << conv2.aimc_w << endl;
#endif

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        doLayer(conv1);
        doLayer(conv2);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");
    printMatrix(conv2.output[T_x-1], 0, 5, 5);

    delete[] conv1.input;
    delete[] conv2.input;
    delete[] conv2.output;

    return 0;
}
