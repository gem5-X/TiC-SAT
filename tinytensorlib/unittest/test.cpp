/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file is just for testing individual functions and utilities and
 * should not be otherwise used.
 *
 */

#include <iostream>
#include <chrono>

#include "tinytensorlib.hh"

using namespace std;

// Number of inferences.
const int T_x = 3;
int layer_num = 0;

dwconv_layer_args dwconv1 = dwconv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    true,               // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    112,                // Input height.
    112,                // Input width.
    32,                 // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    16,                 // Number of filters.
    1,                  // Stride.
#if defined (AIMC)
    true,               // Are we using AIMC tiles?
    -1,                 // Allocated tile height? (-1 = infinite)
    -1,                 // Allocated tile width? (-1 = infinite)
#endif
    1,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    RELU6_ACT_TYPE      // Activation function.
);

chrono::nanoseconds convTime{};
chrono::nanoseconds poolTime{};
chrono::nanoseconds flatTime{};
chrono::nanoseconds denseTime{};


int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;

    // Initialize everything.
    cout << "Initializing...\n";
    generateSingleOutputDataStructure(dwconv1);
    printLayerInfo(&dwconv1);

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        DepthwiseConvolution2D(&dwconv1, dwconv1.input[0], dwconv1.weights, dwconv1.output[0]);
        Activation(dwconv1.output[0], dwconv1.activation);
        printMatrix(dwconv1.output[0], 1, 3, 3);
        // auto t12 = chrono::high_resolution_clock::now();

        // convTime += chrono::duration_cast<chrono::nanoseconds>(t1 - t0)
        //     + chrono::duration_cast<chrono::nanoseconds>(t3 - t2)
        //     + chrono::duration_cast<chrono::nanoseconds>(t5 - t4)
        //     + chrono::duration_cast<chrono::nanoseconds>(t6 - t5)
        //     + chrono::duration_cast<chrono::nanoseconds>(t7 - t6);

        // poolTime += chrono::duration_cast<chrono::nanoseconds>(t2 - t1)
        //     + chrono::duration_cast<chrono::nanoseconds>(t4 - t3)
        //     + chrono::duration_cast<chrono::nanoseconds>(t8 - t7);

        // flatTime += chrono::duration_cast<chrono::nanoseconds>(t9 - t8);

        // denseTime += chrono::duration_cast<chrono::nanoseconds>(t10 - t9)
        //     + chrono::duration_cast<chrono::nanoseconds>(t11 - t10)
        //     + chrono::duration_cast<chrono::nanoseconds>(t12 - t11);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");

    // cout << "Convs time: " << convTime.count() << endl;
    // cout << "Pool time:  " << poolTime.count() << endl;
    // cout << "Flat time:  " << flatTime.count() << endl;
    // cout << "Dense time: " << denseTime.count() << endl;

    return 0;
}
