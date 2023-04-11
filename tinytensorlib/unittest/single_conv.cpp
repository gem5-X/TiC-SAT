/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * Unit test for: Single whole convolutional layer, testing different doLayer
 * functions specifically for convolution operations.
 *
 */

#include <chrono>
#include <iostream>
#include <string>

#include "../tinytensorlib.hh"

using namespace std;

// Number of inferences and layer definitions.
const int T_x = 3;

conv_layer_args conv1 = conv_layer_args(
    0,                  // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    true,               // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    13,                 // Input height.
    13,                 // Input width.
    99,                // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    389,                // Number of filters.
    1,                  // Stride.
#if defined (SA)
    true,               // Are we using AIMC tiles?
    SA_SIZE,                 // Allocated tile height? (-1 = infinite)
#endif
    1,                  // Padding.
    NO_NORM_TYPE,       // Normalization.
    NO_ACT_TYPE         // Activation function.
);

chrono::nanoseconds convTime{};

int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;
    cout << "Initializing...\n";
    printLayerInfo(&conv1);

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        auto t1 = chrono::high_resolution_clock::now();
        doLayer(conv1);
        auto t2 = chrono::high_resolution_clock::now();

        convTime += chrono::duration_cast<chrono::nanoseconds>(t2 - t1);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");
    printMatrix(conv1.output[T_x-1], 0, 5, 5);

    cout << "Convolution time: " << convTime.count() << endl;

//    delete[] conv1.input;
//    delete[] conv1.output;

    return 0;
}
