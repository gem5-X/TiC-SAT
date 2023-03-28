/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * Unit test for: Partial dense layer calculation using mixed analog-
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

fc_layer_args dense1 = fc_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    true,               // Is first layer.
    false,              // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    4096,               // Input size.
#if defined (AIMC)
    true,               // Are we using AIMC tiles?
    -1,                 // Allocated tile height? (-1 = infinite)
    -1,                 // Allocated tile width? (-1 = infinite)
#endif
    4096,               // Output size.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE       // Activation function.
);

fc_layer_args dense2 = fc_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    0,                  // Assigned thread.
    false,              // Is first layer.
    true,               // Is last layer.
    SINGLE_BUFFER_TYPE, // Buffer type.
    dense1.output_size, // Input size.
#if defined (AIMC)
    true,               // Are we using AIMC tiles?
    -1,                 // Allocated tile height? (-1 = infinite)
    -1,                 // Allocated tile width? (-1 = infinite)
#endif
    1024,               // Output size.
    NO_NORM_TYPE,       // Normalization.
    RELU_ACT_TYPE       // Activation function.
);


int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;
    cout << "Initializing...\n";

    dense1.output = new TB_Vector[1];
    dense1.output[0] = TB_Vector(dense1.output_size);
    dense1.output[0].setZero();
    dense2.input = dense1.output;

    printLayerInfo(&dense1);
    printLayerInfo(&dense2);

    // Accept command line arguments for resizing AIMC tiles.
#if defined (AIMC)
    // dense1 tile height.
    if (argc > 2) {
        dense1.aimc_h = stoi(argv[1]);
        dense1.aimc_w = stoi(argv[2]);
    }

    if (argc > 4) {
        dense2.aimc_h = stoi(argv[3]);
        dense2.aimc_w = stoi(argv[4]);
    }
    cout << "Dense 1 AIMC tile dims: " << dense1.aimc_h << "x" << dense1.aimc_w << endl;
    cout << "Dense 2 AIMC tile dims: " << dense2.aimc_h << "x" << dense2.aimc_w << endl;
#endif

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        doLayer(dense1);
        doLayer(dense2);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");
    printVector(dense2.output[T_x-1], 10);

    delete[] dense1.input;
    delete[] dense2.input;
    delete[] dense2.output;

    return 0;
}
