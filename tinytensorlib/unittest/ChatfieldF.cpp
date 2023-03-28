/* Copyright EPFL 2023
 * Joshua Klein
 *
 * Implementation of Chatfield-F variant CNN based on VGG8 [1].
 *
 * [1] Chatfield, Ken, et al. "Return of the devil in the details: Delving deep
 * into convolutional nets." arXiv preprint arXiv:1405.3531 (2014).
 *
 */

#include <iostream>

#include "../tinytensorlib.hh"
#include "ChatfieldF.hh"

using namespace std;


int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;

    generateSingleOutputDataStructure(conv1);
    printLayerInfo(&conv1);
    
    connectLayers(pool1, conv1);
    connectLayers(conv2, pool1);
    connectLayers(pool2, conv2);
    connectLayers(conv3, pool2);
    connectLayers(conv4, conv3);
    connectLayers(conv5, conv4);
    connectLayers(pool3, conv5);
    connectLayers(flatten1, pool3);
    connectLayers(dense1, flatten1);
    connectLayers(dense2, dense1);

    dense3.input = dense2.output;
    printLayerInfo(&dense3);

    // Do inference.
    sys_info += system("m5 resetstats");
    for (int inf = 0; inf < T_x; inf++)
    {   
        cout << "Inference " << inf << endl;
        doLayer(conv1, inf, 0);
        doLayer(pool1);

        doLayer(conv2);
        doLayer(pool2);

        doLayer(conv3);
        doLayer(conv4);
        doLayer(conv5);
        doLayer(pool3);
        doLayer(flatten1);
        doLayer(dense1);
        doLayer(dense2);
        doLayer(dense3, 0, inf);
    }

    // Finish and clean up.
    sys_info += system("m5 exit");

    printVector(dense3.output[T_x-1], 5);

    delete[] conv1.input;
    delete[] pool1.input;
    delete[] conv2.input;
    delete[] pool2.input;
    delete[] conv3.input;
    delete[] conv4.input;
    delete[] conv5.input;
    delete[] pool3.input;
    delete[] flatten1.input;
    delete[] dense1.input;
    delete[] dense2.input;
    delete[] dense3.input;
    delete[] dense3.output;

    return 0;
}
