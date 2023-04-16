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
#include "Chatfield_layers.hh"

using namespace std;


int
main(int argc, char * argv[])
{
    // Initialize threading stuff.
    pthread_t thread0, thread1, thread2, thread3, thread4, thread5, thread6,
    thread7;
    int n_threads = 8;
    initSyncStructures(n_threads);


    generatePingPongOutputDataStructure(conv1);
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

    // Connect thread sync arrays.
    connectSyncStructures(conv1, conv1.thread_n);
    connectSyncStructures(pool1, pool1.thread_n);
    connectSyncStructures(conv2, conv2.thread_n);
    connectSyncStructures(pool2, pool2.thread_n);
    connectSyncStructures(conv3, conv3.thread_n);
    connectSyncStructures(conv4, conv4.thread_n);
    connectSyncStructures(conv5, conv5.thread_n);
    connectSyncStructures(pool3, pool3.thread_n);
    connectSyncStructures(flatten1, flatten1.thread_n);
    connectSyncStructures(dense1, dense1.thread_n);
    connectSyncStructures(dense2, dense2.thread_n);
    connectSyncStructures(dense3, dense3.thread_n);


    // Do inference.
    cout << "Starting inference...\n";
    pthread_create(&thread0, NULL, run_layer_thread0, NULL);
    pthread_create(&thread1, NULL, run_layer_thread1, NULL);
    pthread_create(&thread2, NULL, run_layer_thread2, NULL);
    pthread_create(&thread3, NULL, run_layer_thread3, NULL);
    pthread_create(&thread4, NULL, run_layer_thread4, NULL);
    pthread_create(&thread5, NULL, run_layer_thread5, NULL);
    pthread_create(&thread6, NULL, run_layer_thread6, NULL);
    pthread_create(&thread7, NULL, run_layer_thread7, NULL);

    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);
    pthread_join(thread5, NULL);
    pthread_join(thread6, NULL);
    pthread_join(thread7, NULL);

    // Finish and clean up.
    printVector(dense3.output[T_x-1], 15);

    cleanSyncStructures();
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
