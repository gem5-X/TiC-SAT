/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * Unit test for: pthreaded conv layers.
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
    PING_PONG_BUFFER_TYPE,// Buffer type.
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
conv_layer_args * conv1_ptr = &conv1;

conv_layer_args conv2 = conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    1,                  // Assigned thread.
    false,              // Is first layer.
    false,              // Is last layer.
    PING_PONG_BUFFER_TYPE,// Buffer type.
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
conv_layer_args * conv2_ptr = &conv2;

conv_layer_args conv3 = conv_layer_args(
    layer_num++,        // Layer number.
    T_x,                // Number of inferences.
    2,                  // Assigned thread.
    false,              // Is first layer.
    true,               // Is last layer.
    PING_PONG_BUFFER_TYPE,// Buffer type.
    conv2.output_h,     // Input height.
    conv2.output_w,     // Input width.
    conv2.output_c,     // Input channels.
    3,                  // Kernel height.
    3,                  // Kernel width.
    256,                // Number of filters.
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
conv_layer_args * conv3_ptr = &conv3;

// Thread work.
void *
run_layer(void * args_ptr)
{
    // Get args, set thread and buffer.
    conv_layer_args & args = *(conv_layer_args*)args_ptr;
    stickThisThreadToCore(args.thread_n);
    args.pong_idx = 0;

    // Enter inference loop.
    if (!args.isFirstLayer) {
        pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do inference.
    for (int inf = 0; inf < args.T_x; inf++) {
        // Wait for previous thread to populate buffer.
        cout << "Thread " << args.thread_n
            << "; Inference " << inf 
            << "; Buffer " << (int)args.pong_idx << endl;
        if (!args.isFirstLayer) {
            pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                &cnt_mutex[args.cnt_mutex_prv_lyr_idx]); 
        }

        // Do thread work.
        if (args.isFirstLayer) {
            doLayer(args, inf, args.pong_idx);
        } else if (args.isLastLayer) {
            doLayer(args, args.pong_idx, inf);
        } else {
            doLayer(args, args.pong_idx, args.pong_idx);
        }

        // Current inference buffer is populated, so inform next layer and set
        // next inference buffer.
        if (!args.isLastLayer) {
            pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_nxt_lyr_idx]);
            pthread_cond_signal(&cond_vars[args.cond_var_nxt_lyr_idx]);
            pthread_mutex_unlock(&cnt_mutex[args.cnt_mutex_nxt_lyr_idx]);
        }

        args.pong_idx = (args.pong_idx + 1) % 2;
    }

    // Last unlock.
    if (!args.isFirstLayer) {
        pthread_mutex_unlock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]); 
    }

    cout << "Thread " << args.thread_n << " finished!\n";

    return 0;
}

int
main(int argc, char * argv[])
{
    // Initialize threading stuff.
    int sys_info = 0;
    pthread_t thread1, thread2, thread3;
    initSyncStructures(3);

    // Initialize layer buffers and threading vars.
    cout << "Initializing...\n";

    generatePingPongOutputMatrix3D(conv1);
    connectSyncStructures(conv1, 0);
    printLayerInfo(&conv1);

    connectLayers(conv2, conv1);
    connectSyncStructures(conv2, 1);

    conv3.input = conv2.output;
    connectSyncStructures(conv3, 2);
    printLayerInfo(&conv3);



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

    if (argc > 6) {
        conv3.aimc_h = stoi(argv[5]);
        conv3.aimc_w = stoi(argv[6]);
    }
    cout << "Conv 1 AIMC tile dims: " << conv1.aimc_h << "x" << conv1.aimc_w << endl;
    cout << "Conv 2 AIMC tile dims: " << conv2.aimc_h << "x" << conv2.aimc_w << endl;
    cout << "Conv 3 AIMC tile dims: " << conv3.aimc_h << "x" << conv3.aimc_w << endl;
#endif

    // Do inference.
    sys_info += system("m5 resetstats");
    pthread_create(&thread1, NULL, run_layer, conv1_ptr);
    pthread_create(&thread2, NULL, run_layer, conv2_ptr);
    pthread_create(&thread3, NULL, run_layer, conv3_ptr);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);

    // Finish and clean up.
    sys_info += system("m5 exit");
    printMatrix(conv3.output[T_x-1], 0, 5, 5);

    cleanSyncStructures();
    delete[] conv1.input;
    delete[] conv2.input;
    delete[] conv2.output;

    return 0;
}
