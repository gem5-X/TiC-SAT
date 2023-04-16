/* Copyright EPFL 2023
 * Joshua Klein
 *
 * Multithreaded layer definitions for custom multi-core Chatfield CNN based
 * on [1].
 *
 * [1] Chatfield, Ken, et al. "Return of the devil in the details: Delving deep
 * into convolutional nets." arXiv preprint arXiv:1405.3531 (2014).
 *
 */

#ifndef TINYTENSORLIB_CHATFIELD_LAYERS_HH
#define TINYTENSORLIB_CHATFIELD_LAYERS_HH

// Thread work.
void *
run_layer_thread0(void * args_ptr)
{
    // Get args, set thread and buffer.
#if defined (AIMC)
analog_case1_conv_layer_args & args = conv1;
#else
conv_layer_args & args = conv1;
#endif
stickThisThreadToCore(args.thread_n);
args.pong_idx = 0;

// Enter inference loop.
if (!args.isFirstLayer) {
    pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
}

// Do inference.
for (int inf = 0; inf < args.T_x; inf++) {
    // Wait for previous thread to populate buffer.
    if (!args.isFirstLayer) {
        pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                          &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do thread work.
    doLayer(conv1, inf, 0);
    doLayer(pool1, 0, args.pong_idx);

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

void *
run_layer_thread1(void * args_ptr)
{
    // Get args, set thread and buffer.
#if defined (AIMC)
analog_case2_conv_layer_args & args = conv2;
#else
conv_layer_args & args = conv2;
#endif
stickThisThreadToCore(args.thread_n);
args.pong_idx = 0;

// Enter inference loop.
if (!args.isFirstLayer) {
    pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
}

// Do inference.
for (int inf = 0; inf < args.T_x; inf++) {
    // Wait for previous thread to populate buffer.
    if (!args.isFirstLayer) {
        pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                          &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do thread work.
    doLayer(conv2, args.pong_idx, 0);
    doLayer(pool2, 0, args.pong_idx);

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

void *
run_layer_thread2(void * args_ptr)
{
    // Get args, set thread and buffer.
#if defined (AIMC) && defined (CHATFIELDF)
analog_case2_conv_layer_args & args = conv3;
#elif defined (AIMC)
analog_case4_conv_layer_args & args = conv3;
#else
conv_layer_args & args = conv3;
#endif
stickThisThreadToCore(args.thread_n);
args.pong_idx = 0;

// Enter inference loop.
if (!args.isFirstLayer) {
    pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
}

// Do inference.
for (int inf = 0; inf < args.T_x; inf++) {
    // Wait for previous thread to populate buffer.
    if (!args.isFirstLayer) {
        pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                          &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do thread work.
    doLayer(conv3, args.pong_idx, args.pong_idx);

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

void *
run_layer_thread3(void * args_ptr)
{
    // Get args, set thread and buffer.
#if defined (AIMC) && defined (CHATFIELDF)
analog_case2_conv_layer_args & args = conv4;
#elif defined (AIMC)
analog_case4_conv_layer_args & args = conv4;
#else
conv_layer_args & args = conv4;
#endif
stickThisThreadToCore(args.thread_n);
args.pong_idx = 0;

// Enter inference loop.
if (!args.isFirstLayer) {
    pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
}

// Do inference.
for (int inf = 0; inf < args.T_x; inf++) {
    // Wait for previous thread to populate buffer.
    if (!args.isFirstLayer) {
        pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                          &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do thread work.
    doLayer(conv4, args.pong_idx, args.pong_idx);

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

void *
run_layer_thread4(void * args_ptr)
{
    // Get args, set thread and buffer.
#if defined (AIMC) && defined (CHATFIELDF)
analog_case2_conv_layer_args & args = conv5;
#elif defined (AIMC)
analog_case4_conv_layer_args & args = conv5;
#else
conv_layer_args & args = conv5;
#endif
stickThisThreadToCore(args.thread_n);
args.pong_idx = 0;

// Enter inference loop.
if (!args.isFirstLayer) {
    pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
}

// Do inference.
for (int inf = 0; inf < args.T_x; inf++) {
    // Wait for previous thread to populate buffer.
    if (!args.isFirstLayer) {
        pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                          &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do thread work.
    doLayer(conv5, args.pong_idx, 0);
    doLayer(pool3, 0, 0);
    doLayer(flatten1, 0, args.pong_idx);

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

void *
run_layer_thread5(void * args_ptr)
{
    // Get args, set thread and buffer.
    fc_layer_args & args = dense1;
    stickThisThreadToCore(args.thread_n);
    args.pong_idx = 0;

    // Enter inference loop.
    if (!args.isFirstLayer) {
        pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do inference.
    for (int inf = 0; inf < args.T_x; inf++) {
        // Wait for previous thread to populate buffer.
        if (!args.isFirstLayer) {
            pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                              &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
        }

        // Do thread work.
        doLayer(dense1, args.pong_idx, args.pong_idx);

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

void *
run_layer_thread6(void * args_ptr)
{
    // Get args, set thread and buffer.
    fc_layer_args & args = dense2;
    stickThisThreadToCore(args.thread_n);
    args.pong_idx = 0;

    // Enter inference loop.
    if (!args.isFirstLayer) {
        pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do inference.
    for (int inf = 0; inf < args.T_x; inf++) {
        // Wait for previous thread to populate buffer.
        if (!args.isFirstLayer) {
            pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                              &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
        }

        // Do thread work.
        doLayer(dense2, args.pong_idx, args.pong_idx);

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

void *
run_layer_thread7(void * args_ptr)
{
    // Get args, set thread and buffer.
    int sys_info = 0;
    fc_layer_args & args = dense3;
    stickThisThreadToCore(args.thread_n);
    args.pong_idx = 0;

    // Enter inference loop.
    if (!args.isFirstLayer) {
        pthread_mutex_lock(&cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
    }

    // Do inference.
    for (int inf = 0; inf < args.T_x; inf++) {
        // Wait for previous thread to populate buffer.
        if (!args.isFirstLayer) {
            pthread_cond_wait(&cond_vars[args.cond_var_prv_lyr_idx],
                              &cnt_mutex[args.cnt_mutex_prv_lyr_idx]);
        }

        // Do thread work.
        if (inf == warmup_infs) {
            sys_info += system("m5 resetstats");
        }
        doLayer(dense3, args.pong_idx, inf);
        cout << "Finished inference " << inf << "!\n";
        if (inf == warmup_infs + roi_infs) {
            sys_info += system("m5 exit");
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

#endif //TINYTENSORLIB_CHATFIELD_LAYERS_HH
