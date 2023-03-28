/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file contains utilities for using POSIX threads (pthreads).  Methods
 * here were originally written and modified by Ruben Braojos Lopez and Yasir
 * Qureshi, respectively, for use by the Embedded Systems Laboratory in CNN
 * solver.
 *
 */

#ifndef __THREADS_HH__
#define __THREADS_HH__

#include <cmath>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Global conditional/mutex arrays for thread synchronization.
pthread_cond_t * cond_vars;
pthread_mutex_t * cnt_mutex;

// Set the function affinity.
inline int
stickThisThreadToCore(int core_id)
{
   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
   if (core_id < 0 || core_id >= num_cores)
      return EINVAL;

   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id, &cpuset);

   pthread_t current_thread = pthread_self();
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

// Assign indexes for mutexes/semaphores.  Like the layer structures
// generation, only concerns self with current and previous layer.
template<typename T> inline void
connectSyncStructures(T & args, int idx)
{
    if (args.isFirstLayer) {
        args.cond_var_nxt_lyr_idx = 0;
        args.cnt_mutex_nxt_lyr_idx = 0;
        args.cond_var_prv_lyr_idx = -1;
        args.cnt_mutex_prv_lyr_idx = -1;
    } else if (args.isLastLayer) {
        args.cond_var_nxt_lyr_idx = -1;
        args.cnt_mutex_nxt_lyr_idx = -1;
        args.cond_var_prv_lyr_idx = idx - 1;
        args.cnt_mutex_prv_lyr_idx = idx - 1;
    } else {
        args.cond_var_prv_lyr_idx = idx - 1;
        args.cnt_mutex_prv_lyr_idx = idx - 1;
        args.cond_var_nxt_lyr_idx = idx;
        args.cnt_mutex_nxt_lyr_idx = idx;
    }

    return;
}

// Used for thread coordination and ping-pong buffers.
inline void
initSyncStructures(int n_layers)
{
    cond_vars = new pthread_cond_t[n_layers-1];
    cnt_mutex = new pthread_mutex_t[n_layers-1];

    for (int i = 0; i < n_layers-1; i++) {
        pthread_cond_init(&cond_vars[i], NULL);
        pthread_mutex_init(&cnt_mutex[i], NULL);
    }

    return;
}

inline void
cleanSyncStructures()
{
    if (cond_vars != NULL) delete[] cond_vars;
    if (cnt_mutex != NULL) delete[] cnt_mutex;
    return;
}

#endif // __THREADS_HH__
