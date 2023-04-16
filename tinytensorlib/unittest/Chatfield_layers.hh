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
inline void
thread0Work(int idx = 0, int inf = 0)
{
    doLayer(conv1, inf, 0);
    doLayer(pool1, 0, idx);
    return;
}

inline void
thread1Work(int idx = 0)
{
    doLayer(conv2, idx, 0);
    doLayer(pool2, 0, idx);
    return;
}

inline void
thread2Work(int idx = 0)
{
    doLayer(conv3, idx, idx);
    return;
}

inline void
thread3Work(int idx = 0)
{
    doLayer(conv4, idx, idx);
    return;
}

inline void
thread4Work(int idx = 0)
{
    doLayer(conv5, idx, 0);
    doLayer(pool3, 0, 0);
    doLayer(flatten1, 0, idx);
    return;
}

inline void
thread5Work(int idx = 0)
{
    doLayer(dense1, idx, idx);
    return;
}

inline void
thread6Work(int idx = 0)
{
    doLayer(dense2, idx, idx);
    return;
}

inline void
thread7Work(int idx = 0, int inf = 0)
{
    doLayer(dense3, idx, inf);
    return;
}

#endif //TINYTENSORLIB_CHATFIELD_LAYERS_HH
