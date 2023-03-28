/*
 * Copyright EPFL 2023
 * Joshua Klein
 *
 * Contains function prototypes and implementations ported from aimclib for use
 * with Eigen C++ tensors.
 *
 */

#ifndef __AIMC_UTILITIES_HH__
#define __AIMC_UTILITIES_HH__

#include "aimc.hh"

// int8_t vector queueing from array.
inline void
queueVector(int size, TB_Vector & v, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    uint32_t tmp = 0;

    // Pack 4 int8 values into tmp, and queue into AIMC input memory.
    for (int i = 0; i < size; i++) {
        if ((i % 4 == 0) && i > 0) {
            aimcQueue(tmp, tid);
            tmp = 0;
        }

        tmp |= (v(i) & 0xff) << (8 * (i % 4));
    }

    // Queue the leftover values, in case |v| % 4 != 0.
    aimcQueue(tmp, tid);
#endif

    return;
}

// int8_t vector queueing from array.
inline void
queueVector(int size, TB_Matrix3D & m, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    uint32_t tmp = 0;
    int dim_ch = m.dimensions()[0];
    int dim_row = m.dimensions()[1];
    int dim_col = m.dimensions()[2];

    // Pack 4 int8 values into tmp, and queue into AIMC input memory.
    for (int ch = 0, offset = size; ch < dim_ch; ch++) {
        for (int i = 0; i < dim_row; i++) {
            for (int j = 0; j < dim_col && offset; j++, offset--) {
                if ((i % 4 == 0) && i > 0) {
                    aimcQueue(tmp, tid);
                    tmp = 0;
                }

                tmp |= (m(ch, i, j) & 0xff) << (8 * (i % 4));
            }
        }
    }

    // Queue the leftover values, in case |v| % 4 != 0.
    aimcQueue(tmp, tid);
#endif

    return;
}

// int8_t matrix queueing from patch row.
inline void
queuePatch(TB_Matrix2D & m, int patch_row, int patch_size, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    uint32_t tmp = 0;

    // Pack 4 int8 values into tmp, and queue into AIMC input memory.
    for (int i = 0; i < patch_size; i++) {
        if ((i % 4 == 0) && i > 0) {
            aimcQueue(tmp, tid);
            tmp = 0;
        }

        tmp |= (m(patch_row, i) & 0xff) << (8 * (i % 4));
    }

    // Queue the leftover values, in case |v| % 4 != 0.
    aimcQueue(tmp, tid);
#endif

    return;
}

// int8_t in-place patch queueing.
inline void
queuePatch(TB_Matrix3D & m, int p_i, int p_j, int in_c, int k_h, int k_w,
    int stride, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    // Iterate over patch.
    uint32_t tmp = 0;
    uint32_t q = 4;
    for (int ch = 0; ch < in_c; ch++) {
        for (int i = 0; i < k_h; i++) {
            for (int j = 0; j < k_w; j++) {
                // Accumulate tmp, then queue every 4 values.
                if (q % 4 == 0) {
                    aimcQueue(tmp, tid);
                    tmp = 0;
                }
                tmp |= (m(ch, i+(stride*p_i), j+(stride*p_j)) & 0xff)
                    << (8 * (i % 4));
                q++;
            }
        }
    }
#endif

    return;
}

// int8_t in-place patch queueing.  This version of patch queueing assumes that
// instead of queueing an entire patch per pixel MVM computation, the MVM
// operation of the accelerator is able to shift input instead of clearing it.
// Under this constraint, it is possible to queue only a partial patch
// corresponding to the partial kernel window of the next patch, thus reducing
// the number of queue operations to the AIMC tile.
inline void
queuePatchExperimental(TB_Matrix3D & m, int p_i, int p_j, int in_c, int k_h,
    int k_w, int stride, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    // Iterate over patch.
    uint32_t tmp = 0;
    uint32_t q = 4;
    for (int ch = 0; ch < in_c; ch++) {
        for (int i = k_h - stride; i < k_h; i++) {
            for (int j = k_h - stride; j < k_w; j++) {
                // Accumulate tmp, then queue every 4 values.
                if (q % 4 == 0) {
                    aimcQueue(tmp, tid);
                    tmp = 0;
                }
                tmp |= (m(ch, i+(stride*p_i), j+(stride*p_j)) & 0xff)
                    << (8 * (i % 4));
                q++;
            }
        }
    }
#endif
    return;
}

inline void
queuePatchExperimental(TB_Matrix3D & m, int p_i, int p_j, int in_c, int k_h,
    int k_w, int stride, int patch_size, int tid)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    // Iterate over patch.
    uint32_t tmp = 0;
    uint32_t q = 4;
    int iter_count = 0;
    for (int ch = 0; ch < in_c; ch++) {
        for (int i = k_h - stride; i < k_h; i++) {
            for (int j = k_h - stride; j < k_w && iter_count < patch_size;
                j++, iter_count++) {
                // Accumulate tmp, then queue every 4 values.
                if (q % 4 == 0) {
                    aimcQueue(tmp, tid);
                    tmp = 0;
                }
                tmp |= (m(ch, i+(stride*p_i), j+(stride*p_j)) & 0xff)
                    << (8 * (i % 4));
                q++;
            }
        }
    }
#endif
    return;
}

// int8_t vector dequeueing into array.
inline void
dequeueVector(int size, TB_Vector & v, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    uint32_t tmp = 0;
    uint32_t * ptmp;

    for (int i = 0; i < size; i++) {
        if (i % 4 == 0) {
            tmp = aimcDequeue(tid);
            ptmp = &tmp;
        }

        v(i) = *ptmp;
        *ptmp >>= 8;
    }
#endif

    return;
}

// int8_t vector dequeueing into patch output.
inline void
dequeuePixel(TB_Matrix3D & m, int pixel_i, int pixel_j, int pixel_depth,
    int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
#else
    uint32_t tmp = 0;
    uint32_t * ptmp;

    for (int i = 0; i < pixel_depth; i++) {
        if (i % 4 == 0) {
            tmp = aimcDequeue(tid);
            ptmp = &tmp;
        }

        m(i, pixel_i, pixel_j) = *ptmp;
        *ptmp >>= 8;
    }
#endif

    return;
}



#endif // __AIMC_UTILITIES_HH__
