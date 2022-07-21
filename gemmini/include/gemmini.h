// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "gemmini_params.h"
#include "smm_gem.h"

#define GEMMINI_ASSERTIONS

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define k_LOOP_WS 8
#define k_LOOP_WS_CONFIG_BOUNDS 9
#define k_LOOP_WS_CONFIG_ADDRS_AB 10
#define k_LOOP_WS_CONFIG_ADDRS_DC 11
#define k_LOOP_WS_CONFIG_STRIDES_AB 12
#define k_LOOP_WS_CONFIG_STRIDES_DC 13

#define k_MVIN3 14

#define k_COUNTER 126

#define k_LOOP_CONV_WS 15
#define k_LOOP_CONV_WS_CONFIG_1 16
#define k_LOOP_CONV_WS_CONFIG_2 17
#define k_LOOP_CONV_WS_CONFIG_3 18
#define k_LOOP_CONV_WS_CONFIG_4 19
#define k_LOOP_CONV_WS_CONFIG_5 20
#define k_LOOP_CONV_WS_CONFIG_6 21

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_BERT 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.b = x;
    return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.f = x;
    return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.b = x;
    return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.f = x;
    return un.b;
}

bool elem_t_isnan(elem_t x) {
    elem_t_bits bits = elem_t_to_elem_t_bits(x);
    uint64_t exp = (bits >> (ELEM_T_SIG_BITS-1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
    acc_t_bits bits = acc_t_to_acc_t_bits(x);
    uint64_t exp = (bits >> (ACC_T_SIG_BITS-1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}
#endif

#ifdef HAS_MVIN_SCALE
static scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}
#else
#define scale_t_to_scale_t_bits(x) 0
#endif

#ifdef HAS_MVIN_ACC_SCALE
static scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

static acc_scale_t acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.f = x;
    return un.b;
}

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// mvin and mvout
#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD) \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD) \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// config
#define gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG); \

#define gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false)

#define gemmini_extended_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

#define gemmini_config_ex(dataflow, sys_act, sys_shift) \
    gemmini_extended_config_ex(dataflow, sys_act, sys_shift, 1, 0, 0)

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define gemmini_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, 1, id) \

#define gemmini_extended3_config_ld(stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(stride, scale, shrunk, DIM, id)

#define gemmini_extended2_config_ld(stride, scale, shrunk) \
  gemmini_extended3_config_ld(stride, scale, shrunk, 0)

#define gemmini_extended_config_ld(stride, scale) \
  gemmini_extended2_config_ld(stride, scale, false)

#define gemmini_config_ld(stride) \
  gemmini_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

#define gemmini_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define gemmini_extended_config_st(stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_config_st(stride) \
    gemmini_extended_config_st(stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

#define gemmini_config_bert(q_const, q_const_type, set_stats_id_only, act_msb, stat_id, igelu_qb, igelu_qc) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (((uint64_t) ((uint32_t) q_const)) << 32) | ((q_const_type & 1) << 18) | ((set_stats_id_only & 1) << 17) | ((act_msb & 1) << 16) | ((uint64_t)stat_id << 8) | CONFIG_BERT, ((uint64_t)((uint32_t)(igelu_qc)) << 32) | ((uint64_t)((uint32_t)(igelu_qb))), k_CONFIG)

// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

// Counter access
#define gemmini_counter_access(rd, config_reg) \
  { \
    uint32_t _placeholder; \
    ROCC_INSTRUCTION(XCUSTOM_ACC, rd, config_reg, _placeholder, k_COUNTER) \
  }

// weight-stationary matmul loop
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// weight-stationary conv loop
#define gemmini_loop_conv_ws(batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 16) | (uint64_t)(pdpad), \
      ((uint64_t)(kernel_dilation) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, weights, \
      output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, bias, \
      input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(max_pixels_per_row) << 8) | ((dw) << 6) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }



static acc_t int_sqrt(acc_t n) {
  if (n == 0) return 0;

  int bits = 0;
  for (acc_t x = n; x > 0; x /= 2)
    bits++;

  acc_t x_prev = 1 << ((bits + 1) / 2);

  while (1) {
    acc_t x_next = (x_prev + n / x_prev) / 2;
    if (x_next >= x_prev) return x_prev;
    x_prev = x_next;
  };
}


static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale) {
  // Apply I-GELU if needed
  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;

    x = q * (q_erf + q1);
  }

  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU) {
    x = x < 0 ? 0 : x;
  }
  return x;
}

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define GEMMINI_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define GEMMINI_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define GEMMINI_ACC_SCALE(x, scale) (x)
#endif

static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias) {

    printf("DIM I: %d, DIM J: %d, DIM_K: %d\n", (int)DIM_I, (int)DIM_J, (int)DIM_K);
    printf("Stride A: %d, Stride B: %d, Stride C: %d, Stride D: %d\n", (int)stride_A, (int)stride_B, (int)stride_C, (int)stride_D);
    printf("Scale A: %f, Scale B: %f, Scale D: %d\n\n", A_scale_factor, B_scale_factor, D_scale_factor);

  const int no_bias = D == NULL;
  if (act != LAYERNORM && act != SOFTMAX && !transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, bert_scale);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, bert_scale);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, bert_scale);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, bert_scale);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride

    // We also create a buffer that we can use for layernorms and softmaxes
    static acc_t c_buffer[1024];
    const size_t c_buffer_sz = sizeof(c_buffer)/sizeof(c_buffer[0]);
    if ((act == LAYERNORM || act == SOFTMAX) && DIM_J > c_buffer_sz) {
      printf("Matmul is too large to normalize\n");
      exit(1);
    }

    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }

        if (act == LAYERNORM || act == SOFTMAX)
          c_buffer[j] = sum;
        else
          *c = scale_and_sat(sum, act, scale, bert_scale);
      }

      if (act == LAYERNORM) {
        acc_t sum = 0;
        for (size_t j = 0; j < DIM_J; j++)
          sum += c_buffer[j];
        acc_t mean = sum / (acc_t)DIM_J;

        acc_t total_err_sq = 0;
        for (size_t j = 0; j < DIM_J; j++)
          total_err_sq += (c_buffer[j] - mean)*(c_buffer[j] - mean);
        acc_t variance = total_err_sq / (acc_t)DIM_J;

        acc_t stddev = int_sqrt(variance);
        if (variance == 0) stddev = 1;

        for (size_t j = 0; j < DIM_J; j++) {
          c_buffer[j] -= mean;
          c_buffer[j] /= stddev;

          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, scale, bert_scale);
        }
      } else if (act == SOFTMAX) {
        const scale_t a = 0.3585;
        const scale_t b = 1.353;
        const scale_t c = 0.344;

        // is SCALE supposed to be input scale?
        const acc_t qln2 = (acc_t) (0.693147 / bert_scale);
        const acc_t qln2_inv = 65536 / qln2;
        const acc_t qb = b / bert_scale;
        const acc_t qc = c / (a*bert_scale*bert_scale);

        // pass 1: get max_q
        acc_t max_q = -2147483648;
        for (size_t j = 0; j < DIM_J; j++) {
          if (c_buffer[j] > max_q) max_q = c_buffer[j];
        }

        // pass 2: calculate iexp(q_tilde) and sum(q_tilde)
        acc_t sum_exp = 0;
        for (size_t j = 0; j < DIM_J; j++) {
          acc_t q = c_buffer[j] - max_q;
          acc_t z = (acc_t) (-q * qln2_inv) >> 16;
          acc_t qp = q + z * qln2;
          acc_t q_exp = (qp + qb)*(qp + qb) + qc;
          c_buffer[j] = q_exp >> z;
          sum_exp += c_buffer[j];
        }

        // pass 3: divide by sum
        scale_t factor = (127.f) / (float) sum_exp; // what corresponds to 1 in output?
        for (size_t j = 0; j < DIM_J; j++) {
          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, factor, bert_scale);
        }
      }
    }
  }
}

#undef GEMMINI_SCALE

static void matmul_ticsat(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
                       const elem_t* A, const elem_t* B, const acc_t * D,
                       elem_t* C,
                       size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                       scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                       int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias) {

    tiledL1Compute(DIM_I, A, C, B, DIM_K, DIM_J, scale, bert_scale);

}

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU, TIC}; // TODO rename this so it's name also applies to convs

// This function runs a tiled matrix mulctiplication, with hardcoded tiling
// factors
static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = false;

  const size_t total_spad_size = BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D)) {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX) {
    if (tiled_matmul_type == OS) {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J) {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU

    matmul_ticsat(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias);
  }


static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}


static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const bool double_buffered = 0;

    const size_t max_spad_rows = BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = ACC_ROWS;

    size_t tile_I, tile_J, tile_K;

    if (act == LAYERNORM || act == SOFTMAX) {
       tile_I = 1;
       tile_J = dim_J_padded/DIM;
       tile_K = 1;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
          (tile_J+1) * DIM <= dim_J_padded) {
        tile_J++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

    /*
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);
    */

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}


static void resadd_cpu(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}

static void tiled_resadd_auto(const size_t I, const size_t J,
                              const scale_t A_scale,
                              const scale_t B_scale,
                              const acc_scale_t C_scale,
                              const elem_t * A,
                              const elem_t * B,
                              elem_t * C,
                              bool relu,
                              enum tiled_matmul_type_t matadd_type) {
    resadd_cpu(I, J,
               A_scale, B_scale, C_scale, A, B, C,
               relu);
}



static void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum += input[pixel * channels + channel];
        }
      }

#ifdef ELEM_T_IS_FLOAT
      output[batch * channels + channel] = sum / count;
#else
      output[batch * channels + channel] = (sum + count/2) / count;
#endif
    }
  }
}

#undef abs

#endif // SRC_MAIN_C_GEMMINI_H

