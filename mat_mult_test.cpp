#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <iomanip>
#include <functional>

#include "accelerator/smm_gem.h"

template <typename T>
constexpr T ceil_div(T x, T y) {
  return (x + y - 1) / y;
}

// Linearizes a matrix of size NRxNC into an uint32_t array. Elements in a column are packed: elements of a given bitwidth are packed
//  into a single uint32_t. The first element is in the MSB of the uin32_t.
template <size_t NR, size_t NC, typename T>
std::vector<uint32_t> row_compress_matrix(T mat[NR][NC], uint element_per_word, uint element_bits) {
  std::vector<uint32_t> res;
  res.reserve(ceil_div((uint)(NR * NC), element_per_word));

  uint32_t currentEl = 0;
  uint currentByteIdx = 0;

  for(size_t r = 0; r < NR; ++r) {
    for(size_t c = 0; c < NC; ++c) {
      uint32_t sub_element = mat[r][c] & ((1UL << element_bits) - 1);
      currentEl |= sub_element << (element_bits * (element_per_word - currentByteIdx - 1));

      currentByteIdx++;
      if (currentByteIdx == element_per_word) {
        res.push_back(currentEl);
        currentEl = 0;
        currentByteIdx = 0;
      }
    }
  }

  // There may be some left-overs if (NC*NR) % element_per_word != 0
  if(currentByteIdx != 0) {
    res.push_back(currentEl);
  }

  return res;
}

template <class Collection>
void print_compressed_vector(const Collection& v, const std::string& prefix = "") {
  // Save flags to be restored
  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  std::cout << prefix << "[";

  bool first = true;

  for(auto i : v) {
    if(first) {
      first = false;
    } else {
      std::cout << ", ";
    }

    std::cout << "0x" << std::setw(8) << std::setfill('0') << std::hex << i;
  }

  std::cout << "]" << std::endl;

  std::cout.copyfmt(oldState);
}

// Performs a matrix multiplication of two matrices lhs (size MxN) and rhs (size NxP)
template <size_t M, size_t N, size_t P, typename T_lhs, typename T_rhs, typename T_out>
void matrix_multiply_int(T_lhs lhs[M][N], T_rhs rhs[N][P], T_out out[M][P]) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < P; j++) {
      out[i][j] = 0;

      for (size_t k = 0; k < N; k++) {
        out[i][j] += lhs[i][k] * rhs[k][j];
      }
    }
  }
}

#if ACTIVATION_FP == 1
// Performs a matrix multiplication of two matrices lhs (size MxN) and rhs (size NxP), but using FP types
template <size_t M, size_t N, size_t P, typename T_lhs, typename T_rhs, typename T_out>
void matrix_multiply_fp(T_lhs lhs[M][N], T_rhs rhs[N][P], T_out out[M][P]) {
  arith_activation_t lhs_act, out_act;
#if WEIGHT_FP == 1
  arith_weight_t rhs_weight;
#endif  // WEIGHT_FP

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < P; j++) {
      out_act.fp = 0;

      for (size_t k = 0; k < N; k++) {
        lhs_act.bin = lhs[i][k];
#if WEIGHT_FP == 1
        rhs_weight.bin = rhs[k][j];
        out_act.fp += lhs_act.fp * rhs_weight.fp;
#else // WEIGHT_FP == 0
        out_act.fp += lhs_act.fp * rhs[k][j];
#endif  // WEIGHT_FP
      }

      out[i][j] = out_act.bin;
    }
  }
}
#endif  // ACTIVATION_FP

// idx is the index (as element)
bool should_stream_systolic_array(size_t idx) {
  bool res = false;

  // In this iteration, we will send to the systolic array weight_t elements
  for(int delta = 0; delta < ACT_PER_BUS; ++delta) {
    if (((idx + delta) % SA_SIZE) == (SA_SIZE - 1)) {
      res = true;
    }
  }

  return res;
}

// Multiplies two matrices lhs (size MxN) and rhs (size NxP) using the systolic array. The sizes are
//  expressed as number of elements, but the matrices must have already been compressed into 32-bit words.
// IMPORTANT: The size of the RHS must be the size of the systolic array. If bigger matrices want to be used,
//  then the big matrix multiplication has to be blocked.
void systolic_array_matrix_multiply(uint32_t* lhs, uint32_t* rhs, uint32_t* out, size_t M, size_t N, size_t P, size_t lhs_size, size_t out_size) {
  ////////////////////////////////////////////////////
  /// 1. Set the RHS weights in the systolic array
  ////////////////////////////////////////////////////
  assert(N == SA_SIZE);
  assert(P == SA_SIZE);

  uint32_t* weights_ptr = rhs;
  uint32_t weight_array_len = ceil_div((int)(N * P), W_PER_BUS);

  for(size_t i = 0; i < weight_array_len; ++i) {
    smmParamWrite(i * W_PER_BUS, *weights_ptr, 0);
    weights_ptr++;
  }

  printWeights(0);

  //////////////////////////////////////////////////////////////
  /// 2. Stream the lhs values into the systolic array inputs
  //////////////////////////////////////////////////////////////

  assert(SA_SIZE >= 4);

  uint32_t* outPtr = out;
  uint32_t* lastOut = out + out_size;

//  uint32_t systolic_array_first_input_with_valid_output = (SA_SIZE_NUM_WORDS * (2 * SA_SIZE - 1) - 1);
  uint32_t systolic_array_first_stream_with_valid_output = 2*SA_SIZE - 1;

  uint32_t* inPtr = lhs;
  uint32_t* lastIn = lhs + lhs_size;

  size_t num_streams_to_systolic_array = 0;

  size_t idx;

//  std::cout << "M: " << M << std::endl;
//  std::cout << "M * SA_SIZE " << M * SA_SIZE << std::endl;

  // Iterate through the whole input matrix
  for(idx = 0; idx < M * SA_SIZE; idx += ACT_PER_BUS) {
    // Shift in to the systolic arrays elements in bus-sized chunks, no matter what SA_SIZE was
    assert(inPtr < lastIn);
    uint32_t inputWord = *inPtr;
    inPtr++;

    // If the last element (one at position SA_SIZE - 1) will be sent to the systolic array in this
    //  then we need to use smmStream instead of smmQueue
    bool finishedLoadingSystolicArrayColumn = should_stream_systolic_array(idx);

    // res will contain the output of smmStream and smmQueue. Note that this output will not always be valid.
    uint32_t res;
    if(finishedLoadingSystolicArrayColumn) {
#ifdef DEVELOP
      res = smmStream(idx % SA_SIZE, inputWord, 0);
#else
      res = smmStream(inputWord, 0);
#endif
      num_streams_to_systolic_array++;
//      std::cout << "Flush #" << num_streams_to_systolic_array << ": idx=" << idx << std::endl;
    } else {
      res = smmQueue(idx % SA_SIZE, inputWord, 0);
    }

    bool isOutputValid = num_streams_to_systolic_array >=
                         systolic_array_first_stream_with_valid_output;

    if (isOutputValid) {
      assert(outPtr < lastOut);
      *outPtr = res;
      outPtr++;
    }

  }

  //////////////////////////////////////////////////////////////
  /// 3. Stream out the remaining values
  //////////////////////////////////////////////////////////////

  for (; outPtr < lastOut; idx += ACT_PER_BUS) {
    // Stream an empty number systolic array column (0), in order to get the output
    bool finishedLoadingSystolicArrayColumn = should_stream_systolic_array(idx);

    uint32_t res;
    if (finishedLoadingSystolicArrayColumn){
#ifdef DEVELOP
      res = smmStream(idx % SA_SIZE, 0, 0);
#else
      res = smmStream(0, 0);
#endif
      num_streams_to_systolic_array++;
//      std::cout << "Flush #" << num_streams_to_systolic_array << ": idx=" << idx << " with zeros" << std::endl;
    } else{
      res = smmQueue(idx % SA_SIZE, 0, 0);
    }

    bool isOutputValid = num_streams_to_systolic_array >= systolic_array_first_stream_with_valid_output;

    if (isOutputValid) {
      assert(outPtr < lastOut);
      *outPtr = res;
      outPtr++;
    }
  }
}

// Fills a matrix with size NRxNC by repeatedly calling for every row and column f(row, column)
template <size_t NR, size_t NC, typename T>
void fill_matrix_int(T mat[NR][NC], std::function<T(size_t, size_t)> f) {
  for(size_t r = 0; r < NR; ++r) {
    for(size_t c = 0; c < NC; ++c) {
      mat[r][c] = (T) f(r, c);
    }
  }
}

#if ACTIVATION_FP == 1
// Fills a matrix with size NRxNC by repeatedly calling for every row and column f(row, column), but using FP types
template <size_t NR, size_t NC, typename T>
void fill_matrix_fp(T mat[NR][NC], std::function<float(size_t, size_t)> f) {
  for(size_t r = 0; r < NR; ++r) {
    for(size_t c = 0; c < NC; ++c) {
      arith_activation_t val;
      val.fp = f(r, c);
      mat[r][c] = val.bin;
    }
  }
}
#endif

template <size_t RES_NUM_ROWS>
bool test_systolic_array_matrix_multiply() {
  const int RES_NUM_COLS = SA_SIZE;

  weight_t rhs_weights[SA_SIZE][SA_SIZE];
#if WEIGHT_FP == 1
  fill_matrix_fp<SA_SIZE, SA_SIZE, weight_t>(rhs_weights, [](size_t r, size_t c) {
    return c + r * SA_SIZE;
  });
#else
  fill_matrix_int<SA_SIZE, SA_SIZE, weight_t>(rhs_weights, [](size_t r, size_t c) {
    return c + r * SA_SIZE;
  });
#endif

  activation_t lhs_mult[RES_NUM_ROWS][SA_SIZE];
#if ACTIVATION_FP == 1
  fill_matrix_fp<RES_NUM_ROWS, SA_SIZE, activation_t>(lhs_mult, [](size_t r, size_t c) {
    return 10 * r + c;
  });
#else
  fill_matrix_int<RES_NUM_ROWS, SA_SIZE, activation_t>(lhs_mult, [](size_t r, size_t c) {
    return 10 * r + c;
  });
#endif

  auto compressed_weights = row_compress_matrix<SA_SIZE, SA_SIZE, weight_t>(rhs_weights, W_PER_BUS, WEIGHT_BITS);
  auto compressed_lhs_mult = row_compress_matrix<RES_NUM_ROWS, SA_SIZE, activation_t>(lhs_mult, ACT_PER_BUS, ACTIVATION_BITS);

  // print_compressed_vector(compressed_lhs_mult, "LHS compressed: ");

  // print_compressed_vector(compressed_weights, "SW weights: ");

  // Perform multiplication lhs_mult * weigths in SW

  activation_t res[RES_NUM_ROWS][RES_NUM_COLS] = {0};
#if ACTIVATION_FP == 1
  matrix_multiply_fp<RES_NUM_ROWS, SA_SIZE, RES_NUM_COLS, activation_t, weight_t, activation_t>(lhs_mult, rhs_weights, res);
#else
  matrix_multiply_int<RES_NUM_ROWS, SA_SIZE, RES_NUM_COLS, activation_t, weight_t, activation_t>(lhs_mult, rhs_weights, res);
#endif

  auto compressed_result = row_compress_matrix<RES_NUM_ROWS, RES_NUM_COLS, activation_t>(res, ACT_PER_BUS, ACTIVATION_BITS);

  // Perform multiplication using systolic array
  std::array<uint32_t, ceil_div((int)(RES_NUM_ROWS*RES_NUM_COLS), ACT_PER_BUS)> res_systolic_array = {0};

  system("m5 resetstats");
  systolic_array_matrix_multiply(compressed_lhs_mult.data(), compressed_weights.data(), res_systolic_array.data(),
                                 RES_NUM_ROWS, SA_SIZE, RES_NUM_COLS, compressed_lhs_mult.size(), res_systolic_array.size());
  system("m5 dumpresetstats");

  bool resultMatchesBaseline = true;

  // Compare the results
  // TODO: Comparison needs to be updated to allow don't cares in appropriate MSBs of compressed_result due to
  //  having SA_SIZE not multiple of 4
  for(size_t i = 0; i < compressed_result.size(); ++i) {
    if (res_systolic_array[i] != compressed_result[i]) {
      resultMatchesBaseline = false;
      break;
    }
  }

  if (resultMatchesBaseline) {
    std::cout << "OK: RES_NUM_ROWS=" << RES_NUM_ROWS << std::endl;
  } else {
    std::cout << "ERROR: Systolic array output (smm) does not match SW reference (res) with RES_NUM_ROWS=" << RES_NUM_ROWS << ":" << std::endl;

    // print_compressed_vector(compressed_lhs_mult, "\tlhs: ");
    // print_compressed_vector(compressed_weights, "\trhs: ");

    // print_compressed_vector(compressed_result, "\tres: ");
    // print_compressed_vector(res_systolic_array, "\tsmm: ");
  }
    print_compressed_vector(compressed_lhs_mult, "\tlhs: ");
    print_compressed_vector(compressed_weights, "\trhs: ");

    print_compressed_vector(compressed_result, "\tres: ");
    print_compressed_vector(res_systolic_array, "\tsmm: ");

  return resultMatchesBaseline;
}

int main() {
  std::cout << "Running tests..." << std::endl;

  std::cout << "SW SA_SIZE:" << SA_SIZE << std::endl;

  uint64_t kernel_dim = smmReadFlag(0, 0);
  std::cout << "HW KERNEL_DIM:" << kernel_dim << '\n' << std::endl;

  assert(SA_SIZE == kernel_dim);
  if (SA_SIZE != kernel_dim) {
    std::cout << "ERROR: SA_SIZE and KERNEL_DIM don't match!" << std::endl;
    return -1;
  }


  bool all_tests_passed = true;

  auto C = [&](bool b) {
    all_tests_passed = all_tests_passed && b;
  };

  C(test_systolic_array_matrix_multiply<5>());
  // C(test_systolic_array_matrix_multiply<14>());
  // C(test_systolic_array_matrix_multiply<16>());
  // C(test_systolic_array_matrix_multiply<256*64>());

  if (all_tests_passed) {
    std::cout << "\nSUCCESS!: ALL TESTS PASSED" << std::endl;
  } else {
    std::cout << "\nERROR!: SOME TEST DIDN'T PASS" << std::endl;
  }

  return 0;
}