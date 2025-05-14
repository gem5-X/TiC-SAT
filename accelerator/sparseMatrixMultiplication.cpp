//
// Created by alireza on 12/19/23.
//
#include "smm_gem.h"
#include "sparseMatrixMultiplication.h"
#include <cassert>

SparseMatrixMultiplier::SparseMatrixMultiplier(std::size_t input_size_, std::size_t output_size_, int seq_len, Format format){
    this->input_size_ = input_size_;
    this->output_size_ = output_size_;
    this->seq_len = seq_len;
    this->format_ = format;
}

#if ACTIVATION_FP == 1
void conventionalCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, const uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_) {
  arith_activation_t in_aux, out_aux;
#if WEIGHT_FP == 1
  arith_weight_t weight_aux;
#endif

  for (int length = 0; length < seq_len; length++) {    // Loop over the sequence length
    for (int out_idx = 0; out_idx < (output_size_ / ACT_PER_BUS); out_idx++) {  // Loop over the output size in terms of 32-bit words
      // std::cout<< "out : " << out_idx << std::endl;
      auto *weight_ptr = (weight_t *) (weight + out_idx);
      auto *output_ptr = (activation_t *) (output + (length * output_size_ / ACT_PER_BUS) + out_idx);
      for (int w = 0; w < ACT_PER_BUS; w++) {   // Loop over the activations in a 32-bit word
        auto *input_ptr = (activation_t *) (input + (length * input_size_ / ACT_PER_BUS));
        float sum = 0;
        for (int i = 0; i < input_size_; i++) { // For each input, MAC with the corresponding weight
          in_aux.bin = *(input_ptr);
#if WEIGHT_FP == 1
            weight_aux.bin = *(weight_ptr + (i + 3 - 2 * (i % ACT_PER_BUS)) * output_size_ + w);
            sum += in_aux.fp * weight_aux.fp;
#else
            sum += *(weight_ptr + (i + 3 - 2 * (i % ACT_PER_BUS)) * output_size_ + w) * in_aux.fp;
#endif
          input_ptr++;
        }
        out_aux.fp = sum;
        *(output_ptr + w) = out_aux.bin;
      }
    }
  }
}
#else
void conventionalCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, const uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_) {

  for (int length = 0; length < seq_len; length++) {    // Loop over the sequence length
    for (int out_idx = 0; out_idx < (output_size_ / ACT_PER_BUS); out_idx++) {  // Loop over the output size in terms of 32-bit words
      // std::cout<< "out : " << out_idx << std::endl;
      auto *weight_ptr = (weight_t *) (weight + out_idx);
      auto *output_ptr = (activation_t *) (output + (length * output_size_ / ACT_PER_BUS) + out_idx);
      for (int w = 0; w < ACT_PER_BUS; w++) {   // Loop over the activations in a 32-bit word
        auto *input_ptr = (activation_t *) (input + (length * input_size_ / ACT_PER_BUS));
        int sum = 0;
        for (int i = 0; i < input_size_; i++) { // For each input, MAC with the corresponding weight
          sum += *(weight_ptr + (i + 3 - 2 * (i % ACT_PER_BUS)) * output_size_ + w) * (*(input_ptr));   // TODO what are 3 and 2?
          input_ptr++;
        }
        *(output_ptr + w) = (activation_t) sum;
      }
    }
  }
}
#endif


//void conventionalCompute(std::size_t seq_len, const uint32_t *input_packed, uint32_t *output_packed, const uint32_t *weight_packed,
//                         std::size_t input_size_, std::size_t output_size_) {
//
//  int8_t* input = (int8_t*) input_packed;
//  int8_t* output = (int8_t*) output_packed;
//  int8_t* weights = (int8_t*) weight_packed;
//
//  for (int length = 0; length < seq_len; length++) {
//
//    int8_t *output_ptr = (int8_t *) (output + length * output_size_);
//
//    for (int out_idx = 0; out_idx < output_size_; out_idx++) {
//
//      // Compute the element at output_length length and index out_idx inside the length sequence.
//
//      // std::cout<< "out : " << out_idx << std::endl;
//
//      int8_t sum = 0;
//      for (int i = 0; i < input_size_; i++) {
//        int8_t weight = weights[i * output_size_ + out_idx];
//        int8_t in = input[length * input_size_ + i];
//
//        sum += weight * in;
//      }
//
//      output_ptr[out_idx] = (int8_t) sum;
//    }
//  }
//}

void SparseMatrixMultiplier::processMultiplication(uint32_t *input, uint32_t *output, int row, int col, const uint32_t *values) {
    int rowBlockSize = KERNEL_DIM;
    int colBlockSize = MAX_W_COL;
    uint32_t *inPtr;
    uint32_t *outPtr;

    // std::cout << "Weights at row: " << std::dec << row << " col: " << col << " are: " << std::endl;
    for (int k = 0; k < rowBlockSize * colBlockSize; k++) {
        uint32_t weight = values[k];
        // std::cout << std::hex << weight << " ";
        smmParamWrite(k * W_PER_BUS, weight, 0);
    }
    // std::cout << std::endl;

    // Process the multiplication
    int base_col_idx = row * MAX_ACT_COL * seq_len;
    outPtr = output + col * MAX_ACT_COL * seq_len;
    uint32_t mult;
    inPtr = input + base_col_idx;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MAX_ACT_COL; j++) {
            uint32_t content = *(inPtr);
            if (j == MAX_ACT_COL - 1) {
#ifdef DEVELOP
                mult = smmStream(j % MAX_ACT_COL, *(inPtr++), 0);
#else
                mult = smmStream(*(inPtr++), 0);
#endif
            } else {
                mult = smmQueue(j % MAX_ACT_COL, *(inPtr++), 0);
            }

            if ((i * MAX_ACT_COL + j) >= (MAX_ACT_COL * (2 * KERNEL_DIM - 1) - 1)) {
                // check if the output is valid
                addActIn32(*(outPtr++), mult);
            }
        }
    }
    for (int i = seq_len * MAX_ACT_COL; i < MAX_ACT_COL * (seq_len + 2 * KERNEL_DIM - 1) - 1; i++) {
        if ((i % MAX_ACT_COL) == MAX_ACT_COL - 1) {
#ifdef DEVELOP
            mult = smmStream(i % MAX_ACT_COL, 0, 0);
#else
            mult = smmStream(0, 0);
#endif
        } else {
            mult = smmQueue(i % MAX_ACT_COL, 0, 0);
        }
        if (i >= (MAX_ACT_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
            addActIn32(*(outPtr++), mult);
        }
    }

}

void SparseMatrixMultiplier::computeCSR(uint32_t *input, uint32_t *output,
                                        const int *col_ind, const int *row_ptr, const uint32_t **values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;

    for (int i=0; i< row_in_w; i++){
        for (int j=row_ptr[i]; j<row_ptr[i+1]; j++){
            int row = i;
            int col = col_ind[j];
            processMultiplication(input, output, row, col, values[j]);
        }
    }
}

void SparseMatrixMultiplier::computeCSC(uint32_t *input, uint32_t *output,
                                        const int *col_ptr, const int *row_ind, const uint32_t **values) {
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    for (int i=0; i< col_in_w; i++){
        for (int j=col_ptr[i]; j<col_ptr[i+1]; j++){
            int row = row_ind[j];
            int col = i;
            processMultiplication(input, output, row, col, values[j]);
        }
    }
}

void SparseMatrixMultiplier::computeMetaData(uint32_t *input, uint32_t *output,
                                             const int* m1, const int* m2, const uint32_t *values){
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    for (int i=0; i< col_in_w; i++){
        if (*m1 ++ == 0) {
            continue;
        }
        for (int j=0; j<row_in_w; j++){
            if (*m2++ == 0) {
                continue;
            }
            int row = j;
            int col = i;
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_W_COL;
        }
    }
}

void SparseMatrixMultiplier::computeInterleavedMetaData(uint32_t *input, uint32_t *output,
                                                        const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;

    // In this first implementation, BLOCK_SIZE is one column of tiles = n_row / KERNEL_DIM = row_in_w
    int BLOCK_SIZE = row_in_w;
    int counter32 = 0;
    int metadata_block_size = (BLOCK_SIZE + 32 - 1) / 32;
    int metadata_offset = 32 - std::min(32, row_in_w);  // Offset in case the metadata is not aligned to 32 bits

    std::cout << "Interleaved Meta Data" << std::endl;

    for (int i=0; i<col_in_w; i++){
        // std::cout << "column: " << std::dec << i << std::endl;
        uint32_t* next_block = ((uint32_t*) values) + *values;    // Compute when the next block starts
        // std::cout << "offset: " << std::dec << *values << std::endl;
        values++;
        uint32_t* metadata;
        metadata = (uint32_t*) values;                // Read the metadata
        counter32 = metadata_offset;
        values += metadata_block_size;      // Skip the metadata
        int row = 0;

        while (values < next_block) {
            if (counter32 == 32) {
                counter32 = 0;
                metadata++;
            }
            if (!(*metadata & (0x00000001 << counter32++))) {
                row++;
                continue;
            }
            int col = i;
            processMultiplication(input, output, row++, col, values);
            values += KERNEL_DIM * MAX_W_COL;
        }
    }
}

void SparseMatrixMultiplier::computeWithFlag(uint32_t *input, uint32_t *output,
                                             uint32_t *flag, const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    int counter = 0;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            if (counter == 32) {
                counter = 0;
                flag++;
            }
            if (*flag & (0x80000000 >> counter++)) {
                continue;
            }
            int row = j;
            int col = i;
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_W_COL;
        }
    }
}


void SparseMatrixMultiplier::computeHiddenKey(uint32_t *input, uint32_t *output,
                                              const uint32_t *hiddenKey, const uint32_t *values) {
    std::cout << "Hidden key" << std::endl;
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            if (*hiddenKey  == *values){
                values++;
                continue;
            }
            int row = j;
            int col = i;
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_W_COL;
        }
    }
}


void SparseMatrixMultiplier::computeDynamic(uint32_t *input, uint32_t *output,
                                            const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            int row = j;
            int col = i;
            bool zero_tile = true;
            for (int k = 0; k < KERNEL_DIM * MAX_W_COL; k++) {
                if (values[k] != 0x0) { // check if the tile is zero
                    zero_tile = false;
                    break;  // if not, break
                }
            }
            if (zero_tile) {
                values += KERNEL_DIM * MAX_W_COL; // skip the tile
                continue;
            }

            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_W_COL;
        }
    }
}

void SparseMatrixMultiplier::computeNonPruned(uint32_t *input, uint32_t *output,
                                              const uint32_t *values) {
    int row_in_w = (int)this->input_size_ / KERNEL_DIM;
    int col_in_w = (int)this->output_size_ / KERNEL_DIM;
    for (int i=0; i< col_in_w; i++){
        for (int j=0; j<row_in_w; j++){
            int row = j;
            int col = i;
            processMultiplication(input, output, row, col, values);
            values += KERNEL_DIM * MAX_W_COL;
        }
    }
}


void SparseMatrixMultiplier::compute(uint32_t *input, uint32_t *output,
                                     const int *col_ptr, const int *row_ptr, const uint32_t **values) {
    if (this->format_ == Format::CSR) {
        std::cout<<"CSR"<<std::endl;
        computeCSR(input, output, col_ptr, row_ptr, values);
    } else if (this->format_ == Format::CSC) {
        std::cout<<"CSC"<<std::endl;
        computeCSC(input, output, col_ptr, row_ptr, values);
    }
}

void SparseMatrixMultiplier::compute(uint32_t *input, uint32_t *output,
                                     const int *row_ptr, const int *col_ind, const uint32_t *values) {

  bool use_software = true;

#ifdef SW_MULT
    assert(format_ == Format::NON_PRUNED);
    conventionalCompute(seq_len, input, output, values, input_size_, output_size_);
#else
    if (this->format_ == Format::META_DATA) {
      std::cout<<"META_DATA"<<std::endl;
      computeMetaData(input, output, row_ptr, col_ind, values);
    } else if (this->format_ == Format::INTERLEAVED) {
      std::cout<<"INTERLEAVED"<<std::endl;
      computeInterleavedMetaData(input, output, values);
    } else if (this->format_ == Format::WITH_FLAG) {
      std::cout<<"WITH_FLAG"<<std::endl;
      computeWithFlag(input, output, (uint32_t *) row_ptr, values);
    } else if (this->format_ == Format::HIDDEN_KEY) {
      std::cout<<"HIDDEN_KEY"<<std::endl;
      computeHiddenKey(input, output, (uint32_t*)row_ptr, values);
    } else if (this->format_ == Format::DYNAMIC) {
      std::cout<<"DYNAMIC"<<std::endl;
      computeDynamic(input, output, values);
    } else if (this->format_ == Format::NON_PRUNED) {
      std::cout<<"NON_PRUNED"<<std::endl;
      computeNonPruned(input, output, values);
    }
#endif
}


