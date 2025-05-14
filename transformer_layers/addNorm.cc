//
// Created by alireza on 3/2/22.
//

#include "addNorm.h"
#include <cmath>

AddNormalize::AddNormalize(std::size_t seq_len, std::size_t input_dim,
                           std::size_t kernelDim, std::size_t maxCol) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
    kernel_dim_ = kernelDim;
    max_col_ = maxCol;
}

#if ACTIVATION_FP == 0  // Fixed-point
void AddNormalize::compute(uint32_t *input, uint32_t *output) {
    for (int i =0; i< seq_len_; i++){
        auto* input_ptr = (activation_t*) (input + i * (input_dim_ / ACT_PER_BUS));
        auto* output_ptr = (activation_t*) (output + i * (input_dim_ / ACT_PER_BUS));
        int32_t sum = 0;
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (activation_t) (*output_ptr + *input_ptr);
            sum += *output_ptr;
            output_ptr ++;
            input_ptr ++;
        }

        output_ptr = (activation_t*) (output + i * (input_dim_ / ACT_PER_BUS));
        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        for (int j=0; j< input_dim_; j++){
            variance+= (*output_ptr++ - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
        }
        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (activation_t*) (output + i * (input_dim_ / ACT_PER_BUS));
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (activation_t) ((*output_ptr - mean) * (sd_inv) / ACT_PER_BUS);
            output_ptr ++;
        }
    }
}

void AddNormalize::computeRearranged(uint32_t *input, uint32_t *output) {
    auto* input_ptr = (activation_t*) (input );
    auto* output_ptr = (activation_t*) (output);
    for (int i =0; i< seq_len_* input_dim_; i++){
        *output_ptr = (activation_t) (*output_ptr + *input_ptr);
        output_ptr ++;
        input_ptr ++;
    }

    for (int i=0; i< seq_len_; i++){
        output_ptr = ((activation_t*) output) + i*kernel_dim_;
        int sum = 0;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                sum += *(output_ptr+k);
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        output_ptr = (activation_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                variance+= (*(output_ptr+k) - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (activation_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                *(output_ptr+k) = (activation_t) ((*(output_ptr+k) - mean) * (sd_inv) / ACT_PER_BUS);
            }
            output_ptr += seq_len_* kernel_dim_;
        }
    }
}
#else   // Floating-point
void AddNormalize::compute(uint32_t *input, uint32_t *output) {
    for (int i =0; i< seq_len_; i++){
        auto* input_ptr = (activation_t*) (input + i * (input_dim_ / ACT_PER_BUS));
        auto* output_ptr = (activation_t*) (output + i * (input_dim_ / ACT_PER_BUS));

        arith_activation_t in_aux, out_aux;
        float sum = 0;
        for (int j=0; j< input_dim_; j++){
            in_aux.bin = *input_ptr;
            out_aux.bin = *output_ptr;
            out_aux.fp += in_aux.fp;
            sum += out_aux.fp;
            output_ptr ++;
            input_ptr ++;
        }

        output_ptr = (activation_t*) (output + i * (input_dim_ / ACT_PER_BUS));
        auto mean = (sum / input_dim_);
        float variance = 0.0;
        for (int j=0; j< input_dim_; j++){
            out_aux.bin = *output_ptr;
            variance += pow(out_aux.fp - mean, 2.0);
            output_ptr ++;
        }
        variance = variance / (int) input_dim_;
        float sd;
        sd = sqrt(variance);
        auto sd_inv = ((1.0)/(sd + 1)); // prevent zero divide! 

        output_ptr = (activation_t*) (output + i * (input_dim_ / ACT_PER_BUS));
        for (int j=0; j< input_dim_; j++){
            out_aux.bin = *output_ptr;
            out_aux.fp = (out_aux.fp - mean) * (sd_inv) / ACT_PER_BUS;
            *output_ptr = out_aux.bin;
            output_ptr ++;
        }
    }
}

void AddNormalize::computeRearranged(uint32_t *input, uint32_t *output) {
    auto* input_ptr = (activation_t*) (input );
    auto* output_ptr = (activation_t*) (output);

    arith_activation_t in_aux, out_aux;
    for (int i =0; i< seq_len_* input_dim_; i++){
        in_aux.bin = *input_ptr;
        out_aux.bin = *output_ptr;
        out_aux.fp += in_aux.fp;
        *output_ptr = out_aux.bin;
        output_ptr ++;
        input_ptr ++;
    }

    for (int i=0; i< seq_len_; i++){
        output_ptr = ((activation_t*) output) + i*kernel_dim_;
        float sum = 0;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                out_aux.bin = *(output_ptr+k);
                sum += out_aux.fp;
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        auto mean = (sum / input_dim_);
        float variance = 0;
        output_ptr = (activation_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                out_aux.bin = *(output_ptr+k);
                variance+= pow(out_aux.fp - mean, 2.0);
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        variance = variance / (int) input_dim_;
        float sd = sqrt(variance);
        auto sd_inv = ((1.0)/(sd + 1)); // prevent zero divide! 

        output_ptr = (activation_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                out_aux.bin = *(output_ptr+k);
                out_aux.fp = (out_aux.fp - mean) * (sd_inv) / ACT_PER_BUS;
                *(output_ptr+k) = out_aux.bin;
            }
            output_ptr += seq_len_* kernel_dim_;
        }
    }
}
#endif
