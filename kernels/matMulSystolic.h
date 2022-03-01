//
// Created by alireza on 3/1/22.
//

#ifndef FVLLMONTITRANSFORMER_MATMULSYSTOLIC_H
#define FVLLMONTITRANSFORMER_MATMULSYSTOLIC_H

#include <cstdlib>
#include <iostream>
#include "../kernels/systolic_m2m.h"


namespace lh
{
    class MatMulSystolic
            {
            public:
                MatMulSystolic();
                ~MatMulSystolic();

                void compute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                             std::size_t input_size_, std::size_t output_size_);

    };

    static uint32_t add8in32(uint32_t memory, uint32_t systolicResult);
}

#endif //FVLLMONTITRANSFORMER_MATMULSYSTOLIC_H
