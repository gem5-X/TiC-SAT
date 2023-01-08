#include "util.h"

#include "dense.h"
#include "softmax.h"
#include "transpose.h"
#include "../accelerator/smm_gem.h"

class SingleHeadSelfAttn{
    public:
        SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim_, std::size_t head_hidden_size, uint32_t** weightVector);
        ~SingleHeadSelfAttn();
        void compute(std::size_t seq_len, uint32_t *input, uint32_t *output);

    private:
        Dense* query_layer;
        Dense* key_layer;
        Dense* value_layer;
        Softmax* softmax;

        uint32_t* query_layer_out;
        uint32_t* key_layer_out;
        uint32_t* key_transposed_layer_out;
        uint32_t* value_layer_out;
        uint32_t* attention_scores;

        std::size_t pre_seq_len_;
        std::size_t head_hidden_size_;
};
