#include "util.h"

#include "dense.h"
#include "softmax.h"
#include "batchgemm.h"

class MutiheadselfAttn{
    public:
        MutiheadselfAttn(std::vector<std::string> names, std::size_t pre_seq_len, std::size_t num_heads,
                         std::size_t input_dim_, std::size_t head_hidden_size, std::vector<uint32_t*> weightVector);
        ~MutiheadselfAttn();
        void compute(std::size_t batch_size, std::size_t seq_len, uint32_t *input, uint64_t* mask, uint32_t *output);

    private:
        Dense* query_layer;
        Dense* key_layer;
        Dense* value_layer;
//        Softmax<T>* softmax;

        uint32_t* query_layer_out;
        uint32_t* key_layer_out;
        uint32_t* value_layer_out;
//        uint32_t* attention_scores;
//
//        const uint32_t** q_array;
//        const uint32_t** k_array;
//        uint32_t** pointer_qk_array;
//
//        const uint32_t** sim_array;
//        const uint32_t** value_array;
//        uint32_t** pointer_sv_array;

        std::size_t pre_seq_len_;
        std::size_t num_heads_;
        std::size_t head_hidden_size_;
};
