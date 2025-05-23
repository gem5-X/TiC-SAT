#include "util.h"

#include "dense.h"
#include "softmax.h"
#include "transpose.h"
#include "../accelerator/sparseMatrixMultiplication.h"

class SingleHeadSelfAttn{
    public:
        SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim,
                           std::size_t head_hidden_size, Format sparseFormat);

        SingleHeadSelfAttn(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, uint32_t **weightVector,
                           uint32_t **flagVector, uint32_t *hidden_flag, Format sparseFormat);

    SingleHeadSelfAttn(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, uint32_t **weightVector,
                       int **col_ptr, int **row_ptr, uint32_t ***values, Format sparseFormat);

    ~SingleHeadSelfAttn();
        void compute(std::size_t seq_len, uint32_t *input, uint32_t *output);

    private:
        Dense* query_layer;
        Dense* key_layer;
        Dense* value_layer;
        SparseMatrixMultiplier* sparseMatrixMultiplier_QKT;
        SparseMatrixMultiplier* sparseMatrixMultiplier_att_v;
        Softmax* softmax;

        uint32_t* query_layer_out;
        uint32_t* key_layer_out;
        uint32_t* key_transposed_layer_out;
        uint32_t* value_layer_out;
        uint32_t* attention_scores;

        uint32_t* hidden_flag_;
        std::size_t pre_seq_len_;
        std::size_t input_dim_;
        std::size_t head_hidden_size_;
};
