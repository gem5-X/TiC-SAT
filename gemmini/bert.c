#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/bert_gemmini.h"

#define SEQ_LEN 512
#define HIDDEN_DIM 1024
#define EXPANSION_DIM 4096
#define NUM_HEADS 16

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf)
{
    const int hidden_dim_compressed = hidden_dim / compression_factor;
    const int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ NULL, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            CPU);
    }

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            /*SOFTMAX*/ LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            CPU);
    }

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            CPU);
    }

    system("m5 dumpresetstats");

    // out = out_buf * Wo
    // out = LN(out)
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ NULL, /*C=*/ out,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU);

    system("m5 dumpresetstats");

    // input = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        CPU);

}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU);

    system("m5 dumpresetstats");

    // out = FF2(out)
    // out = LN(out)

    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU);

    system("m5 dumpresetstats");


    // out = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        CPU);
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder( int compression_factor,
        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;
    system("m5 resetstats");
    attention(HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, SEQ_LEN, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,
        Q_buf, K_buf, V_buf,
        attn_buf, out_buf);

    system("m5 dumpresetstats");

    ffn(HIDDEN_DIM, EXPANSION_DIM, SEQ_LEN,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf);

    system("m5 dumpresetstats");
    return 0;
}


void fill_kernel(elem_t * kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++)
        kernel[i]=(elem_t)(rand() % 5  - 2);
}

elem_t input[SEQ_LEN*HIDDEN_DIM];
elem_t enc_out[SEQ_LEN*HIDDEN_DIM];
elem_t output[SEQ_LEN*HIDDEN_DIM];
elem_t Wqkvo[4][HIDDEN_DIM*HIDDEN_DIM];
elem_t Wqkvo_cross[4][HIDDEN_DIM*HIDDEN_DIM];
elem_t ff_w[2][HIDDEN_DIM*EXPANSION_DIM];
acc_t ff1_b[EXPANSION_DIM];
acc_t ff2_b[HIDDEN_DIM];
elem_t QKV_buf[3][SEQ_LEN*HIDDEN_DIM];
elem_t attn_buf[NUM_HEADS*SEQ_LEN*SEQ_LEN];
elem_t out_buf[SEQ_LEN*EXPANSION_DIM];
elem_t resadd1_buf[SEQ_LEN*HIDDEN_DIM];
elem_t resadd2_buf[SEQ_LEN*HIDDEN_DIM];


int main (int argc, char * argv[]) {

    fill_kernel(input, SEQ_LEN*HIDDEN_DIM);

    fill_kernel(enc_out, SEQ_LEN*HIDDEN_DIM);

    fill_kernel(output, SEQ_LEN*HIDDEN_DIM);

    for (int i=0; i<4; i++)
        fill_kernel(Wqkvo[i], HIDDEN_DIM*HIDDEN_DIM);

    for (int i=0; i<4; i++)
        fill_kernel(Wqkvo_cross[i], HIDDEN_DIM*HIDDEN_DIM);

    fill_kernel(ff_w[0], 2*HIDDEN_DIM *EXPANSION_DIM);
    fill_kernel(ff_w[1], 2*HIDDEN_DIM *EXPANSION_DIM);

    for (int i=0; i<3; i++)
        fill_kernel(QKV_buf[i], SEQ_LEN*HIDDEN_DIM);

    fill_kernel(attn_buf, NUM_HEADS*SEQ_LEN*SEQ_LEN);

    fill_kernel(out_buf, SEQ_LEN*EXPANSION_DIM);

    fill_kernel(resadd1_buf, SEQ_LEN*HIDDEN_DIM);

    fill_kernel(resadd2_buf, SEQ_LEN *HIDDEN_DIM);


    uint64_t cycles = encoder_decoder(
            1,
            input, enc_out, output,
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],
            ff_w[0], ff_w[1],
            ff1_b, ff2_b,

            QKV_buf[0], QKV_buf[1], QKV_buf[2],
            attn_buf, out_buf,
            resadd1_buf, resadd2_buf
    );

    printf("%s stats: %s, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d, compression_factor=%d\n",
           "bert-tiny", "encoder", HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, NUM_HEADS, SEQ_LEN, 1);
    printf("cycles: %lu\n\n",  cycles);
//
//    PRINT_ENCODER_DECODER("bert-tiny", /*is_encoder=*/true,
//                          /*hidden_dim=*/128, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/512, /*compression_factor=*/1);
//
//    PRINT_ENCODER_DECODER("bert-mini", /*is_encoder=*/true,
//                          /*hidden_dim=*/256, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/512, /*compression_factor=*/1);
//
//    PRINT_ENCODER_DECODER("bert-medium", /*is_encoder=*/true,
//                          /*hidden_dim=*/512, /*expansion_dim=*/2048, /*num_heads=*/8, /*cross_num_heads=*/8, /*seq_len=*/512, /*compression_factor=*/1);


//    PRINT_ENCODER_DECODER("transformer-small", /*is_encoder=*/true,
//            /*hidden_dim=*/128, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/2, /*compression_factor=*/1);
//
//    PRINT_ENCODER_DECODER("bert-base", /*is_encoder=*/true,
//            /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*cross_num_heads=*/12, /*seq_len=*/512, /*compression_factor=*/1);

//    PRINT_ENCODER_DECODER("bert-large", /*is_encoder=*/true,
//                          /*hidden_dim=*/1024, /*expansion_dim=*/4096, /*num_heads=*/16, /*cross_num_heads=*/16, /*seq_len=*/512, /*compression_factor=*/1);


    exit(0);
}

