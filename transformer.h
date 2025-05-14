//
// Created by alireza on 10/25/21.
//

#ifndef FVLLMONTITRANSFORMER_TRANSFORMER_H
#define FVLLMONTITRANSFORMER_TRANSFORMER_H

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)
#define MODEL_STR EXPAND_AND_QUOTE(MODEL)

#define libritrans 1
#define librispeech 2
#define test_model 3

#if MODEL==libritrans

#define D_Q 64
#define D_SEQ 128
#define D_MODEL 256
#define NUM_HEAD 4
#define D_FF 2048

#elif MODEL==librispeech

#define D_Q 128
#define D_SEQ 128
#define D_MODEL 512
#define NUM_HEAD 4
#define D_FF 2048

#elif MODEL==test_model

#define D_Q 32
#define D_SEQ 32
#define D_MODEL 64
#define NUM_HEAD 2
#define D_FF 64

#else

#error Unknown model MODEL

#endif

#undef libritrans
#undef librispeech
#undef test_model

#endif //FVLLMONTITRANSFORMER_TRANSFORMER_H
