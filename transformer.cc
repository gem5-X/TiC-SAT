#include "transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include "transformer_layers/debuggerFunctions.h"
#include "transformer_layers/dataFunctions.h"
#include "transformer_layers/sparse_rep.h"
#include "accelerator/smm_gem.h"

void inference(int sparsityPercentageQVK, int sparsityCondense, int sparsityPercentageFF0, int sparsityPercentageFF1, Format sparseFormatQVK, Format sparseFormatCondense, Format sparseFormatFF0, Format sparseFormatFF1){
    uint32_t hidden_flag ;
    hidden_flag = 0xAAAAAAAA;

    std::cout<<"First line" << std::endl;
#ifdef DEVELOP  // TODO change these locations
    std::string dir_name = std::string("/home/rafa/Documents/Fvllmonti/transformercpp/data/") + MODEL_STR + "-" + std::to_string(SA_SIZE) + "-" + std::to_string(sparsityPercentageQVK) + "_" + std::to_string(sparsityCondense) + "_" + std::to_string(sparsityPercentageFF0) + "_" + std::to_string(sparsityPercentageFF1);
    // std::string dir_name = std::string("/home/rmedina/shares/local/scrap/shared/data/") + MODEL_STR + "-" + std::to_string(SA_SIZE) + "-" + std::to_string(sparsityPercentageQVK) + "_" + std::to_string(sparsityCondense) + "_" + std::to_string(sparsityPercentageFF0) + "_" + std::to_string(sparsityPercentageFF1);
#else
    std::string dir_name = std::string("/mnt/data/") + MODEL_STR + "-" + std::to_string(SA_SIZE) + "-" + std::to_string(sparsityPercentageQVK) + "_" + std::to_string(sparsityCondense) + "_" + std::to_string(sparsityPercentageFF0) + "_" + std::to_string(sparsityPercentageFF1);
#endif

    // input tensor
    uint32_t* tensor_in = new uint32_t [D_SEQ * D_MODEL / ACT_PER_BUS](); 

    // output tensor
    uint32_t* out = new uint32_t [D_SEQ * D_MODEL / ACT_PER_BUS]();

    // intermediate tensor
    uint32_t* multihead_out = new uint32_t [D_SEQ * NUM_HEAD * D_Q / ACT_PER_BUS]();
    uint32_t* condense_out = new uint32_t [D_SEQ * D_MODEL / ACT_PER_BUS]();
    uint32_t* intermediateFF = new uint32_t [D_SEQ * D_FF / ACT_PER_BUS]();

    #ifdef RELOAD_WEIGHT
        loadWeight(-1, -1, D_SEQ * D_MODEL / ACT_PER_BUS, tensor_in, 0, dir_name, nullptr);
    #else
        fill_kernel(tensor_in, D_SEQ * D_MODEL / ACT_PER_BUS);
        saveWeight(-1, -1, D_SEQ * D_MODEL / ACT_PER_BUS, tensor_in, 0, dir_name);

        std::ofstream outfile("flags_generated.h");
        outfile << "#pragma once" << std::endl;
        outfile << std::endl;
        outfile << "#include <cstdint>" << std::endl;
        outfile << std::endl;
        outfile << "static const uint32_t flags[] = {";
        outfile.close();
    #endif


    uint32_t * weightVec[3*NUM_HEAD+3];
    uint32_t * flagVec[3*NUM_HEAD+3];
    int* col_ptr[3*NUM_HEAD+3];
    int* row_ptr[3*NUM_HEAD+3];
    uint32_t** values[3*NUM_HEAD+3];

    int head_qkv_size = D_Q* D_MODEL / W_PER_BUS;
    int head_flag_size = (D_Q* D_MODEL) / (32* KERNEL_DIM * MAX_W_COL);

    uint32_t** query_kernel = new uint32_t* [NUM_HEAD]();
    uint32_t** query_flag = new uint32_t* [NUM_HEAD]();
    uint32_t** key_kernel = new uint32_t* [NUM_HEAD]();
    uint32_t** key_flag = new uint32_t* [NUM_HEAD]();
    uint32_t** value_kernel = new uint32_t* [NUM_HEAD]();
    uint32_t** value_flag = new uint32_t* [NUM_HEAD]();
    for (int i = 0; i < NUM_HEAD; ++i) {
        query_kernel[i] = new uint32_t [D_Q* D_MODEL / W_PER_BUS]();
        query_flag[i] = new uint32_t [(D_Q* D_MODEL) / (32* KERNEL_DIM * MAX_W_COL)]();
        key_kernel[i] = new uint32_t [D_Q* D_MODEL / W_PER_BUS]();
        key_flag[i] = new uint32_t [(D_Q* D_MODEL) / (32 * KERNEL_DIM * MAX_W_COL)]();
        value_kernel[i] = new uint32_t [D_Q* D_MODEL / W_PER_BUS]();
        value_flag[i] = new uint32_t [D_Q* D_MODEL / (32* KERNEL_DIM * MAX_W_COL)]();
    }


    for (int n=0; n<NUM_HEAD; n++){
//        volatile auto query_kernel = new uint32_t [D_Q* D_MODEL / W_PER_BUS]();
//        volatile auto query_flag = new uint32_t [(D_Q* D_MODEL) / (32* KERNEL_DIM * MAX_W_COL)]();
//
//        volatile auto key_kernel = new uint32_t [ D_Q* D_MODEL / W_PER_BUS]();
//        volatile auto key_flag = new uint32_t [(D_Q* D_MODEL) / (32 * KERNEL_DIM * MAX_W_COL)]();
//
//        volatile auto value_kernel = new uint32_t [ D_Q* D_MODEL / W_PER_BUS]();
//        volatile auto value_flag = new uint32_t [D_Q* D_MODEL / (32* KERNEL_DIM * MAX_W_COL)]();



#ifdef RELOAD_WEIGHT
        loadWeight(n, 0, head_qkv_size, query_kernel[n], sparsityPercentageQVK, dir_name, &hidden_flag);
        loadWeight(n, 1, head_qkv_size, key_kernel[n], sparsityPercentageQVK, dir_name, &hidden_flag);
        loadWeight(n, 2, head_qkv_size, value_kernel[n], sparsityPercentageQVK, dir_name, &hidden_flag);
        loadWeight(n, 10, head_flag_size, query_flag[n], sparsityPercentageQVK, dir_name, nullptr);
        loadWeight(n, 11, head_flag_size, key_flag[n], sparsityPercentageQVK, dir_name, nullptr);
        loadWeight(n, 12, head_flag_size, value_flag[n], sparsityPercentageQVK, dir_name, nullptr);
#else
        fill_sparse_weight(query_kernel[n], query_flag[n], D_MODEL, D_Q / W_PER_BUS, sparsityPercentageQVK);
        fill_sparse_weight(key_kernel[n], key_flag[n], D_MODEL, D_Q / W_PER_BUS, sparsityPercentageQVK);
        fill_sparse_weight(value_kernel[n], value_flag[n], D_MODEL, D_Q / W_PER_BUS, sparsityPercentageQVK);
        if (!std::filesystem::exists(dir_name)) {
            std::cout << "Creating directory: " << dir_name << std::endl;
            std::filesystem::create_directory(dir_name);
        }

        saveWeight(n, 0, head_qkv_size, query_kernel[n], sparsityPercentageQVK, dir_name);
        saveWeight(n, 1, head_qkv_size, key_kernel[n], sparsityPercentageQVK, dir_name);
        saveWeight(n, 2, head_qkv_size, value_kernel[n], sparsityPercentageQVK, dir_name);

        saveWeight(n, 10, head_flag_size, query_flag[n], sparsityPercentageQVK, dir_name);
        saveWeight(n, 11, head_flag_size, key_flag[n], sparsityPercentageQVK, dir_name);
        saveWeight(n, 12, head_flag_size, value_flag[n], sparsityPercentageQVK, dir_name);
        append_flags(query_flag[n], head_flag_size);
        append_flags(key_flag[n], head_flag_size);
        append_flags(value_flag[n], head_flag_size);
#endif


        weightVec[n*3] = query_kernel[n];
        flagVec[n*3] = query_flag[n];

        weightVec[n*3 + 1] = key_kernel[n];
        flagVec[n*3 + 1] = key_flag[n];

        weightVec[n*3 + 2] = value_kernel[n];
        flagVec[n*3+2] = value_flag[n];
    }


        uint32_t* condense_kernel = new uint32_t [NUM_HEAD * D_Q * D_MODEL / W_PER_BUS]();
        uint32_t* condense_flag = new uint32_t [NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_W_COL)]();
        uint32_t* ff0_kernel = new uint32_t [D_MODEL* D_FF / W_PER_BUS]();
        uint32_t* ff0_flag = new uint32_t [D_MODEL* D_FF / (32 * KERNEL_DIM * MAX_W_COL)]();
        uint32_t* ff1_kernel = new uint32_t [D_FF* D_MODEL / W_PER_BUS]();
        uint32_t* ff1_flag = new uint32_t [D_MODEL* D_FF / (32 * KERNEL_DIM * MAX_W_COL)]();

    #ifdef RELOAD_WEIGHT
        int n = -1; // n=-1 means that we are not saving/loading a head

        loadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL / W_PER_BUS, condense_kernel, sparsityCondense, dir_name, &hidden_flag);
        loadWeight(n, 1, D_MODEL* D_FF / W_PER_BUS, ff0_kernel, sparsityPercentageFF0, dir_name, &hidden_flag);
        loadWeight(n, 2, D_MODEL* D_FF / W_PER_BUS, ff1_kernel, sparsityPercentageFF1, dir_name, &hidden_flag);

        loadWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_W_COL), condense_flag, sparsityCondense, dir_name,
                   nullptr);
        loadWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_W_COL), ff0_flag, sparsityPercentageFF0, dir_name, nullptr);
        loadWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_W_COL), ff1_flag, sparsityPercentageFF1, dir_name, nullptr);

    #else
        fill_sparse_weight(condense_kernel, condense_flag, D_MODEL, NUM_HEAD * D_Q / W_PER_BUS, sparsityCondense);
        fill_sparse_weight(ff0_kernel, ff0_flag, D_MODEL, D_FF / W_PER_BUS, sparsityPercentageFF0);
        fill_sparse_weight(ff1_kernel, ff1_flag, D_FF, D_MODEL / W_PER_BUS, sparsityPercentageFF1);

        int n = -1; // n=-1 means that we are not saving/loading a head
        saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL / W_PER_BUS, condense_kernel, sparsityCondense, dir_name);
        saveWeight(n, 1, D_MODEL* D_FF / W_PER_BUS, ff0_kernel, sparsityPercentageFF0, dir_name);
        saveWeight(n, 2, D_MODEL* D_FF / W_PER_BUS, ff1_kernel, sparsityPercentageFF1, dir_name);

        saveWeight(n, 10, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_W_COL), condense_flag, sparsityCondense, dir_name);
        saveWeight(n, 11, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_W_COL), ff0_flag, sparsityPercentageFF0, dir_name);
        saveWeight(n, 12, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_W_COL), ff1_flag, sparsityPercentageFF1, dir_name);

        append_flags(condense_flag, NUM_HEAD * D_Q * D_MODEL / (32 * KERNEL_DIM * MAX_W_COL));
        append_flags(ff0_flag, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_W_COL));
        append_flags(ff1_flag, D_MODEL* D_FF / (32* KERNEL_DIM * MAX_W_COL));

        outfile.open("flags_generated.h", std::ios::app);
        outfile << "};" << std::endl;
        outfile.close();
    #endif

    weightVec[NUM_HEAD*3] = condense_kernel;
    flagVec[NUM_HEAD*3] = condense_flag;

    weightVec[NUM_HEAD*3+1] = ff0_kernel;
    flagVec[NUM_HEAD*3+1] = ff0_flag;

    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;
    flagVec[NUM_HEAD*3 + 2] = ff1_flag;

    // if (sparseFormat == Format::WITH_FLAG){
    //     for (int i = 0; i < NUM_HEAD * 3; i++){
    //         // call remove_zero_tiles for each weight vector in query, key and value
    //         remove_zero_tiles(weightVec[i], (int) D_MODEL, (int) D_Q >> 2);
    //     }
    //     remove_zero_tiles(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2);
    //     remove_zero_tiles(weightVec[NUM_HEAD*3 + 1], D_MODEL, D_FF >> 2);
    //     remove_zero_tiles(weightVec[NUM_HEAD*3 + 2], D_FF, D_MODEL >> 2);
    // }
    // else if (sparseFormat == Format::HIDDEN_KEY){
    //     for (int i = 0; i < NUM_HEAD * 3; ++i) {
    //         interleave_hidden_flag_zero_free(weightVec[i], D_MODEL,D_Q >> 2, hidden_flag);
    //     }
    //     interleave_hidden_flag_zero_free(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2, hidden_flag);
    //     interleave_hidden_flag_zero_free(weightVec[NUM_HEAD*3+1], D_MODEL, D_FF >> 2, hidden_flag);
    //     interleave_hidden_flag_zero_free(weightVec[NUM_HEAD*3+2], D_FF, D_MODEL >> 2, hidden_flag);
    // }
    // else if(sparseFormat == Format::CSC){
    //     for (int i = 0; i < NUM_HEAD * 3; ++i) {
    //         col_ptr[i] = new int [D_Q / KERNEL_DIM + 1]();
    //         row_ptr[i] = new int [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();
    //         values[i] = new uint32_t* [(D_MODEL * D_Q) / (KERNEL_DIM * KERNEL_DIM)]();
    //         dense2csc(weightVec[i], D_MODEL, D_Q >> 2, col_ptr[i], row_ptr[i], values[i]);
    //     }
    //     col_ptr[NUM_HEAD*3] = new int [D_MODEL / KERNEL_DIM + 1]();
    //     row_ptr[NUM_HEAD*3] = new int [(NUM_HEAD * D_Q * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
    //     values[NUM_HEAD*3] = new uint32_t* [(NUM_HEAD * D_Q * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
    //     dense2csc(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2, col_ptr[NUM_HEAD*3],
    //               row_ptr[NUM_HEAD*3], values[NUM_HEAD*3]);

    //     col_ptr[NUM_HEAD*3+1] = new int [D_FF / KERNEL_DIM + 1]();
    //     row_ptr[NUM_HEAD*3+1] = new int [(D_MODEL * D_FF) / (KERNEL_DIM * KERNEL_DIM)]();
    //     values[NUM_HEAD*3+1] = new uint32_t* [(D_MODEL * D_FF) / (KERNEL_DIM * KERNEL_DIM)]();
    //     dense2csc(weightVec[NUM_HEAD*3+1], D_MODEL, D_FF >> 2, col_ptr[NUM_HEAD*3+1],
    //               row_ptr[NUM_HEAD*3+1], values[NUM_HEAD*3+1]);

    //     col_ptr[NUM_HEAD*3+2] = new int [D_MODEL / KERNEL_DIM + 1]();
    //     row_ptr[NUM_HEAD*3+2] = new int [(D_FF * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
    //     values[NUM_HEAD*3+2] = new uint32_t* [(D_FF * D_MODEL) / (KERNEL_DIM * KERNEL_DIM)]();
    //     dense2csc(weightVec[NUM_HEAD*3+2], D_FF, D_MODEL >> 2, col_ptr[NUM_HEAD*3+2],
    //               row_ptr[NUM_HEAD*3+2], values[NUM_HEAD*3+2]);
    // }
    // else if(sparseFormat == Format::INTERLEAVED){
    //     for (int i = 0; i < NUM_HEAD * 3; ++i) {
    //         dense2interleavedMetaData(weightVec[i], D_MODEL, D_Q >> 2);
    //     }
    //     dense2interleavedMetaData(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL >> 2);
    //     dense2interleavedMetaData(weightVec[NUM_HEAD*3+1], D_MODEL, D_FF >> 2);
    //     dense2interleavedMetaData(weightVec[NUM_HEAD*3+2], D_FF, D_MODEL >> 2);
    // }

    if(sparseFormatQVK == Format::INTERLEAVED) {
        for (int i = 0; i < NUM_HEAD * 3; ++i) {
            std::cout << "dense2interleavedMetaData(" << i << ")" << std::endl;
            dense2interleavedMetaData(weightVec[i], D_MODEL, D_Q / W_PER_BUS);
        }
    } else if (sparseFormatQVK != Format::NON_PRUNED) {
        std::cerr << "sparseFormatQVK not supported!" << std::endl;
        exit(1);
    }

    if(sparseFormatCondense == Format::INTERLEAVED) {
        dense2interleavedMetaData(weightVec[NUM_HEAD*3], NUM_HEAD * D_Q, D_MODEL / W_PER_BUS);
    } else if (sparseFormatCondense != Format::NON_PRUNED) {
        std::cerr << "sparseFormatCondense not supported!" << std::endl;
        exit(1);
    }

    if(sparseFormatFF0 == Format::INTERLEAVED)  {
        dense2interleavedMetaData(weightVec[NUM_HEAD*3+1], D_MODEL, D_FF / W_PER_BUS);
    } else if (sparseFormatFF0 != Format::NON_PRUNED) {
        std::cerr << "sparseFormatFF0 not supported!" << std::endl;
        exit(1);
    }

    if(sparseFormatFF1 == Format::INTERLEAVED) {
        dense2interleavedMetaData(weightVec[NUM_HEAD*3+2], D_FF, D_MODEL / W_PER_BUS);
    } else if (sparseFormatFF1 != Format::NON_PRUNED) {
        std::cerr << "sparseFormatFF1 not supported!" << std::endl;
        exit(1);
    }

    // if (sparseFormat == Format::CSC || sparseFormat == Format::CSR){
    //     TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF,
    //                                weightVec, col_ptr, row_ptr, values, sparseFormat);
    //     selfatten.compute(D_SEQ, tensor_in, out, multihead_out, condense_out, intermediateFF);
    // }else{
        TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF,
                                   weightVec, flagVec, &hidden_flag, sparseFormatQVK, sparseFormatCondense, sparseFormatFF0, sparseFormatFF1);
        selfatten.compute(D_SEQ, tensor_in, out, multihead_out, condense_out, intermediateFF);
    // }

    // Delete the dynamically allocated memory
    delete[] tensor_in;
    delete[] out;
    delete[] multihead_out;
    delete[] condense_out;
    delete[] intermediateFF;
    for (int i = 0; i < NUM_HEAD; ++i) {
        delete[] query_kernel[i];
        delete[] query_flag[i];
        delete[] key_kernel[i];
        delete[] key_flag[i];
        delete[] value_kernel[i];
        delete[] value_flag[i];
    }
    delete[] query_kernel;
    delete[] query_flag;
    delete[] key_kernel;
    delete[] key_flag;
    delete[] value_kernel;
    delete[] value_flag;
    delete[] condense_kernel;   
    delete[] condense_flag;
    delete[] ff0_kernel;
    delete[] ff0_flag;
    delete[] ff1_kernel;
    delete[] ff1_flag;
}

int main(int argc, char** argv) {
    
    if((argc - 1) % 4 != 0 || argc == 1) {
        std::cerr << "Invalid invocation." << std::endl;
        exit(1);
    }

    int numRuns = (argc - 1) / 4;
    char** sparsities = argv + 1;

    for (int i = 0; i < numRuns; ++i) {
        int sparsityPercentageQVK = atoi(sparsities[4*i + 0]);
        int sparsityCondense = atoi(sparsities[4*i + 1]);
        int sparsityPercentageFF0 = atoi(sparsities[4*i + 2]);
        int sparsityPercentageFF1 = atoi(sparsities[4*i + 3]);

        std::cout << "RUN #" << i << "\n\n";
        std::cout << "SW SA_SIZE=" << SA_SIZE << '\n';

        uint64_t kernel_dim = smmReadFlag(0, 0);
        std::cout << "HW KERNEL_DIM=" << kernel_dim << '\n';

        if (kernel_dim != SA_SIZE) {
            std::cerr << "ERROR: SA_SIZE != KERNEL_DIM" << std::endl;
            exit(1);
        }

        std::cout << "MODEL=" << MODEL_STR << '\n';
        std::cout << "sparsityPercentageQVK=" << sparsityPercentageQVK << '\n';
        std::cout << "sparsityCondense=" << sparsityCondense << '\n';
        std::cout << "sparsityPercentageFF0=" << sparsityPercentageFF0 << '\n';
        std::cout << "sparsityPercentageFF1=" << sparsityPercentageFF1 << "\n\n";
        
        Format sparseFormatQVK, sparseFormatCondense, sparseFormatFF0, sparseFormatFF1;
        sparseFormatQVK = sparseFormatCondense = sparseFormatFF0 = sparseFormatFF1 = Format::INTERLEAVED;

        if(!sparsityPercentageQVK) { sparseFormatQVK = Format::NON_PRUNED; }
        if(!sparsityCondense) { sparseFormatCondense = Format::NON_PRUNED; }
        if(!sparsityPercentageFF0) { sparseFormatFF0 = Format::NON_PRUNED; }
        if(!sparsityPercentageFF1) { sparseFormatFF1 = Format::NON_PRUNED; }

        std::cout << "sparseFormatQVK=" << sparseFormatQVK << '\n';
        std::cout << "sparseFormatCondense=" << sparseFormatCondense << '\n';
        std::cout << "sparseFormatFF0=" << sparseFormatFF0 << '\n';
        std::cout << "sparseFormatFF1=" << sparseFormatFF1 << std::endl;

#ifdef SW_MULT
      std::cout << "\nSoftware multiplication\n\n";
      if(sparsityPercentageQVK != 0 || sparsityCondense != 0 || sparsityPercentageFF0 != 0 || sparsityPercentageFF1 != 0) {
        std::cerr << "ERROR: SW mult doesn't support sparsity!\n";
        exit(1);
      }
#endif

        inference(sparsityPercentageQVK, sparsityCondense, sparsityPercentageFF0, sparsityPercentageFF1, sparseFormatQVK, sparseFormatCondense, sparseFormatFF0, sparseFormatFF1);

        // if (sparsity)
        //     inference(sparsity, Format::INTERLEAVED);
        // else
        //     inference(sparsity, Format::NON_PRUNED);
    }
    return 0;
}
