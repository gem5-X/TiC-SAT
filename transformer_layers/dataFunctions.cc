//
// Created by alireza on 2/7/24.
//

#include "dataFunctions.h"
#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/4)


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * j);
        }
        kernel[i]=result;
    }
}

void print_binary(uint32_t value) {
    for (int i = 31; i >= 0; i--) {
        std::cout << ((value >> i) & 1);
    }
    std::cout<< std::endl;
}

void fill_sparse_weight(uint32_t * kernel, uint32_t* sparse_flag, int n_row, int n_col, int sparsity){
    auto *flag_ptr = sparse_flag;
    uint32_t *kernel_ptr = kernel;
    int counter32 = 0;
    uint32_t flag32 = 0;
    for (int i=0; i<n_row/KERNEL_DIM; i++){
        for (int j=0; j<n_col/MAX_COL; j++){
            if (rand() % 100 >= sparsity){
                int tile_index = (i * (n_col/MAX_COL) + j) * KERNEL_DIM * MAX_COL;
                for (int ii=0; ii<KERNEL_DIM; ii++){
                    for (int jj=0; jj<MAX_COL; jj++){
                        uint32_t result = 0;
                        for (int k=0; k<4; k++){
                            result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * k);
                        }
                        //                        kernel[tile_index + ii * MAX_COL + jj]=result;
                        *kernel_ptr=result;
                        kernel_ptr++;
                    }
                }
                counter32++;
            }
            else{
                for (int ii=0; ii<KERNEL_DIM*MAX_COL; ii++){
                    *kernel_ptr = 0x0;
                    kernel_ptr++;
                }
                flag32 |= 0x00000001;
                counter32++;
            }

            if (counter32 == 32){
                *flag_ptr = flag32;
                flag_ptr++;
                counter32=0;
                flag32 = 0;
            }
            else{
                flag32 = flag32 << 1;
            }
        }
    }
}


void remove_zero_tiles(uint32_t*& kernel, int n_row, int n_col) {
    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;
    int counter = 0;

    for (int i = 0; i < n_row / KERNEL_DIM; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (i * (n_col / MAX_COL) + j) * KERNEL_DIM * MAX_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < KERNEL_DIM; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                for (int ii = 0; ii < KERNEL_DIM; ii++) {
                    for (int jj = 0; jj < MAX_COL; jj++) {
                        *new_kernel ++ = kernel[tile_index + ii * MAX_COL + jj];
                    }
                }
            }
            else{
                counter ++;
            }
        }
    }
    kernel = new_kernel_ptr;
}

void load_kernel_from_file(std::vector<uint32_t> &kernel, int n_row, int n_col, const char *filename) {
    std::ifstream file(filename, std::ios::binary);
    kernel.resize(n_row * n_col);
    file.read(reinterpret_cast<char *>(kernel.data()), n_row * n_col * sizeof(uint32_t));
    file.close();
}

void saveWeight(int n_head, int qkv, int size, uint32_t *array, int sparsity_level, const std::string &dir_name) {
    // Write the kernel array to file
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + "_S" + std::to_string(sparsity_level) + ".bin";
    std::ofstream fout(filename);
    if (fout.is_open()) {
        for (int i = 0; i < size; i++) {
            fout << array[i] << " ";
        }
        fout.close();
    }
}

void loadWeight(int n_head, int qkv, int size, uint32_t * array,  int sparsity_level, const std::string &dir_name,
                const uint32_t* hidden_flag){
    bool hidden_flag_check = (hidden_flag != nullptr);
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + "_S" + std::to_string(sparsity_level) + ".bin";
    std::ifstream fin(filename);
    if (fin.is_open()) {
        for (int i = 0; i < size; i++) {
            fin >> array[i];
            if (hidden_flag_check)
                if (array[i] == *hidden_flag){
                    std::cout << *hidden_flag << " is found in the arrays!" << std::endl;
                    exit(404);
                }
        }
        fin.close();
    }
    else{
        std::cout << filename + " Not loaded" << std::endl;
    }

}

void append_flags(uint32_t* new_flags, int new_flags_count) {
    std::ofstream outfile("flags_generated.h", std::ios::app);
    for (int i = 0; i < new_flags_count; i++) {
        outfile << new_flags[i] << ", ";
    }
    outfile.close();
}