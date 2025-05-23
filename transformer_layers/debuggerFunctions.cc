//
// Created by alireza on 4/24/23.
//

#include "debuggerFunctions.h"

void print_weight(uint32_t* kernel, int n_row, int n_col){
    for (int i=0; i< n_row; i++){
        for (int j=0; j<n_col; j++){
            printf("0x%08x,\t", kernel[i*n_col + j]);
        }
        printf("\n");
    }
}
void write_weight_to_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            file.write(reinterpret_cast<const char*>(&kernel[i * n_col + j]), sizeof(uint32_t));
        }
    }

    file.close();
}

void read_weight_from_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            file.read(reinterpret_cast<char*>(&kernel[i * n_col + j]), sizeof(uint32_t));
        }
    }

    file.close();
}



void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col){
    uint32_t* initialRowWise = rowWise;
    for (int col=0; col<n_col/MAX_W_COL; col++){
        rowWise = initialRowWise + col * MAX_W_COL;
        for (int row=0; row < n_row; row++){
            for (int i=0; i<MAX_W_COL; i++){
                *(rowWise + i) = *(blockWise+ i);
            }
            blockWise += MAX_W_COL;
            rowWise += n_col;
        }
    }
}

void rowWise2BlockWise(const uint32_t* rowWise, uint32_t* blockWise, int n_row, int n_col) {
    const uint32_t* initialRowWise = rowWise;
    for (int col = 0; col < n_col / MAX_W_COL; col++) {
        rowWise = initialRowWise + col * MAX_W_COL;
        for (int row = 0; row < n_row; row++) {
            for (int i = 0; i < MAX_W_COL; i++) {
                *(blockWise + i) = *(rowWise + i);
            }
            rowWise += n_col;
            blockWise += MAX_W_COL;
        }
    }
}


void interleave_hidden_flag(uint32_t* kernel, int n_row, int n_col, uint32_t hidden_flag) {
    for (int i = 0; i < n_row / SA_SIZE; i++) {
        for (int j = 0; j < n_col / MAX_W_COL; j++) {
            int tile_index = (i * (n_col / MAX_W_COL) + j) * SA_SIZE * MAX_W_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (all_zeros) {
                kernel[tile_index] = hidden_flag;
            }
        }
    }
}
