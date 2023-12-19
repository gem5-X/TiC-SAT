//
// Created by alireza on 12/18/23.
//

#include "sparse_rep.h"
#define MAX_COL (SA_SIZE/4)

void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        uint32_t result = 0;
        for (int j=0; j<4; j++){
            int value = rand() % 5 - 2;
            if (rand() % 20 != 0) {
                value = 0;
            }
            result |=  ((uint8_t)(value)) << (8 * j);
        }
        kernel[i]=result;
    }
}


int dense2csr(uint32_t* kernel, int n_row, int n_col,
               uint32_t* row_ptr, uint32_t* col_ind, uint32_t** values) {
    //parameters:
    //kernel: dense matrix
    //n_row: number of rows
    //n_col: number of columns
    //row_ptr: row pointer of csr matrix
    //col_ind: column index of csr matrix
    //values: value pointers of csr matrix

    //initialize new row_ptr, col_ind, values

    int nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < n_row / SA_SIZE; i++) {

        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (j * (n_row / SA_SIZE) + i) * SA_SIZE * MAX_COL;
            std::cout << std::dec << "tile index : "<<tile_index << "\t";
            std::cout << std::hex << kernel[tile_index] <<std::endl;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    std::cout << std::hex << value << ", ";
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }
            std::cout <<    std::endl;

            if (!all_zeros) {
                col_ind[nnz] = j;
                values[nnz] = &kernel[tile_index];
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }

    return nnz;
}

int dense2csc(uint32_t* kernel, int n_row, int n_col,
               uint32_t* col_ptr, uint32_t* row_ind, uint32_t** values) {
    //parameters:
    //kernel: dense matrix
    //n_row: number of rows
    //n_col: number of columns
    //col_ptr: column pointer of csc matrix
    //row_ind: row index of csc matrix
    //values: value pointers of csc matrix

    //initialize new row_ptr, col_ind, values

    int nnz = 0;
    col_ptr[0] = 0;
    for (int i = 0; i < n_col / MAX_COL; i++) {
        for (int j = 0; j < n_row / SA_SIZE; j++) {
            int tile_index = (i * (n_row / SA_SIZE) + j) * SA_SIZE * MAX_COL;
            std::cout << std::dec << "tile index : "<<tile_index << "\t";
            std::cout << std::hex << kernel[tile_index] <<std::endl;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    std::cout << std::hex << value << ", ";
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }
            std::cout <<    std::endl;

            if (!all_zeros) {
                row_ind[nnz] = j;
                values[nnz] = &kernel[tile_index];
                nnz++;
            }
        }
        col_ptr[i+1] = nnz;
    }
    return nnz;
}

void dense2metaData(uint32_t* kernel, int n_row, int n_col,
                   bool* m1, bool* m2, uint32_t* values){
    //parameters:
    //kernel: dense matrix
    //n_row: number of rows
    //n_col: number of columns
    //m1: metadata1
    //m2: metadata2 (Block Level)
    //values: Non-zero-blocks

    int m2StartIndex = 0;
    bool columnAllZeros;
    for (int j = 0; j < n_col / MAX_COL; j++) {
        columnAllZeros = true;
        for (int i = 0; i < n_row / SA_SIZE; i++) {
            int tile_index = (j * (n_row / SA_SIZE) + i) * SA_SIZE * MAX_COL;
            std::cout << std::dec << "tile index : "<<tile_index << "\t";
            std::cout << std::hex << kernel[tile_index] <<std::endl;
            bool tileAllZeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    std::cout << std::hex << value << ", ";
                    if (value != 0) {
                        tileAllZeros = false;
                        break;
                    }
                }
                if (!tileAllZeros) {
                    break;
                }
            }
            std::cout << std::endl;
            m2[m2StartIndex + i] = !tileAllZeros;
            if (!tileAllZeros) {
                columnAllZeros = false;
                // copy the tile to values
                // values only contains non-zero blocks
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_COL; jj++) {
                        *values ++ = kernel[tile_index + ii * MAX_COL + jj];
                    }
                }
            }
        }
        if (columnAllZeros) {
            m1[j] = false; // column is all zeros
            // m2StartIndex is not updated so rewrite the same column
        }else{
            m1[j] = true; // column is not all zeros
            m2StartIndex += n_row / SA_SIZE;
        }
    }
}


void dense2csr_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/4]();
    fill_kernel(kernel, row_size*col_size/4);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/4]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/4);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/4; j++){
            std::cout << std::hex << rowWise[i*col_size/4+j] << " ";
        }
        std::cout << std::endl;
    }

    uint32_t *row_ptr;
    uint32_t* col_ind;
    uint32_t** values;
    row_ptr = new uint32_t [row_size / SA_SIZE + 1]();
    col_ind = new uint32_t [(row_size * col_size) / (SA_SIZE * MAX_COL)]();
    values = new uint32_t* [(row_size * col_size) / (SA_SIZE * MAX_COL)]();

    int nnz = dense2csr(kernel, row_size, col_size/4, row_ptr, col_ind, values);

    std::cout << std::dec << std::endl;
    std::cout<< "nnz: " << nnz << std::endl;
    std::cout << "row_ptr:" << std::endl;
    for (int i = 0; i < row_size / SA_SIZE + 1; i++) {
        std::cout << row_ptr[i] << " ";
    }
    std::cout << std::endl;

   //print the col_ind
    std::cout << "col_ind:" << std::endl;
    for (int i = 0; i < nnz; i++) {
            std::cout << col_ind[i] << " ";
        }
    std::cout << std::endl;
    //print the values
    std::cout << "values:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        for (int j = 0; j < SA_SIZE * MAX_COL; j++) {
            std::cout << std::hex << values[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void dense2csc_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/4]();
    fill_kernel(kernel, row_size*col_size/4);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/4]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/4);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/4; j++){
            std::cout << std::hex << rowWise[i*col_size/4+j] << " ";
        }
        std::cout << std::endl;
    }

    uint32_t *col_ptr;
    uint32_t* row_ind;
    uint32_t** values;
    col_ptr = new uint32_t [col_size / SA_SIZE + 1]();
    row_ind = new uint32_t [(row_size * col_size) / (SA_SIZE * MAX_COL)]();
    values = new uint32_t* [(row_size * col_size) / (SA_SIZE * MAX_COL)]();

    int nnz = dense2csc(kernel, row_size, col_size/4, col_ptr, row_ind, values);

    std::cout << std::dec << std::endl;
    std::cout<< "nnz: " << nnz << std::endl;
    std::cout << "col_ptr:" << std::endl;
    for (int i = 0; i < col_size / SA_SIZE + 1; i++) {
        std::cout << col_ptr[i] << " ";
    }
    std::cout << std::endl;

   //print the row_ind
    std::cout << "row_ind:" << std::endl;
    for (int i = 0; i < nnz; i++) {
            std::cout << row_ind[i] << " ";
        }
    std::cout << std::endl;
    //print the values
    std::cout << "values:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        for (int j = 0; j < SA_SIZE * MAX_COL; j++) {
            std::cout << std::hex << values[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// test function for dense2metaData
void dense2metaData_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/4]();
    fill_kernel(kernel, row_size*col_size/4);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/4]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/4);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/4; j++){
            std::cout << std::hex << rowWise[i*col_size/4+j] << " ";
        }
        std::cout << std::endl;
    }


    bool*m1;
    bool* m2;
    uint32_t* values;
   values = new uint32_t [(row_size * col_size/4)]();
    m1 = new bool [(col_size/SA_SIZE)]();
    m2 = new bool [((row_size * col_size) / (SA_SIZE* SA_SIZE))]();

    dense2metaData(kernel, row_size, col_size/4, m1, m2, values);

    std::cout << std::dec << std::endl;
    std::cout << "m1:" << std::endl;
    for (int i = 0; i < col_size/SA_SIZE; i++) {
        std::cout << (int) m1[i] << " ";
    }
    std::cout << std::endl;

    //print the m2
    std::cout << "m2:" << std::endl;
    for (int i = 0; i < (row_size * col_size) / (SA_SIZE* SA_SIZE); i++) {
            std::cout << (int) m2[i] << " ";
        }
    std::cout << std::endl;
    //print the values
    std::cout << "values:" << std::endl;
    for (int i = 0; i < row_size * col_size / 4; i++) {
            std::cout << std::hex << values[i] << " ";
        }
    std::cout << std::endl;

}


//  create a main
int main() {
    dense2metaData_test();
    return 0;
}


