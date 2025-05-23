//
// Created by alireza on 12/18/23.
//

#include "sparse_rep.h"

void fill_sparse_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
#if WEIGHT_FP == 1
        arith_weight_t result;
        result.fp = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/4)) - 2;
        if (rand() % 20 != 0) {
            result.fp = 0.0;
        }
        kernel[i]=result.bin;
#else
        uint32_t result = 0;
        for (int j=0; j<W_PER_BUS; j++){
            int value = rand() % 5 - 2;
            if (rand() % 20 != 0) {
                value = 0;
            }
            result |=  ((weight_t)(value)) << (WEIGHT_BITS * j);
        }
        kernel[i]=result;
#endif
    }
}


int dense2csr(uint32_t* kernel, int n_row, int n_col,
               int* col_ptr, int* row_ptr, uint32_t** values) {
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

        for (int j = 0; j < n_col / MAX_W_COL; j++) {
            int tile_index = (j * (n_row / SA_SIZE) + i) * SA_SIZE * MAX_W_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
#if WEIGHT_FP == 1
                    arith_weight_t result;
                    result.bin = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (result.fp != 0) {
                        all_zeros = false;
                        break;
                    }
#else
                    uint32_t value = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
#endif
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                col_ptr[nnz] = j;
                values[nnz] = &kernel[tile_index];
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }

    return nnz;
}

int dense2csc(uint32_t* kernel, int n_row, int n_col,
               int* col_ptr, int* row_ind, uint32_t** values) {
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
    for (int i = 0; i < n_col / MAX_W_COL; i++) {
        for (int j = 0; j < n_row / SA_SIZE; j++) {
            int tile_index = (i * (n_row / SA_SIZE) + j) * SA_SIZE * MAX_W_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
#if WEIGHT_FP == 1
                    arith_weight_t result;
                    result.bin = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (result.fp != 0) {
                        all_zeros = false;
                        break;
                    }
#else
                    uint32_t value = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
#endif
                }
                if (!all_zeros) {
                    break;
                }
            }

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

void dense2metaData(uint32_t** kernel, int n_row, int n_col,
                    uint32_t* m1, uint32_t* m2){
    //parameters:
    //kernel: dense matrix
    //n_row: number of rows
    //n_col: number of columns
    //m1: metadata1
    //m2: metadata2 (Block Level)
    //values: Non-zero-blocks

    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;

    int m2StartIndex = 0;
    bool columnAllZeros;
    for (int j = 0; j < n_col / MAX_W_COL; j++) {
        columnAllZeros = true;
        for (int i = 0; i < n_row / SA_SIZE; i++) {
            int tile_index = (j * (n_row / SA_SIZE) + i) * SA_SIZE * MAX_W_COL;
            bool tileAllZeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
#if WEIGHT_FP == 1
                    arith_weight_t result;
                    result.bin = (*kernel)[tile_index + ii * MAX_W_COL + jj];
                    if (result.fp != 0) {
                        tileAllZeros = false;
                        break;
                    }
#else
                    uint32_t value = (*kernel)[tile_index + ii * MAX_W_COL + jj];
                    if (value != 0) {
                        tileAllZeros = false;
                        break;
                    }
#endif
                }
                if (!tileAllZeros) {
                    break;
                }
            }
            m2[m2StartIndex + i] = !tileAllZeros;
            if (!tileAllZeros) {
                columnAllZeros = false;
                // copy the tile to values
                // values only contains non-zero blocks
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_W_COL; jj++) {
                        *new_kernel ++ = (*kernel)[tile_index + ii * MAX_W_COL + jj];
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

    *kernel = new_kernel_ptr;
}

void dense2interleavedMetaData(uint32_t*& kernel, int n_row, int n_col) {
    //parameters:
    //kernel: dense matrix (input and eventual output),
    //      Non-zero-blocks with the metadata interleaved
    //      For each block, first 32 bits offset to the next block,
    //      next BLOCK_SIZE/32 bits are the metadata,
    //      next uint32_t until the next block are the block values 
    //n_row: number of rows
    //n_col: number of columns

    // In this first implementation, BLOCK_SIZE is one column of tiles = n_row / SA_SIZE
    uint BLOCK_SIZE = n_row / SA_SIZE;

    uint32_t * new_kernel;
    uint32_t new_kernel_size = 100 + (n_row * n_col) + (n_row * n_col * MAX_W_COL / (32 * SA_SIZE * SA_SIZE)) + n_col * MAX_W_COL / SA_SIZE;

    new_kernel = new uint32_t[new_kernel_size]();
    uint32_t * new_kernel_ptr = new_kernel;

    uint32_t* start_index = new_kernel;
    uint32_t counter32 = 0;
    uint32_t* offset_to_next_block;
    uint32_t* metadata;
    uint32_t* values_start; // Pointer to the start of the values, to avoid overwriting with metadata
    uint metadata_block_size = (BLOCK_SIZE + 32 - 1) / 32;  // Number of uint32_t to store the metadata of a block (rounding up)
    for (int j = 0; j < n_col / MAX_W_COL; j++) {
        offset_to_next_block = new_kernel++;    // Here we will store the offset at the end of the block
        metadata = new_kernel;                  // Here we will store the metadata of the block
        new_kernel += metadata_block_size;      // Here we will store the values of the block
        values_start = metadata + metadata_block_size;
        counter32 = 0;
        for (int i = 0; i < n_row / SA_SIZE; i++) {
            int tile_index = (j * (n_row / SA_SIZE) + i) * SA_SIZE * MAX_W_COL;
            // std::cout << std::dec << "tile index : "<<tile_index << "\t";
            // std::cout << std::hex << kernel[tile_index] <<std::endl;
            bool tileAllZeros = true;

            // Check if the tile is all zeros
            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
#if WEIGHT_FP == 1
                    arith_weight_t result;
                    result.bin = kernel[tile_index + ii * MAX_W_COL + jj];
                    // std::cout << std::hex << result.bin << ", ";
                    if (result.fp != 0) {
                        tileAllZeros = false;
                        break;
                    }
#else
                    uint32_t value = kernel[tile_index + ii * MAX_W_COL + jj];
                    // std::cout << std::hex << value << ", ";
                    if (value != 0) {
                        tileAllZeros = false;
                        break;
                    }
#endif
                }
                if (!tileAllZeros) {
                    break;
                }
            }
            // std::cout << std::endl;
            // If the tile is not all zeros, copy the tile to new_kernel
            if (!tileAllZeros) {
                // copy the tile to new_kernel
                // new_kernel only contains non-zero blocks
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_W_COL; jj++) {
                        *(new_kernel++) = kernel[tile_index + ii * MAX_W_COL + jj];
                    }
                }
                *metadata |= (0x00000001 << counter32);
            }
            counter32++;

            if (counter32 == 32) {
                if (++metadata < values_start) {
                    *metadata = 0;
                }
                counter32 = 0;
            }
        }
        *offset_to_next_block = new_kernel - offset_to_next_block;  // Store the offset to the next block
    }
    kernel = new_kernel_ptr;
}


void remove_zero_tiles(uint32_t*& kernel, int n_row, int n_col) {
    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;
    int counter = 0;

    for (int i = 0; i < n_row / KERNEL_DIM; i++) {
        for (int j = 0; j < n_col / MAX_W_COL; j++) {
            int tile_index = (i * (n_col / MAX_W_COL) + j) * KERNEL_DIM * MAX_W_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < KERNEL_DIM; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
#if WEIGHT_FP == 1
                    arith_weight_t result;
                    result.bin = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (result.fp != 0) {
                        all_zeros = false;
                        break;
                    }
#else
                    uint32_t value = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
#endif
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                for (int ii = 0; ii < KERNEL_DIM; ii++) {
                    for (int jj = 0; jj < MAX_W_COL; jj++) {
                        *new_kernel ++ = kernel[tile_index + ii * MAX_W_COL + jj];
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


void interleave_hidden_flag_zero_free(uint32_t*& kernel, int n_row, int n_col, uint32_t hidden_flag) {
    // It removes all zero tiles and replace them with hidden_flag
    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;
    int counter = 0;

    for (int i = 0; i < n_row / SA_SIZE; i++) {
        for (int j = 0; j < n_col / MAX_W_COL; j++) {  //TODO: reverse the order of i and j
            int tile_index = (i * (n_col / MAX_W_COL) + j) * SA_SIZE * MAX_W_COL;
            bool all_zeros = true;

            // We can use the following loop to check if the tile is zero or not
            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_W_COL; jj++) {
#if WEIGHT_FP == 1
                    arith_weight_t result;
                    result.bin = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (result.fp != 0) {
                        all_zeros = false;
                        break;
                    }
#else
                    uint32_t value = kernel[tile_index + ii * MAX_W_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
#endif
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_W_COL; jj++) {
                        *new_kernel ++ = kernel[tile_index + ii * MAX_W_COL + jj];
                    }
                }
            }
            else{
                *new_kernel ++ = hidden_flag;
                counter ++;
            }
        }
    }
    kernel = new_kernel_ptr;
}


void dense2csr_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/W_PER_BUS]();
    fill_sparse_kernel(kernel, row_size * col_size / W_PER_BUS);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/W_PER_BUS]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/W_PER_BUS);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/W_PER_BUS; j++){
            std::cout << std::hex << rowWise[i*col_size/W_PER_BUS+j] << " ";
        }
        std::cout << std::endl;
    }

    int *row_ptr;
    int* col_ind;
    uint32_t** values;
    row_ptr = new int [row_size / SA_SIZE + 1]();
    col_ind = new int [(row_size * col_size) / (SA_SIZE * MAX_W_COL)]();
    values = new uint32_t* [(row_size * col_size) / (SA_SIZE * MAX_W_COL)]();

    int nnz = dense2csr(kernel, row_size, col_size/W_PER_BUS, col_ind, row_ptr, values);

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
        for (int j = 0; j < SA_SIZE * MAX_W_COL; j++) {
            std::cout << std::hex << values[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void dense2csc_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/W_PER_BUS]();
    fill_sparse_kernel(kernel, row_size * col_size / W_PER_BUS);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/W_PER_BUS]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/W_PER_BUS);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/W_PER_BUS; j++){
            std::cout << std::hex << rowWise[i*col_size/W_PER_BUS+j] << " ";
        }
        std::cout << std::endl;
    }

    int *col_ptr;
    int* row_ind;
    uint32_t** values;
    col_ptr = new int [col_size / SA_SIZE + 1]();
    row_ind = new int [(row_size * col_size) / (SA_SIZE * MAX_W_COL)]();
    values = new uint32_t* [(row_size * col_size) / (SA_SIZE * MAX_W_COL)]();

    int nnz = dense2csc(kernel, row_size, col_size/W_PER_BUS, col_ptr, row_ind, values);

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
        for (int j = 0; j < SA_SIZE * MAX_W_COL; j++) {
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

    uint32_t* kernel = new uint32_t [row_size*col_size/W_PER_BUS]();
    fill_sparse_kernel(kernel, row_size * col_size / W_PER_BUS);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/W_PER_BUS]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/W_PER_BUS);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/W_PER_BUS; j++){
            std::cout << std::hex << rowWise[i*col_size/W_PER_BUS+j] << " ";
        }
        std::cout << std::endl;
    }


    uint32_t* m1;
    uint32_t* m2;
    m1 = new uint32_t [(col_size/SA_SIZE)]();
    m2 = new uint32_t [((row_size * col_size) / (SA_SIZE* SA_SIZE))]();

    dense2metaData(&kernel, row_size, col_size/W_PER_BUS, m1, m2); // TODO: it is changed! check it again.

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
    for (int i = 0; i < row_size * col_size / W_PER_BUS; i++) {
            std::cout << std::hex << kernel[i] << " ";
        }
    std::cout << std::endl;

}

// test function for dense2interleavedMetaData
void dense2interleavedMetaData_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    //define the metadata sizes
    uint BLOCK_SIZE = row_size / SA_SIZE;
    uint metadata_block_size = (BLOCK_SIZE + 32 - 1) / 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/W_PER_BUS]();
    fill_sparse_kernel(kernel, row_size * col_size / W_PER_BUS);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/W_PER_BUS]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/W_PER_BUS);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/W_PER_BUS; j++){
            std::cout << std::hex << rowWise[i*col_size/W_PER_BUS+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dense2interleavedMetaData(kernel, row_size, col_size/W_PER_BUS);

    // for each column, print the metadata and the values
    for (int i = 0; i < col_size / SA_SIZE; i++) {
        std::cout << "column: " << std::dec << i << std::endl;
        uint32_t* cur_block = kernel;
        uint32_t* next_block = cur_block + *kernel;
        std::cout << "offset: " << std::dec << *(kernel++) << std::endl;
        
        std::cout << "metadata: ";
        for (int j = 0; j < metadata_block_size; j++) {
            std::cout << std::hex << *(kernel++) << " ";
        }
        std::cout << std::endl;

        std::cout << "values: ";
        while (kernel < next_block) {
            std::cout << std::hex << *(kernel++) << " ";
        }

        std::cout << std::endl << std::endl;
    }
}


// //  create a main
// int main() {
//    dense2interleavedMetaData_test();
//    return 0;
// }


