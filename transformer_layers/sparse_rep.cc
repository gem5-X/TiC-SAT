//
// Created by alireza on 12/18/23.
//

#include "sparse_rep.h"

void fill_sparse_kernel(uint32_t* kernel, int kernel_size){
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

void dense2interleavedMetaData(uint32_t* kernel, int n_row, int n_col, uint32_t* values, uint* size) {
    //parameters:
    //kernel: dense matrix
    //n_row: number of rows
    //n_col: number of columns
    //values: Non-zero-blocks with the metadata interleaved
    //      For each block, first 32 bits offset to the next block,
    //      next BLOCK_SIZE/32 bits are the metadata,
    //      next uint32_t until the next block are the block values 
    //size: true size of values

    // In this first implementation, BLOCK_SIZE is one column of tiles = n_row / SA_SIZE
    uint BLOCK_SIZE = n_row / SA_SIZE;

    uint32_t* start_index = values;
    uint32_t counter32 = 0;
    uint32_t* offset_to_next_block;
    uint32_t* metadata;
    uint metadata_block_size = (BLOCK_SIZE + 32 - 1) / 32;  // Number of uint32_t to store the metadata of a block (rounding up)
    for (int j = 0; j < n_col / MAX_COL; j++) {
        offset_to_next_block = values++;    // Here we will store the offset at the end of the block
        metadata = values;                  // Here we will store the metadata of the block
        values += metadata_block_size;      // Here we will store the values of the block
        counter32 = 0;
        for (int i = 0; i < n_row / SA_SIZE; i++) {
            int tile_index = (j * (n_row / SA_SIZE) + i) * SA_SIZE * MAX_COL;
            std::cout << std::dec << "tile index : "<<tile_index << "\t";
            std::cout << std::hex << kernel[tile_index] <<std::endl;
            bool tileAllZeros = true;

            // Check if the tile is all zeros
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
            // If the tile is not all zeros, copy the tile to values
            if (!tileAllZeros) {
                // copy the tile to values
                // values only contains non-zero blocks
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_COL; jj++) {
                        *(values++) = kernel[tile_index + ii * MAX_COL + jj];
                    }
                }
                *metadata |= 0x00000001;
            }
            counter32++;

            if (counter32 == 32) {
                *(++metadata) = 0; // Advance metadata pointer and reset to zero
                counter32 = 0;
            } else if (i != n_row / SA_SIZE - 1) {
                *metadata <<= 1;
            }
        }
        *offset_to_next_block = values - offset_to_next_block;  // Store the offset to the next block
    }
    *size = values - start_index;
}


void dense2csr_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/4]();
    fill_sparse_kernel(kernel, row_size * col_size / 4);

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
    fill_sparse_kernel(kernel, row_size * col_size / 4);

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
    fill_sparse_kernel(kernel, row_size * col_size / 4);

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

// test function for dense2interleavedMetaData
void dense2interleavedMetaData_test() {
    //define row and col sizes
    int row_size = 16;
    int col_size = 32;

    //define the metadata sizes
    uint BLOCK_SIZE = row_size / SA_SIZE;
    uint metadata_block_size = (BLOCK_SIZE + 32 - 1) / 32;

    uint32_t* kernel = new uint32_t [row_size*col_size/4]();
    fill_sparse_kernel(kernel, row_size * col_size / 4);

    //print the dense format
    uint32_t *rowWise = new uint32_t [row_size*col_size/4]();
    blockWise2RowWise(kernel, rowWise, row_size, col_size/4);

    for(int i=0; i<row_size; i++){
        for(int j=0; j<col_size/4; j++){
            std::cout << std::hex << rowWise[i*col_size/4+j] << " ";
        }
        std::cout << std::endl;
    }

    uint32_t* values;
    // Max size of values holds all tiles, metadata and offsets
    values = new uint32_t [(row_size * col_size/4) + (metadata_block_size * col_size/4) * col_size/4]();
    uint32_t size = 0;

    dense2interleavedMetaData(kernel, row_size, col_size/4, values, &size);
    std::cout << "size: " << size << std::endl;

    // for each column, print the metadata and the values
    for (int i = 0; i < col_size / 4; i++) {
        std::cout << "column: " << std::dec << i << std::endl;
        uint32_t* cur_block = values;
        uint32_t* next_block = cur_block + *values;
        std::cout << "offset: " << std::dec << *(values++) << std::endl;
        
        std::cout << "metadata: ";
        for (int j = 0; j < metadata_block_size; j++) {
            std::cout << std::hex << *(values++) << " ";
        }
        std::cout << std::endl;

        std::cout << "values: ";
        while (values < next_block) {
            std::cout << std::hex << *(values++) << " ";
        }

        std::cout << std::endl << std::endl;
    }
}


//  create a main
//int main() {
//    dense2interleavedMetaData_test();
//    return 0;
//}


