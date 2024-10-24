//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16]; 
  uint8_t rows[16]; 
} __tilecfg;

/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo)
{
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  for (i = 0; i < 1; ++i)
  {
    tileinfo->colsb[i] = MAX_ROWS;
    tileinfo->rows[i] =  MAX_ROWS;
  }

  for (i = 1; i < 4; ++i)
  {
    tileinfo->colsb[i] = MAX_COLS;
    tileinfo->rows[i] =  MAX_ROWS;
  }

  _tile_loadconfig (tileinfo);
}

/* Initialize uint8_t buffer */
static void init_buffer (uint8_t *buf, uint8_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;

  for (i = 0; i < rows; i++)
    for (j = 0; j < colsb; j++)
    {
        buf[i * colsb + j] = value;
    }
}

/* Initialize random uint16_t buffer */
static void init_random_buffer16(uint16_t *buf, uint32_t size)
{
  for (uint32_t i = 0; i < size; i++)
  {
    buf[i] = rand() % 256; // Random values between 0 and 255
  }
}

/* Initialize constant uint16_t buffer */
static void init_const_buffer16(uint16_t *buf, uint32_t value)
{
  for (uint32_t i = 0; i < MAX; i++)
  {
    buf[i] = value; // Set all values to the constant
  }
}

/* Ensures values are within Z_257*/
static void ensure_correct_range(uint16_t *buf, uint32_t size)
{
  for (uint32_t i = 0; i < size; i++){
    assert(buf[i] <= 256);
  }
}

/* Initialize uint32_t buffer */
static void init_buffer32 (uint32_t *buf, uint32_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;
  int colsb2=colsb/4;

  for (i = 0; i < rows; i++)
    for (j = 0; j < (colsb2); j++)
    {
        buf[i * colsb2 + j] = value;
    }
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

/* Print uint8_t buffer */
static void print_buffer8(uint8_t* buf, uint32_t rows, uint32_t colsb) 
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Print uint16_t buffer */
static void print_buffer16(uint16_t* buf, uint32_t rows, uint32_t colsb) 
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Print uint32_t buffer */
static void print_buffer32(uint32_t* buf, uint32_t rows, uint32_t colsb)
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Naive matmul uint16_t matmul w/ zero extension to uint32_t */
static void naive_matmul(uint16_t *A, uint16_t *B, uint32_t *c, int A_rows, int A_cols, int B_cols)
{
  for (int i = 0; i < A_rows; i++) {
    for (int j = 0; j < B_cols; j++) {

      for (int k = 0; k < A_cols; k++){
        // zero extend mul args from 16->32
        c[i * B_cols + j] += (uint32_t)A[i * A_cols + k] * (uint32_t)B[k * B_cols + j];
      }
    }
  }
} 

/**
 * For the purposes of accelerating LibSWIFFT, we are working under Z_257. This means
 * that to represent the full range of values 0-256, we need at least 9 bits of precision.
 * 
 * This function splits a uint16_t array into two separate uint8_t arrays, one containing
 * the low 8 bits and the other containing the high 1 bit of each element. For which these
 * arrays can be operated on seperately, and later rejoined into a single integer.
 */
void bit_split(const uint16_t *input, uint8_t *low_bits, uint8_t *high_bits, size_t length) {
    for (size_t i = 0; i < length; ++i) {

        // Extract the low 8 bits (bits 0-7)
        low_bits[i] = input[i] & 0xFF; // 0xFF = 11111111 in binary
        
        // Extract the high 1 bits (bit 9)
        high_bits[i] = (input[i] >> 8) & 0x01; // 0x01 = 00000001 in binary 
    }
}

/**
 * After taking an int16_t array and splitting it into two uint8_t arrays, where the low
 * 8 bits are in one array and the high 1 bit are in another.
 * 
 * In this case, two low-low uint6_t and two high-high uint8_t arrays have their respective matmul 
 * computing using AMX. Internally, the AMX _tile_dpbuud instruction will zero extend the 8-bit 
 * integers to 32-bit integers. 
 * 
 * Thsi function accepts the two int32_t AMX matmul results for low and high bits, and recombines
 * them by bit-shifting the 1 high bit left by 8 and adding the low bits with an OR operation.
 */
void bit_recombine(const uint32_t *low_bits, const uint32_t *high_bits, uint32_t *output, size_t length) {

    // create uint64_t output to avoid overflow
    uint64_t output_temp_64[256] = {0};
    memset(output_temp_64, 0, sizeof(output_temp_64));

    for (size_t i = 0; i < length; ++i) {
        // Shift the high bits left by 8 and add the low bits
        output_temp_64[i] = (high_bits[i] << 8) | low_bits[i];
    }

    // Cast back to uint32_t
    for (size_t i = 0; i < length; ++i) {
        output[i] = (uint32_t)(output_temp_64[i] & 0xFFFFFFFF);
    }
}

void modular_reduction(uint32_t *buf, uint32_t modulus, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        buf[i] %= modulus;
    }
}

/**
 * performs matmul of two int16_t buffers representing 16x64 by 64x16 matrices 
 * utilizing bit-splitting of 16-bit input buffers into two 8-bit buffers, in 
 * order to perform AMX matmul on the low and high bits separately. low-low
 * and high-high matmuls are recombined into a single int32_t buffer representing
 * a 16x16 matrix.
 * 
 * All argument buffers must be pre-initialized, w/ the output buffer containing
 * all 0s.
 */
void bit_split_amx_matmul_int16_t(const int16_t* src1, const int16_t*  src2, uint32_t* res, size_t length) {

    // uint8_t arrays for low 5 bits and high 4 bits of input buffers
    uint8_t src1_low_bits[MAX] __attribute__((aligned(64)));
    uint8_t src1_high_bits[MAX] __attribute__((aligned(64)));
    uint8_t src2_low_bits[MAX] __attribute__((aligned(64)));
    uint8_t src2_high_bits[MAX] __attribute__((aligned(64)));  

    // init 32-bit result buffers for low and high bits
    uint32_t amx_res_32_low_bits[MAX/4] __attribute__((aligned(64)));
    uint32_t amx_res_32_high_bits[MAX/4] __attribute__((aligned(64)));
    init_buffer32(amx_res_32_low_bits, 0);  
    init_buffer32(amx_res_32_high_bits, 0);

    // Split input buffers into low and high bits
    bit_split(src1, src1_low_bits, src1_high_bits, MAX);
    bit_split(src2, src2_low_bits, src2_high_bits, MAX);

    // Load low bit data into AMX tiles from memory
    _tile_loadd (2, src1_low_bits, STRIDE);
    _tile_loadd (3, src2_low_bits, STRIDE);
    _tile_loadd (1, amx_res_32_low_bits, STRIDE);

    // Compute dot-product of bytes in tiles 
    _tile_dpbuud (1, 2, 3);

    // Store low bit tile data to memory
    _tile_stored (1, amx_res_32_low_bits, STRIDE);

    // Load high bit data into AMX tiles from memory
    _tile_loadd (2, src1_high_bits, STRIDE);
    _tile_loadd (3, src2_high_bits, STRIDE);
    _tile_loadd (1, amx_res_32_high_bits, STRIDE);

    // Compute dot-product of bytes in tiles
    _tile_dpbuud (1, 2, 3);

    // Store high bit tile data to memory
    _tile_stored (1, amx_res_32_high_bits, STRIDE);

    // Recombine low and high bits into final result
    bit_recombine(amx_res_32_low_bits, amx_res_32_high_bits, res, MAX/4);


}

int main(){

  for (int constant=0; constant<257 ; constant++){
    // seed random input
    srand(time(NULL));

    // tile configuration and metadata
    __tilecfg tile_data = {0};
    int rows  = MAX_ROWS;
    int colsb = MAX_COLS;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
      exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

    // two uint16_t src buffers
    uint16_t src1_16[MAX] __attribute__((aligned(64)));
    uint16_t src2_16[MAX] __attribute__((aligned(64)));
    uint32_t amx_res_32[MAX/4];
    uint32_t naive_res_32[MAX/4];

    // Init random int16_t buffers
    // init_random_buffer16(src1_16, MAX);
    // init_random_buffer16(src2_16, MAX);
    printf( "Constant: %d\n", constant);
    init_const_buffer16(src1_16, constant);
    init_const_buffer16(src2_16, constant);
    ensure_correct_range(src1_16, MAX);
    ensure_correct_range(src2_16, MAX);

    // init 32-bit result buffers
    init_buffer32(naive_res_32, 0); 
    init_buffer32(amx_res_32, 0);

    // Perform naive matmul
    naive_matmul(src1_16, src2_16, naive_res_32, rows, colsb, rows);

    // Perform AMX matmul
    bit_split_amx_matmul_int16_t(src1_16, src2_16, amx_res_32, MAX);

    // mmodular reduction
    modular_reduction(amx_res_32, 257, MAX/4);
    modular_reduction(naive_res_32, 257, MAX/4);

    // print results
    printf("Naive Result:\n");
    print_buffer32(naive_res_32, MAX_ROWS, MAX_ROWS);
    printf("AMX Result:\n");
    print_buffer32(amx_res_32, MAX_ROWS, MAX_ROWS);

    // assert same result as naive
    for (int i = 0; i < MAX/4; i++){    
      assert(amx_res_32[i] == naive_res_32[i]);
    }

    // Release the tile configuration to return to the init state, 
    // which releases all storage it currently holds
    _tile_release ();
  }
}
