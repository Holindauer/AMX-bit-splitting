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
static void matmul(uint16_t *A, uint16_t *B, uint32_t *c, int A_rows, int A_cols, int B_cols)
{
  for (int i = 0; i < A_rows; i++) {
    for (int j = 0; j < B_cols; j++) {

      // zero if not already initialized
      c[i * B_cols + j] = 0;

      for (int k = 0; k < A_cols; k++){
        // zero extend mul args from 16->32
        c[i * B_cols + j] += (uint32_t)A[i * A_cols + k] * (uint32_t)B[k * B_cols + j];
      }
    }
  }
} 


int main(){

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
  uint16_t src1_16[MAX];
  uint16_t src2_16[MAX];
  uint32_t amx_res_32[MAX/4];
  uint32_t naive_res_32[MAX/4];

  // Init random int16_t buffers
  init_random_buffer16(src1_16, MAX);
  init_random_buffer16(src2_16, MAX);
  ensure_correct_range(src1_16, MAX);
  ensure_correct_range(src2_16, MAX);

  
  // Print input buffers
  printf("Input Buffer 1:\n");
  print_buffer16(src1_16, rows, colsb);
  printf("Input Buffer 2:\n");
  print_buffer16(src2_16, rows, colsb);

  // init 32-bit result buffers
  init_buffer32(naive_res_32, 0); 
  init_buffer32(amx_res_32, 0);

  // Perform naive matmul
  matmul(src1_16, src2_16, naive_res_32, rows, colsb, colsb);

  // Print naive result
  printf("Naive Result:\n");
  print_buffer32(naive_res_32, rows, colsb/4);


  //  // Load tile rows from memory
  //  _tile_loadd (2, src1, STRIDE);
  //  _tile_loadd (3, src2, STRIDE);
  //  _tile_loadd (1, res, STRIDE);

  //  // Compute dot-product of bytes in tiles 
  //  _tile_dpbuud (1, 2, 3);

  //  // Store the tile data to memory
  //  _tile_stored (1, res, STRIDE);
  //  print_buffer32(res, rows, colsb/4);

   // Release the tile configuration to return to the init state, 
   // which releases all storage it currently holds
   _tile_release ();
}
