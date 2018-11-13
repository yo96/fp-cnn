//#include "ap_int.h"
#include "def_helper.h"

#define N_NUM_PER_BUS 16

#define NUM_FIL_BUF   16
#define FIL_BUF_WID   3
#define FIL_BUF_HT    3
#define FIL_BUFF_SIZE 9
#define SYS_WID       16
#define SYS_HT        16
#define SHREG_SIZE    136

typedef int base;

/******************************************************************************
 * [compute] kernel
 *****************************************************************************
  This kernel describe a 16x16 systolic array 
 */
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void compute( 
  int o_wid,
  int o_ht,
  int n_fil,
  int fil_wid,
  int n_iter ) 
{
  bus_t wts_ram[NUM_FIL_BUF][FIL_BUFF_SIZE] 
  __attribute__((xcl_array_partition(complete, 1)));

  base sys[SYS_HT][SYS_WID] 
  __attribute__((xcl_array_partition(complete, 0)));

  base shreg_wts[NUM_FIL_BUF][SHREG_SIZE]  
  __attribute__((xcl_array_partition(complete, 0)));
  
  base shreg_fmap[SHREG_SIZE]
  __attribute__((xcl_array_partition(complete, 1)));

  for (int ni=0;ni<n_iter;ni++){
    // read wts into BRAM


    // feed fmap and wts into shreg

    // 

  } // n_iter

}