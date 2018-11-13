//#include "ap_int.h"
#include "def_helper.h"

/******************************************************************************
 * load_wts
 ******************************************************************************
  This is a functional kernel just to test if the conv kernel functions 
  correctly. */

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_wts(
  __global bus_t* wts,
             int  wts_size )
{
  for (int i=0;i<wts_size;i++){
    bus_t to_pipe = wts[i];
    write_pipe_block(pipe_wts, &to_pipe);
  }
}