//#include "ap_int.h"
#include "def_helper.h"

/******************************************************************************
 * load_fmap
 ******************************************************************************
  This is a functional kernel just to test if the conv kernel functions 
  correctly. */

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_fmap(
  __global bus_t* fmap,
             int  fmap_wid,
             int  fmap_ht,
             int  fmap_dep,
             int  fil_wid,
             int  fil_ht,
             int  padding,
             int  stride,
             int  n_iter    )
{
  bus_t to_pipe;
  int X = fmap_wid + padding*2 - fil_wid + 1;
  int Y = fmap_ht  + padding*2 - fil_ht  + 1;

  for (int ni=0; ni<n_iter; ni++){
    for (int x=0;x<X;x+=stride){
      for (int y=0;y<Y;y+=stride){
        
        for (int c=0;c<fil_wid;c++){
          for (int r=0;r<fil_ht;r++){
            int x_addr = x + c - padding;
            int y_addr = y + r - padding;
            
            if (x_addr<0 || x_addr>fil_wid-1 || y_addr<0 || y_addr>fil_wid-1)
              to_pipe = 0; //padding
            else {
              // Assumes only fetch once in channel direction
              int rd_addr = x_addr + y_addr*fil_wid;
              to_pipe = fmap[rd_addr];
            }
            write_pipe_block(pipe_fmap, &to_pipe);
          } // fil_wid
        } // fil_ht
      } // Y
    } // X
  } // n_iter
  printf("load_fmap: DONE\n");
}