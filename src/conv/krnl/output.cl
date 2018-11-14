//#include "ap_int.h"
#include "def_helper.h"

#define N_NUM_PER_BUS 16

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_out( 
  __global bus_t* o_fmap,
             int  fil_size,
             int  o_size    ) 
{
  bus_t from_pipe;
  int   from_pipe_int[16];

  bus_t out_bus;
  int   out_reg[16];

  for (int i=0;i< o_size;i++){
    out_bus = 0;
    for (int j=0;j<fil_size;j++){
      read_pipe_block(pipe_out, &from_pipe);
      
      bus_to_int(from_pipe, from_pipe_int);
      bus_to_int(out_bus,   out_reg      );

      for (int k=0;k<N_NUM_PER_BUS;k++){
        out_reg[k] += from_pipe_int[k];
      }
      out_bus = int_to_bus(out_reg);
    } // fil_size
    o_fmap[i] = out_bus;
  } // o_size

}
