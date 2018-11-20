//#include "ap_int.h"
//#include "def_helper.h"
#include "defs.h"
//#define N_NUM_PER_BUS 16

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_out( 
  __global out_bus_t* o_fmap,
                 int  fil_size,
                 int  o_size    ) 
{
  out_bus from_pipe;
  out_bus to_ddr;

  for (int i=0;i< o_size;i++){
    to_ddr.bus_val = 0;
    for (int j=0;j<fil_size;j++){
      // read an output bus from pipe
      read_pipe_block(pipe_out, &from_pipe.bus_val);
      // accumulate partial sum
      for (int k=0;k<BASE_PER_OBUS;k++){
        to_ddr.vec[k] += from_pipe.vec[k];
      }
    } // fil_size
    o_fmap[i] = to_ddr.bus_val;
  } // o_size

}
