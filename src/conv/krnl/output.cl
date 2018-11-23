//#include "ap_int.h"
//#include "def_helper.h"
#include "defs.h"
#include "configs.h"
//#define N_NUM_PER_BUS 16

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_out( 
  __global ddr_bus_t* o_fmap,
                 int  fil_size,
                 int  o_size    ) 
{
  out_bus from_pipe;
  out_bus acc;
  ddr_bus to_ddr;
  
  int wr_addr = 0;

  for (int i=0;i< o_size;i+=OBUS_PER_DDRBUS){
    to_ddr.bus_val = 0;
    
    for (int m=0;m<OBUS_PER_DDRBUS;m++){
      // accumulate partial sum
      acc.bus_val = 0;
      for (int j=0;j<fil_size;j++){
        // read an output bus from pipe
        read_pipe_block(pipe_out, &from_pipe.bus_val);
        for (int k=0;k<BASE_PER_OBUS;k++){
          acc.vec[k] += from_pipe.vec[k];
        }
      } // fil_size
      
      // write [acc] into ddr_bus
      for (int k=0;k<BASE_PER_OBUS;k++){
        to_ddr.vec[k+m*BASE_PER_OBUS] = acc.vec[k];
      }

    } // m - OBUS_PER_DDRBUS
    o_fmap[wr_addr] = to_ddr.bus_val;
    wr_addr ++;
 
  } // o_size
  //printf("[load_out]: DONE!\n");
}
