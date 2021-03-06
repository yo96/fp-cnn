#include "defs.h"
#include "configs.h"

#define DEBUG_RELU false

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void acc_relu( int fil_size, int o_size ) 
{
  out_bus from_pipe;
  out_bus acc;
  if ( DEBUG_RELU ) printf("acc_relu(): fil_size=%d, o_size=%d", fil_size, o_size);

  for (int i=0;i< o_size;i++){
    
    // accumulate partial sum
    if ( DEBUG_RELU )
      printf("\nacc_relu(): Iterating over a filter window...\n");
    acc.bus_val = 0;
    for (int j=0;j<fil_size;j++){
      // read an output bus from pipe
      if ( DEBUG_RELU )
        printf("acc_relu(): reading from pipe... i=%d, j=%d\n", i, j);
      read_pipe_block(pipe_out, &from_pipe.bus_val);
      for (int k=0;k<BASE_PER_OBUS;k++){
        acc.vec[k] += from_pipe.vec[k];
      }
    } // fil_size
      
    // perform relu
    for (int k=0;k<BASE_PER_OBUS;k++)
      acc.vec[k] = acc.vec[k]>0 ? acc.vec[k] : 0;
    
    // write to pile 
    write_pipe_block(pipe_relu, &acc.bus_val);
  
  } // o_size
  //printf("acc_relu(): DONE!\n");
}
