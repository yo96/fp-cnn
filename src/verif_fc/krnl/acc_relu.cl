#include "defs.h"
#include "configs.h"

#define DEBUG_RELU false

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void acc_relu( int fil_size, int o_size, int use_relu ) 
{
  out_bus from_pipe;
  int acc[BASE_PER_OBUS];

  if ( DEBUG_RELU ) printf("acc_relu(): fil_size=%d, o_size=%d", fil_size, o_size);

  for (int i=0;i< o_size;i++){
    
    // accumulate partial sum
    if ( DEBUG_RELU )
      printf("\nacc_relu(): Iterating over a filter window...\n");
    //acc = 0;
    for (int k=0;k<BASE_PER_OBUS;k++)
      acc[k] = 0;
    
    for (int j=0;j<fil_size;j++){
      // read an output bus from pipe
      if ( DEBUG_RELU )
        printf("acc_relu(): reading from pipe... i=%d, j=%d\n", i, j);
      read_pipe_block(pipe_out, &from_pipe.bus_val);
      for (int k=0;k<BASE_PER_OBUS;k++){
        acc[k] += from_pipe.vec[k];
      }
    } // fil_size
    
    // perform shift
    for (int k=0;k<BASE_PER_OBUS;k++){
      acc[k] = acc[k] >> FLOAT_NBITS;
    }

    // perform relu
    if (use_relu) {  
      for (int k=0;k<BASE_PER_OBUS;k++)
        acc[k] = acc[k]>0 ? acc[k] : 0;
    }
    
    out_bus to_pipe;
    for (int k=0;k<BASE_PER_OBUS;k++)
      to_pipe.vec[k] = acc[k];
    // write to pile 
    write_pipe_block(pipe_relu, &to_pipe.bus_val);
  
  } // o_size
  //printf("acc_relu(): DONE!\n");
}
