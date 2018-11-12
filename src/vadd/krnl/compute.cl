#include "def_helper.h"

#define N_NUM_PER_BUS 16

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void compute( int  size ) 
{
  int real_size = size >> 4;
  bus_t from_pipe0;
  bus_t from_pipe1;
  bus_t to_pipe;
  int reg_int0[N_NUM_PER_BUS];
  int reg_int1[N_NUM_PER_BUS];
  int reg_out [N_NUM_PER_BUS];

  __attribute__((xcl_pipeline_loop))
  for (int i=0;i<real_size;i++){
    // read data from pipes
    read_pipe_block(pipe_in0, &from_pipe0);
    read_pipe_block(pipe_in1, &from_pipe1);

    // compute vvadd;
    bus_to_int(from_pipe0,reg_int0);
    bus_to_int(from_pipe1,reg_int1);

    __attribute__((opencl_unroll_hint))
    for (int j=0;j<N_NUM_PER_BUS;j++)  
      reg_out[j] = reg_int0[j] + reg_int1[j];

    to_pipe = int_to_bus(reg_out);
    write_pipe_block(pipe_out, &to_pipe);
  }

}