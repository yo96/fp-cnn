#include "def_helper.h"

# define BUFFER_SIZE (64) // 4KB per burst transaction

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void write_back( 
  __global bus_t* ptr_c,
             int  size  ) 
{
  int real_size = size >> 4;
  bus_t from_pipe;
  bus_t buf_c[BUFFER_SIZE];

  for (int i=0;i<real_size;i+=BUFFER_SIZE){

    // fill the 4KB buffer
    for (int j=0;j<BUFFER_SIZE;j++){
      read_pipe_block(pipe_out, &from_pipe);
      buf_c[j] = from_pipe;
    }

    // burst write 
    for (int j=0;j<BUFFER_SIZE;j++)
      ptr_c[i+j] = buf_c[j];
  }

}
