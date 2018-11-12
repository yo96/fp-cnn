#include "def_helper.h"
/*============================================================================*
 * Kernel definitions
 *============================================================================*/
# define BUFFER_SIZE (4096/64) // 4KB per burst transaction
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_data(
	__global bus_t* ptr_a,
	__global bus_t* ptr_b,
						 int  size
	) 
{	
	// int -> bus_t
	int real_size = size >> 4;
	bus_t to_pipe;

	bus_t buf_a[BUFFER_SIZE];
	bus_t buf_b[BUFFER_SIZE];

	__attribute__((xcl_pipeline_loop))
	for (int i=0;i<real_size;i+=BUFFER_SIZE){
		// burst read a
		for (int j=0;j<BUFFER_SIZE;j++){
			buf_a[j] = ptr_a[i+j];
		}

		// burst read b
		for (int j=0;j<BUFFER_SIZE;j++){
			buf_b[j] = ptr_b[i+j];
		}

		// feed pipes
		for (int j=0;j<BUFFER_SIZE;j++){
			write_pipe_block(pipe_in0, &buf_a[j]);
			write_pipe_block(pipe_in1, &buf_b[j]);
		}

	}
}
