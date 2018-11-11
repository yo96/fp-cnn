#define BUS_N_ELE 32
/******************************************************************************
 * Type definitions 
 ******************************************************************************/
//typedef short bus_t;  
typedef uint16 bus_t;
typedef union {
	bus_t bus_val;
	short ele[BUS_N_ELE] __attribute__((xcl_array_partition(complete,1)));
} short_bus; 

typedef union {
	bus_t bus_val;
	int ele[BUS_N_ELE/2] __attribute__((xcl_array_partition(complete,1)));
	} int_bus;

/*****************************************************************************
 * Helper functions
 *****************************************************************************/

/* [ bus_to_short( in, out) ] allocates the input bus [in] 
 * into a short array [out] */
void bus_to_short(bus_t in, short out[BUS_N_ELE]){
	short_bus tmp;
	tmp.bus_val = in;

	__attribute__((opencl_unroll_hint(BUS_N_ELE)))
	for(int i = 0; i < BUS_N_ELE; i++) {
		out[i] = tmp.ele[i];
	}
}

/* [ bus_to_short( in, out) ] allocates the input bus [in] 
 * into an int array [out] */
void bus_to_int(bus_t in, int out[BUS_N_ELE/2]){
	int_bus tmp;
	tmp.bus_val = in;

	__attribute__((opencl_unroll_hint(BUS_N_ELE/2)))
	for(int i = 0; i < BUS_N_ELE/2; i++) {
		out[i] = tmp.ele[i];
	}
}

/* [ short_to_bus(in) ] combines the 32 shorts in [in] into a 512-bit bus. */
bus_t short_to_bus(short in[BUS_N_ELE]){
	short_bus ret;

	__attribute__((opencl_unroll_hint(BUS_N_ELE)))
	for(int i = 0; i < BUS_N_ELE; i++) {
		ret.ele[i] = in[i];
	}
	return ret.bus_val;
}

/* [ int_to_bus(in) ] combines the 32 ints in [in] into a 512-bit bus. */
bus_t int_to_bus(int in[BUS_N_ELE/2]){
	int_bus ret;
	__attribute__((opencl_unroll_hint(BUS_N_ELE/2)))
	for(int i = 0; i < BUS_N_ELE/2; i++) {
		ret.ele[i] = in[i];
	}
	return ret.bus_val;
}

pipe bus_t data_pipe __attribute__((xcl_reqd_pipe_depth(64)));
/*============================================================================*
 * Kernel definitions
 *============================================================================*/

/*****************************************************************************
 * 
 *****************************************************************************/
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_data(
	__global bus_t* ptr,
						 int  size
	) 
{	
	// int -> bus_t
	int real_size = size/16;
	bus_t to_pipe;


	__attribute__((xcl_pipeline_loop))
	for (int i=0;i<real_size;i++){
		to_pipe = ptr[i];
		write_pipe( data_pipe, &to_pipe);
	}
}

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void dummy( int  size ) 
{
	int real_size = size/4;
	bus_t from_pipe;

	__attribute__((xcl_pipeline_loop))
	for (int i=0;i<real_size;i++){
		read_pipe( data_pipe, &from_pipe);
	}

}