#define BUS_N_ELE 32
/******************************************************************************
 * Type definitions 
 ******************************************************************************/
//typedef short bus_t;  
typedef uint16 bus_t;
typedef union {
  bus_t bus_val;
  short ele[BUS_N_ELE]; 
  //__attribute__((xcl_array_partition(complete,1)));
} short_bus; 

typedef union {
  bus_t bus_val;
  int ele[BUS_N_ELE/2]; 
  //__attribute__((xcl_array_partition(complete,1)));
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

/* [ int_to_bus(in) ] combines the 16 ints in [in] into a 512-bit bus. */
bus_t int_to_bus(int in[BUS_N_ELE/2]){
  int_bus ret;
  __attribute__((opencl_unroll_hint(BUS_N_ELE/2)))
  for(int i = 0; i < BUS_N_ELE/2; i++) {
    ret.ele[i] = in[i];
  }
  return ret.bus_val;
}

/*============================================================================ 
 *  Pipe definitions
 *============================================================================*/
pipe bus_t pipe_in0 __attribute__((xcl_reqd_pipe_depth(16)));
pipe bus_t pipe_in1 __attribute__((xcl_reqd_pipe_depth(16)));
pipe bus_t pipe_out __attribute__((xcl_reqd_pipe_depth(16)));