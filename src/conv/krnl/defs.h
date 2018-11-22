#define BASE_PER_DDRBUS 32
#define BASE_PER_FBUS   16
#define BASE_PER_WBUS   BASE_PER_FBUS
#define BASE_PER_OBUS   BASE_PER_FBUS

#define FBUS_PER_DDRBUS BASE_PER_DDRBUS/BASE_PER_FBUS
#define WBUS_PER_DDRBUS FBUS_PER_DDRBUS
#define OBUS_PER_DDRBUS FBUS_PER_DDRBUS

#define TILE_BUF_SIZE   28 * 28

#define NUM_FIL_BUF     BASE_PER_OBUS
#define SYS_WID         BASE_PER_FBUS
#define SYS_HT          BASE_PER_OBUS
#define FIL_BUF_SIZE    50

/*****************************************************************************
 *  Data types 
 *****************************************************************************/

typedef int16   ddr_bus_t;
typedef short   base;
typedef short16 fmap_bus_t;
typedef short16 out_bus_t;

typedef union {
  base    vec[BASE_PER_DDRBUS];
  ddr_bus_t bus_val; 
} ddr_bus;

typedef union {
  base     vec[BASE_PER_FBUS];
  fmap_bus_t bus_val;
} fmap_bus;

typedef union {
  base    vec[BASE_PER_OBUS];
  out_bus_t bus_val;
} out_bus;

typedef fmap_bus   wts_bus;
typedef fmap_bus_t wts_bus_t;
/*****************************************************************************
 * Pipes
 *****************************************************************************/
pipe fmap_bus_t pipe_fmap __attribute__((xcl_reqd_pipe_depth(16)));
pipe ddr_bus_t  pipe_wts  __attribute__((xcl_reqd_pipe_depth(16)));
pipe out_bus_t  pipe_out  __attribute__((xcl_reqd_pipe_depth(16)));