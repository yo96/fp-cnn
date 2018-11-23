#define BASE_PER_DDRBUS 16
#define BASE_PER_FBUS   16
#define BASE_PER_FBUS   16
#define BASE_PER_WBUS   BASE_PER_FBUS
#define BASE_PER_OBUS   BASE_PER_WBUS
#define FBUS_PER_DDRBUS 1
#define WBUS_PER_DDRBUS 1

#define TILE_BUF_SIZE   28 * 28 * 2

#define SYS_WID         BASE_PER_FBUS
#define SYS_HT          BASE_PER_OBUS
#define NUM_FIL_BUF     BASE_PER_OBUS
#define FIL_BUF_SIZE    25

/*****************************************************************************
 *  Data types 
 *****************************************************************************/

typedef int   base;
typedef int16 ddr_bus_t;
typedef int16 fmap_bus_t;
typedef int16 out_bus_t;

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
