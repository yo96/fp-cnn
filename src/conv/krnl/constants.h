#define BASE_PER_DDRBUS 32
#define BASE_PER_FBUS   16
#define BASE_PER_WBUS   8
#define FBUS_PER_DDRBUS 2
#define WBUS_PER_DDRBUS 4

#define TILE_BUF_SIZE   28 * 28

#define SYS_WID         BASE_PER_FBUS
#define SYS_HT          BASE_PER_WBUS
#define NUM_FIL_BUF     BASE_PER_WBUS
#define FIL_BUF_SIZE    25

/*****************************************************************************
 *  Data types 
 *****************************************************************************/

typedef short   base;
typedef int16   ddr_bus_t;
typedef short16 fmap_bus_t;
typedef short8  wts_bus_t;

typedef union {
  base    vec[BASE_PER_DDRBUS];
  ddr_bus_t bus_val; 
} ddr_bus;

typedef union {
  base     vec[BASE_PER_FBUS];
  fmap_bus_t bus_val;
} fmap_bus;

typedef union {
  base    vec[BASE_PER_WBUS];
  wts_bus_t bus_val;
} wts_bus;

typedef wts_bus out_bus;

