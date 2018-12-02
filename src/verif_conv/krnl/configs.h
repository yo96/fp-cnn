#define BASE_PER_DDRBUS 32
#define BASE_PER_FBUS   16
#define BASE_PER_OBUS   32
#define BASE_PER_WBUS   BASE_PER_FBUS

#define FBUS_PER_DDRBUS BASE_PER_DDRBUS/BASE_PER_FBUS
#define FBUS_PER_OBUS   BASE_PER_OBUS/BASE_PER_FBUS
#define WBUS_PER_DDRBUS FBUS_PER_DDRBUS
#define OBUS_PER_DDRBUS BASE_PER_DDRBUS/BASE_PER_OBUS

#define TILE_BUF_SIZE   28 * 28 * 2

#define NUM_FIL_BUF     BASE_PER_OBUS
#define SYS_WID         BASE_PER_FBUS
#define SYS_HT          BASE_PER_OBUS
#define FIL_BUF_SIZE    50

#define FLOAT_NBITS     8