#include "defs.h"
//#define N_NUM_PER_BUS 16

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_out( 
                 int  o_size    )
{
  fmap_bus from_pipe;

  /*printf( "load_out(): o_size = %d\n", o_size );*/

  for ( int i = 0 ; i < o_size; i++ ) {
    read_pipe_block( pipe_fmap, &from_pipe.bus_val );
    /*printf( "load_out(): read from pipe successfully\n", o_size );*/
    /*for ( int j = 0; j < BASE_PER_FBUS; j++ )*/
      /*printf( "%d ", from_pipe.vec[ j ] );*/
    /*printf( "\n" );*/
  }

  /*printf( "load_out(): DONE!\n" );*/
}
