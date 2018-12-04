#include "defs.h"
#include "configs.h"

#define WB_BUF_SIZE (4096/sizeof(ddr_bus_t))
#define DEBUG_POOL false

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void pool_wb(
  __global ddr_bus_t* o_fmap,
                 int  pool_size,
                 int  o_wid,
                 int  o_size,
                 int  n_trans ) 
{
  ddr_bus_t wb_buf[WB_BUF_SIZE];
  ddr_bus   to_buf; 

  int ddrAddr = 0;
  int bufAddr = 0;
  int OPD     = OBUS_PER_DDRBUS;
  //int widDSize = (o_wid+OPD-1) / OPD;
  int numTrans = n_trans; // o_nblk * o_ht * widDSize; 

  for (int i=0;i<o_size;i+=o_wid){
    
    int xLeft = o_wid;
    for (int x=0;x<o_wid;x+=OBUS_PER_DDRBUS){

      int numOBUS = OBUS_PER_DDRBUS < xLeft ? OBUS_PER_DDRBUS : xLeft;
      to_buf.bus_val = 0;
      for (int opd=0;opd<numOBUS;opd++){

        // perform max pooling
        if( DEBUG_POOL )
          printf("pooling(): reading from pipe... i=%d, x=%d, opd=%d\n", i, x, opd);
        out_bus pool_bus;
        read_pipe_block(pipe_relu, &pool_bus.bus_val);
        for (int p=1;p<pool_size;p++){
          if ( DEBUG_POOL )
            printf("pooling(): performing max pooling... reading %d-th out_bus from pipe\n", p+1);
          out_bus tmp;
          read_pipe_block(pipe_relu, &tmp.bus_val);
          for (int j=0;j<BASE_PER_OBUS;j++){
            pool_bus.vec[j] = tmp.vec[j] > pool_bus.vec[j] ? 
                              tmp.vec[j] : pool_bus.vec[j];
          }
        } // p<pool_size-1

        // write out_bus into ddr_bus
        if ( DEBUG_POOL )
          printf("pooling(): writing pooling result into ddr_bus... \n");
        for (int k=0;k<BASE_PER_OBUS;k++)
          to_buf.vec[k+opd*BASE_PER_OBUS] = pool_bus.vec[k];

      } // opd < OBUS_PER_DDRBUS
      xLeft -= numOBUS;
      
      // write ddr_bus into [wb_buf]
      if (DEBUG_POOL){
        printf("pooling(): i=%d, x=%d\n", i, x);
        printf("pooling(): writing wb_buf[%d] with\n", bufAddr);
        for (int di=0;di<BASE_PER_DDRBUS;di++)
          printf(" %d", to_buf.vec[di]);
        printf("\n");
      }
      wb_buf[bufAddr] = to_buf.bus_val;
      bufAddr ++;

      // burst write to ddr if buffer is full or no data left
      if (bufAddr==WB_BUF_SIZE || bufAddr == numTrans){
        int wrLen = numTrans < WB_BUF_SIZE ? numTrans : WB_BUF_SIZE;
        if(DEBUG_POOL)
          printf("pooling(): busrt writing %d ddr_bus into DDR\n",wrLen);
        
        for (int j=0;j<wrLen;j++) 
          o_fmap[ddrAddr+j] = wb_buf[j];
        numTrans -= wrLen;
        ddrAddr  += WB_BUF_SIZE;  
        bufAddr   = 0;
      }
    } // x < o_wid
  } // i < o_size
  if(DEBUG_POOL) printf("pooling(): DONE\n");
}//pool_wb
