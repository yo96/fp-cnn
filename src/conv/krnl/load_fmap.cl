//#include "ap_int.h"
//#include "def_helper.h"
#include "defs.h"

#define BUFFER_SIZE 28 * 28 * 2
#define TWO 2
/******************************************************************************
 * load_fmap
 ******************************************************************************
  This is a functional kernel just to test if the conv kernel functions 
  correctly. */

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_fmap(
  __global ddr_bus_t* fmap,
                 int  fmap_wid,
                 int  fmap_ht,
                 int  fmap_nblk,
                 int  fil_wid,
                 int  fil_ht,
                 int  padding,
                 int  stride,
                 int  n_iter    )
{
  bool debug_refill = false;
  bool debug_feed   = false;

  ddr_bus    from_ddr;
  fmap_bus   to_pipe;
  fmap_bus_t tile_buf[BUFFER_SIZE]
  __attribute__((xcl_array_reshape(cyclic, FBUS_PER_DDRBUS, 1)));

  int X = fmap_wid + padding*2 - fil_wid + 1;
  int Y = fmap_ht  + padding*2 - fil_ht  + 1;
  int off_x = (fil_wid-1) / 2;
  int off_y = (fil_ht -1) / 2;
  int fpd = FBUS_PER_DDRBUS; // can't drectly divide by FBUS_PER_DDRBUS??
  int blk_db_size = (fmap_ht * fmap_wid)/ fpd;
  int blk_fb_size = (fmap_ht * fmap_wid);

  for (int ni=0; ni<n_iter; ni++){

    // Load whole fmap
    for (int b=0;b<fmap_nblk;b++){
      int off_ddr  = b * blk_db_size;
      int off_tile = b * blk_fb_size;

      for (int i=0;i<blk_db_size;i++){        
        // read a ddr_bus from DDR
        fmap_bus to_buf;
        from_ddr.bus_val = fmap[i+off_ddr];          
        
        // decompose ddr_bus into fmap_bus and write to tile buffer
        for (int j=0;j<FBUS_PER_DDRBUS;j++){
          for (int k=0;k<BASE_PER_FBUS;k++){
            to_buf.vec[k] = from_ddr.vec[k+j*BASE_PER_FBUS];
          } // k < BASE_PER_FBUS
          int tb_addr = j+i*FBUS_PER_DDRBUS+off_tile;
          if (debug_refill){
            printf("tile_buf[%d]<- ", tb_addr);
            for (int d=0;d<BASE_PER_FBUS;d++){
              printf("%d, ",to_buf.vec[d]);
            }
            printf("\n");
          }
          tile_buf[tb_addr] = to_buf.bus_val;
        } // j < FBUS_PER_DDRBUS

      } // i < blk_db_Size
    } // b < fmap_nblk

    // Feed the pipe
    for (int y=0;y<Y;y+=stride){
      for (int x=0;x<X;x+=stride){

        for (int nb=0;nb<fmap_nblk;nb++){
          int blk_offset = nb * blk_fb_size;   
          for (int r=0;r<fil_ht;r++){
            for (int c=0;c<fil_wid;c++){
              int x_addr = x - off_x + c;
              int y_addr = y - off_y + r;
              
              if (x_addr<0 || x_addr>fmap_wid-1 || y_addr<0 || y_addr>fmap_ht-1){
                to_pipe.bus_val = 0; //padding
                //printf("[load](%d,%d)%d: feeding 0-padding to pipe\n",x,y,c+r*fil_wid);              
              }
              else {
                int rd_addr = x_addr + y_addr*fmap_wid + blk_offset;
                to_pipe.bus_val = tile_buf[rd_addr];
                //printf("[load](%d,%d)%d: feeding fmap[%d] to pipe\n",x,y,c+r*fil_wid,rd_addr);
              }
              if (debug_feed){
                printf("%d (%d,%d,%d): ",x_addr + y_addr*fmap_wid + blk_offset, y_addr, x_addr, nb);
                for (int i=0;i<BASE_PER_FBUS;i++){
                  printf("%d, ",to_pipe.vec[i]);
                }
                printf("\n");
              }
              write_pipe_block(pipe_fmap, &to_pipe.bus_val);
            } // fil_wid
          } // fil_ht
        } // nb < fmap_nblk
        if (debug_feed) printf("\n");
      } // X 
    } // Y

  } // n_iter
  printf("[load_fmap]: DONE\n");
}