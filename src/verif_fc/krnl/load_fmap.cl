#include "defs.h"
#include "configs.h"

#define BUFFER_SIZE 28 * 28 * 2
#define BLK_DEP BASE_PER_OBUS/BASE_PER_FBUS

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
                 int  pool_wid,
                 int  pool_ht,
                 int  pool_stride,
                 int  lpadding,
                 int  rpadding,
                 int  upadding,
                 int  dpadding,
                 int  conv_stride,
                 int  n_iter    )
{
  bool debug_refill = false;
  bool debug_feed   = false;

  ddr_bus    from_ddr;
  fmap_bus   to_pipe;
  fmap_bus_t tile_buf[BUFFER_SIZE]
  __attribute__((xcl_array_reshape(cyclic, FBUS_PER_DDRBUS, 1)));

  int off_x = lpadding;
  int off_y = upadding;
  int fpd = FBUS_PER_DDRBUS; // can't drectly divide by FBUS_PER_DDRBUS??
  int fpo = FBUS_PER_OBUS;
  
  int pfWid    = (pool_wid - 1) * conv_stride + fil_wid;
  int pfHt     = (pool_ht  - 1) * conv_stride + fil_ht ;
  int pfStride = pool_stride * conv_stride; 
  int X = fmap_wid + lpadding + rpadding - pfWid + 1;
  int Y = fmap_ht  + upadding + dpadding - pfHt  + 1;

  int fmap_wid_ddr_size = (fmap_wid*fpo + fpd-1)/fpd;
  int blk_db_size = fmap_ht * fmap_wid_ddr_size;
  int blk_fb_size = (fmap_ht * fmap_wid * BLK_DEP);

  for (int ni=0; ni<n_iter; ni++){

    // Load whole fmap
    for (int b=0;b<fmap_nblk;b++){
      int blk_off_ddr  = b * blk_db_size;
      int blk_off_tile = b * blk_fb_size;

      for (int y=0;y<fmap_ht;y++){
        // Each row is 64B aligned
        int tileAddr = y*fmap_wid*BLK_DEP + blk_off_tile;
        for (int x=0;x<fmap_wid_ddr_size;x++){

          int y_offset = y*fmap_wid_ddr_size;
          if (debug_refill)
            printf("load_fmap(): reading DDR[%d]\n",x+y_offset+blk_off_ddr);
          from_ddr.bus_val = fmap[x+y_offset+blk_off_ddr];
          
          // Decompese ddr_bus into fmap_bus and write to tile buffer
          fmap_bus to_buf;
          for (int j=0;j<FBUS_PER_DDRBUS;j++){
            for (int k=0;k<BASE_PER_FBUS;k++){
              to_buf.vec[k] = from_ddr.vec[k+j*BASE_PER_FBUS];
            } // k < BASE_PER_FBUS
            if (debug_refill){
              printf("load_fmap(): REFILL y=%d, x=%d, b=%d, ni=%d\n", y, x, b, ni);
              printf("load_fmap(): wrtiing tile_buf[%d] with\n    ", tileAddr+j);
              for (int di=0;di<BASE_PER_FBUS;di++)
                printf(" %d,", to_buf.vec[di]);
              printf("\n");
            } 
            tile_buf[tileAddr+j] = to_buf.bus_val;
          } // FBUS_PER_DDRBUS
          tileAddr += FBUS_PER_DDRBUS;
        } // x < fmap_wid_ddr_size
      } // y < fmap_ht
    } // b < fmap_nblk

    // Feed the pipe
    for (int y=0;y<Y;y+=pfStride){
      for (int x=0;x<X;x+=pfStride){

        // iterate over a pooling window's perceptive field
        if (debug_feed){
          printf("load_fmap(): Iterating perceptive field at (%d, %d)\n", y, x);
        }
        for (int pfY=0;pfY<=pfHt-fil_ht;pfY+=conv_stride){
          for (int pfX=0;pfX<=pfWid-fil_wid;pfX+=conv_stride){

            // iterate over a filter window
            if(debug_feed)
              printf("load_fmap(): Iterating filter window (%d,%d)\n", pfY, pfX);
            for (int nb=0;nb<fmap_nblk;nb++){
              int blk_offset = nb * blk_fb_size;   
              for (int r=0;r<fil_ht;r++){
                for (int c=0;c<fil_wid;c++){
                  for (int d=0;d<BLK_DEP;d++){
                    int x_addr = x - off_x + c + pfX;
                    int y_addr = y - off_y + r + pfY;
                    
                    if (x_addr<0 || x_addr>fmap_wid-1 || y_addr<0 || y_addr>fmap_ht-1){
                      to_pipe.bus_val = 0; //padding
                      //printf("[load](%d,%d)%d: feeding 0-padding to pipe\n",x,y,c+r*fil_wid);              
                    }
                    else {
                      int rd_addr = d + x_addr*BLK_DEP + y_addr*fmap_wid*BLK_DEP + blk_offset;
                      to_pipe.bus_val = tile_buf[rd_addr];
                      //printf("[load](%d,%d)%d: feeding fmap[%d] to pipe\n",x,y,c+r*fil_wid,rd_addr);
                    }
                    if (debug_feed){
                      printf("load_fmap(): y=%d, x=%d, d=%d\n", y, x, d);
                      printf("load_fmap(): x_addr=%d, y_addr=%d\n", x_addr, y_addr);
                      printf("load_fmap(): reading from tile_buf[%d]\n", d + x_addr*BLK_DEP + y_addr*fmap_wid*BLK_DEP + blk_offset);
                      printf("load_fmap(): feeding pipe with\n    ");
                      for (int i=0;i<BASE_PER_FBUS;i++){
                        printf("%d, ",to_pipe.vec[i]);
                      }
                      printf("\n");
                    }
                    write_pipe_block(pipe_fmap, &to_pipe.bus_val);
                  } // d<BLK_DEP
                } // fil_wid
              } // fil_ht
            } // nb < fmap_nblk
            if (debug_feed) printf("\n");
          
          } // pfX < pfWid
        } // pfY < pfHt
        if (debug_feed){
          printf("==================================================\n");
        }        
      } // X 
    } // Y

  } // n_iter
  if (debug_feed) printf("load_fmap(): DONE!!!!!\n");
} // load_famp