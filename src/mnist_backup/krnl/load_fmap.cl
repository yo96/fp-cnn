#include "defs.h"

#define DEBUG_LOAD_FMAP 0

/******************************************************************************
 ** load_fmap
 ******************************************************************************
 ** Load the feature map data from the DRAM into local tile buffer. */

__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_fmap(
  __global ddr_bus_t* fmap,       // the DRAM to read data from
                 int  fmap_wid,   // width of feature maps
                 int  fmap_ht,    // height of feature maps
                 int  fmap_nblk,  // number of blocks in feature maps
                 int  fil_wid,    // width of filters
                 int  fil_ht,     // height of filters
                 int  tile_wid,   // the width of each padded tile
                 int  tile_ht,    // the height of each padded tile
                 int  lpadding,   // number of padding pixels in the left
                 int  rpadding,   // number of padding pixels in the right
                 int  upadding,   // number of padding pixels in the upside
                 int  dpadding,   // number of padding pixels in the downside
                 int  pool_wid,   // width of pooling 
                 int  pool_ht,    // height of pooling
                 int  pool_stride,// the stride of pooling windows
                 int  stride,     // the stride of convolution
                 int  n_iter    )
{
  if ( DEBUG_LOAD_FMAP ) {
    printf(
      "load_fmap() args: f_wid:%d, f_ht:%d, f_nblk:%d, fil_wid:%d, fli_ht:%d\n",
      fmap_wid, fmap_ht, fmap_nblk, fil_wid, fil_ht
    );
    printf(
      "load_fmap() args: tile_wid:%d, tile_ht:%d\n",
      tile_wid, tile_ht
    );
    printf(
      "load_fmap() args: lpad:%d, rpad:%d, upad:%d, dpad:%d, pool_wid:%d, pool_ht:%d\n",
      lpadding, rpadding, upadding, dpadding, pool_wid, pool_ht
    );
    printf(
      "load_fmap() args: pool_stride:%d, stride:%d, n_iter:%d\n",
      pool_stride, stride, n_iter
    );
  }

  fmap_bus to_pipe;

  ddr_bus tile_buffer[ TILE_BUF_SIZE ];

  ddr_bus from_ddr;

  // SDAccel 2017 bug?
  int FPD = FBUS_PER_DDRBUS;
  int OPD = OBUS_PER_DDRBUS;

  // the width and height of the perception field
  int pfW = ( pool_wid - 1 ) * stride + fil_wid;
  int pfH = ( pool_ht  - 1 ) * stride + fil_ht;

  int tH = fmap_ht  + upadding + dpadding - fil_ht  + 1;
  int tHInc = tile_ht  - pfH + pool_stride * stride;
  int tW = fmap_wid + lpadding + rpadding - fil_wid + 1;
  int tWInc = tile_wid - pfW + pool_stride * stride;

  for ( int nIter = 0; nIter < n_iter; nIter++ ) {
    // ( tileH, tileW ) marks the upper-left most pixel in a padded tile
    for ( int tileH = 0; tileH < tH; tileH += tHInc ) {
      for ( int tileW = 0; tileW < tW; tileW += tWInc ) {
        // load the ( tileH, tileW ) tile step by step
        // what if the DRAM read gets more data than a line?
        // Solution: just write all the data into the buffer and the
        // extra data will be overwritten at the next buffer write. Make 
        // sure the buffer is big enough so the last write is not out of
        // range. 
        int actual_tile_ht = ( tileH+tHInc >= tH ) ? fmap_ht+upadding+dpadding-tileH : tile_ht;
        int actual_tile_wid = ( tileW+tWInc >= tW ) ? fmap_wid+lpadding+rpadding-tileW : tile_wid;

        int condOffsetH_U = ( tileH == 0 ) ? upadding : 0;
        int condOffsetH_D = ( tileH+actual_tile_ht > fmap_ht+upadding ) ? dpadding : 0;
        int condOffsetW_L = ( tileW == 0 ) ? lpadding : 0;
        int condOffsetW_R = ( tileW+actual_tile_wid > fmap_wid+lpadding ) ? rpadding : 0;
        // calculate the CORRECT value of tileDataH and tileDataW
        int tileDataH = actual_tile_ht  - condOffsetH_U - condOffsetH_D;
        int tileDataW =
          ((actual_tile_wid-condOffsetW_L-condOffsetW_R)+OBUS_PER_DDRBUS-1) / OPD;
        if ( DEBUG_LOAD_FMAP ) {
          printf("load_fmap(): For tile at ( %d, %d )\n", tileH, tileW );
          printf("load_fmap(): Trying to read from DDR into tile buffer...\n");
        }
        for ( int nBlk = 0; nBlk < fmap_nblk; nBlk++ ) {
          // the coordinate ( inTileX, inTileY ) inside one tile
          for ( int inTileY = 0; inTileY < actual_tile_ht; inTileY++ ) {
            int inTileXInc = 0;
            for ( int inTileX = 0; inTileX < actual_tile_wid; inTileX += inTileXInc ) {
              // the actual coordinate ( fmapX, fmapY ) of the feature maps
              int fmapY = tileH + inTileY - upadding;
              int fmapX = tileW + inTileX - lpadding;
              int ddrAddr = 
                ( nBlk * fmap_wid * fmap_ht )*FBUS_PER_OBUS / FPD;

              if ( DEBUG_LOAD_FMAP ) {
                printf("load_fmap(): RD: In-tile coordinate: ( %d, %d )\n",
                    inTileY, inTileX );
                printf("load_fmap(): RD: Fmap coordinate: ( %d, %d )\n",
                    fmapY, fmapX );
              }

              if ( fmapX >= 0 && fmapX < fmap_wid && fmapY >= 0 && fmapY < fmap_ht ) {
                // unpadded data from feature maps
                // offset relative to the start address of the feature maps
                int offset = (( fmapY*fmap_wid+fmapX )*FBUS_PER_OBUS+FPD-1 ) / FPD;
                // the coordinate of this piece of data in the tile buffer
                int tileDataY = inTileY - condOffsetH_U;
                int tileDataX = (inTileX - condOffsetW_L) / OPD;
                int tileAddr = nBlk * tileDataH * tileDataW + tileDataY * tileDataW + tileDataX;
                /*int tileAddr = nBlk * tileDataH * tileDataW + tileDataY * tileDataW + tileDataX;*/
                // OPD because each block has the same depth as OBUS
                // FPD/FPO = OPD
                inTileXInc = OBUS_PER_DDRBUS;

                from_ddr.bus_val = fmap[ ddrAddr + offset ];
                tile_buffer[ tileAddr ].bus_val = from_ddr.bus_val;
                /*tile_buffer[ tileAddr ].bus_val = from_ddr.bus_val;*/

                if ( DEBUG_LOAD_FMAP ) {
                  printf("load_fmap(): RD from DDR at [%d] ( addr=%d, offset=%d )\n",
                      ddrAddr+offset, ddrAddr, offset );
                  printf("\ttile_buf[%d] written, base = %d, offset = %d, tileDataY = %d, tileDataX = %d, tileDataH = %d, tileDataW = %d\n", 
                      tileAddr, tileAddr, offset,
                      tileDataY, tileDataX, tileDataH, tileDataW);
                }
              } else {
                // padding data
                inTileXInc = 1;
              }
            }
          }
        }
        if ( DEBUG_LOAD_FMAP ) {
          for ( int i = 0; i < TILE_BUF_SIZE; i++ )  {
            printf("load_fmap(): tile_buf[%d]=",i);
            int sum = 0;
            for ( int j = 0; j < BASE_PER_DDRBUS; j++ ) {
              printf("%d ", tile_buffer[i].vec[j]);
              sum += tile_buffer[i].vec[j];
            }
            printf("sum = %d \n", sum);
            printf("\n");
          }
          printf("load_fmap(): Trying to feed data into pipe_fmap...\n");
        }
        // feed data into the pipe
        for ( int pfY = 0; pfY < actual_tile_ht-pfH+1; pfY += pool_stride*stride ) {
          for ( int pfX = 0; pfX < actual_tile_wid-pfW+1; pfX += pool_stride*stride ) {
            if ( DEBUG_LOAD_FMAP )
              printf("load_fmap(): pfY = %d, pfX = %d\n", pfY, pfX);
            for ( int pwY = 0; pwY < pool_ht; pwY++ ) {
              for ( int pwX = 0; pwX < pool_wid; pwX++ ) {
                for ( int nBlk = 0; nBlk < fmap_nblk; nBlk++ ) {
                  if ( DEBUG_LOAD_FMAP )
                    printf("==================================\n");
                  for ( int filY = 0; filY < fil_ht; filY++ ) {
                    for ( int filX = 0; filX < fil_wid; filX++ ) {
                      // a new dimension: FBUS_PER_OBUS
                      // follow channel-major way to write data
                      for ( int fpo = 0; fpo < FBUS_PER_OBUS; fpo++ ) {
                        // the coordinate inside the current tile
                        int inTileY = pfY + pwY * stride + filY;
                        int inTileX = pfX + pwX * stride + filX;
                        // the actual coordinate ( fmapX, fmapY ) of the feature maps
                        int fmapY = tileH + inTileY - upadding;
                        int fmapX = tileW + inTileX - lpadding;

                        if ( DEBUG_LOAD_FMAP ) {
                          printf("load_fmap(): FD: In-tile coordinate: ( %d, %d )\n",
                              inTileY, inTileX );
                          printf("load_fmap(): FD: Fmap coordinate: ( %d, %d )\n",
                              fmapY, fmapX );
                        }

                        // only load unpadded data from feature maps
                        if ( fmapX >= 0 && fmapX < fmap_wid && fmapY >= 0 && fmapY < fmap_ht ) {
                          // the coordinate of this piece of data in the tile buffer
                          int tileDataY = inTileY - condOffsetH_U;
                          int tileDataX = (inTileX - condOffsetW_L) / OPD;
                          int whichObus = inTileX-condOffsetW_L - ( tileDataX*OPD );
                          // Bug fix: data from different blocks go to different blocks
                          int tileBaseAddr = nBlk * tileDataH * tileDataW + tileDataY * tileDataW + tileDataX;
                          int tileAddr = tileBaseAddr;

                          int f_tileAddr = tileBaseAddr * FBUS_PER_DDRBUS + whichObus * FBUS_PER_OBUS + fpo;
                          int offset = f_tileAddr - tileAddr*FBUS_PER_DDRBUS;

                          ddr_bus from_tile_buf;
                          from_tile_buf.bus_val =  tile_buffer[ tileAddr ].bus_val;

                          if ( DEBUG_LOAD_FMAP ) {
                            printf("load_fmap(): RD from tile_buf[%d], offset = %d, tileBase = %d, f_tileAddr = %d\n", 
                                tileAddr, offset, tileBaseAddr, f_tileAddr);
                          }

                          for ( int i = 0; i < BASE_PER_FBUS; i++ ) {
                            to_pipe.vec[ i ] = from_tile_buf.vec[ offset*BASE_PER_FBUS + i ];
                          }
                          /*to_pipe.bus_val = tile_buffer[ tileAddr+fpo ].bus_val;*/
                        } else {
                          // feed 0 into the pipe
                          if ( DEBUG_LOAD_FMAP ) {
                            printf("load_fmap(): feed 0 padding to pipe\n" );
                          }
                          to_pipe.bus_val = 0;
                        }
                        if ( DEBUG_LOAD_FMAP ) {
                          printf("load_fmap(): trying to write to pipe\n");
                          printf("load_fmap(): nBlk=%d,inTileY=%d,inTileX=%d\n\t", nBlk,
                              inTileY, inTileX);
                          for ( int i = 0; i < BASE_PER_FBUS; i++ )
                            printf("%d ", to_pipe.vec[i]);
                          printf("\n");
                        }

                        write_pipe_block( pipe_fmap, &to_pipe.bus_val );
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if ( DEBUG_LOAD_FMAP ) {
    printf("load_fmap(): DONE\n");
  }
}
