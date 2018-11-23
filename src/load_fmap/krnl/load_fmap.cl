#include "defs.h"

#define DEBUG_LOAD_FMAP 1

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
                 int  tile_nblk,  // number of blocks in each padded tile
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
      "load_fmap() args: f_wid:%d, f_ht:%d, f_dep:%d, fil_wid:%d, fli_ht:%d\n",
      fmap_wid, fmap_ht, fmap_nblk, fil_wid, fil_ht
    );
    printf(
      "load_fmap() args: tile_wid:%d, tile_ht:%d, tile_nblk:%d\n",
      tile_wid, tile_ht, tile_nblk
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

  fmap_bus tile_buffer[ TILE_BUF_SIZE ];

  fmap_bus to_tile_buf[ FBUS_PER_DDRBUS ];

  ddr_bus from_ddr;

  // SDAccel 2017 bug?
  int FPD = FBUS_PER_DDRBUS;

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
        int actual_tile_ht = ( tileH+tHInc >= tH ) ? fmap_ht+lpadding+rpadding-tileH : tile_ht;
        int actual_tile_wid = ( tileW+tWInc >= tW ) ? fmap_wid+upadding+dpadding-tileW : tile_wid;

        int condOffsetH_LU = ( tileH == 0 ) ? upadding : 0;
        int condOffsetH_LD = ( tileH+actual_tile_ht > fmap_ht-dpadding ) ? dpadding : 0;
        int condOffsetW_LU = ( tileW == 0 ) ? lpadding : 0;
        int condOffsetW_RU = ( tileW+actual_tile_wid > fmap_wid-rpadding ) ? rpadding : 0;
        int tileDataH = actual_tile_ht  - condOffsetH_LU - condOffsetH_LD;
        int tileDataW = actual_tile_wid - condOffsetW_LU - condOffsetW_RU;
        if ( DEBUG_LOAD_FMAP ) {
          printf("load_fmap(): For tile at ( %d, %d )\n", tileH, tileW );
          printf("load_fmap(): Trying to read from DDR into tile buffer...\n");
        }
        /*int condOffsetH_RD = ( tileH+tHInc >= tH ) ? dpadding : 0;*/
        /*int condOffsetW_RD = ( tileW+tWInc >= tW ) ? rpadding : 0;*/
        for ( int nBlk = 0; nBlk < fmap_nblk; nBlk++ ) {
          // the coordinate ( inTileX, inTileY ) inside one tile
          for ( int inTileY = 0; inTileY < actual_tile_ht; inTileY++ ) {
            int inTileXInc = 0;
            for ( int inTileX = 0; inTileX < actual_tile_wid; inTileX += inTileXInc ) {
              // the actual coordinate ( fmapX, fmapY ) of the feature maps
              int fmapY = tileH + inTileY - upadding;
              int fmapX = tileW + inTileX - lpadding;
              int ddrAddr = 
                ( nBlk * fmap_wid * fmap_ht ) / FPD;

              if ( DEBUG_LOAD_FMAP ) {
                printf("load_fmap(): RD: In-tile coordinate: ( %d, %d )\n",
                    inTileY, inTileX );

                printf("load_fmap(): RD: Fmap coordinate: ( %d, %d )\n",
                    fmapY, fmapX );
              }

              if ( fmapX >= 0 && fmapX < fmap_wid && fmapY >= 0 && fmapY < fmap_ht ) {
                // unpadded data from feature maps
                // offset relative to the start address of the feature maps
                int offset = ( fmapY*fmap_wid+fmapX+FPD-1 ) / FPD;
                /*int offset = ( fmapY*fmap_wid+fmapX ) / FPD;*/
                // the coordinate of this piece of data in the tile buffer
                int tileDataY = inTileY - condOffsetH_LU;
                int tileDataX = inTileX - condOffsetW_LU;
                int tileAddr = nBlk * tileDataH * tileDataW + tileDataY * tileDataW + tileDataX;
                // OPD because each block has the same depth as OBUS
                // FPD/FPO = OPD
                inTileXInc = OBUS_PER_DDRBUS;

                if ( DEBUG_LOAD_FMAP ) {
                  printf("load_fmap(): RD from DDR at [%d] ( addr=%d, offset=%d )\n",
                      ddrAddr+offset, ddrAddr, offset );
                }

                from_ddr.bus_val = fmap[ ddrAddr + offset ];

                if ( DEBUG_LOAD_FMAP ) {
                  printf("load_fmap(): \t DDR[%d] read successfully\n",
                      ddrAddr+offset, ddrAddr, offset );

                  printf("load_fmap(): Writing data to tile buffer\n");
                }
                // have FPD pieces of tileData
                for ( int i = 0; i < FBUS_PER_DDRBUS; i++ ) {
                  for ( int j = 0; j < BASE_PER_FBUS; j++ ) {
                    to_tile_buf[ i ].vec[j]
                      = from_ddr.vec[ i*BASE_PER_FBUS+j ];
                  }
                  /*tile_buffer[ tileAddr+i*FBUS_PER_DDRBUS+nBlk ].bus_val*/
                    /*= to_tile_buf[ i ].bus_val;*/
                  tile_buffer[ tileAddr + i ].bus_val = 
                    to_tile_buf[ i ].bus_val;
                  printf("tile_buf[%d] written, tileDataY = %d, tileDataX = %d, tileDataW = %d\n", tileAddr+i, tileDataY, tileDataX, tileDataW);
                }
              } else {
                // padding data
                inTileXInc = 1;
              }
            }
          }
        }
        for ( int i = 0; i < TILE_BUF_SIZE; i++ ) 
          if ( tile_buffer[ i ].vec[0] != 1 ) {
            printf("load_fmap(): sanity check failed. tile_buf[%d]=",i);
            for ( int j = 0; j < BASE_PER_FBUS; j++ )
              printf("%d ", tile_buffer[i].vec[j]);
            printf("\n");
          }
        if ( DEBUG_LOAD_FMAP ) {
          printf("load_fmap(): Trying to feed data into pipe_fmap...\n");
        }
        // feed data into the pipe
        for ( int pfY = 0; pfY < actual_tile_ht-pfH+1; pfY += pool_stride*stride ) {
          for ( int pfX = 0; pfX < actual_tile_wid-pfW+1; pfX += pool_stride*stride ) {
            printf("load_fmap(): pfY = %d, pfX = %d\n", pfY, pfX);
            for ( int pwY = 0; pwY < pool_ht; pwY++ ) {
              for ( int pwX = 0; pwX < pool_wid; pwX++ ) {
                for ( int nBlk = 0; nBlk < fmap_nblk; nBlk++ ) {
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
                          int tileDataY = inTileY - condOffsetH_LU;
                          int tileDataX = inTileX - condOffsetW_LU;
                          // Bug fix: data from different blocks go to different blocks
                          int tileAddr = nBlk * tileDataH * tileDataW + tileDataY * tileDataW + tileDataX;

                          if ( DEBUG_LOAD_FMAP ) {
                            printf("load_fmap(): RD from tile_buf[%d]\n", tileAddr+fpo);
                          }

                          to_pipe.bus_val = tile_buffer[ tileAddr+fpo ].bus_val;
                        } else {
                          // feed 0 into the pipe
                          if ( DEBUG_LOAD_FMAP ) {
                            printf("load_fmap(): feed 0 padding to pipe\n" );
                          }
                          to_pipe.bus_val = 0;
                        }
                        printf("load_fmap(): trying to write to pipe\n");
                        printf("load_fmap(): nBlk=%d,inTileY=%d,inTileX=%d\n\t", nBlk,
                            inTileY, inTileX);
                        for ( int i = 0; i < BASE_PER_FBUS; i++ )
                          printf("%d ", to_pipe.vec[i]);
                        printf("\n");

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
