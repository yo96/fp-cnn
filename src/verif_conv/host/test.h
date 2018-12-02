#include <math.h> 
#include <iostream>
#include <algorithm> 
#include <assert.h>
#include "krnl/configs.h"
// A c++ reference for convolution
template <class base>
void test_conv( base * fmap, base * wts, base * ofmap, 
                const int fmap_wid, const int fmap_ht, const int fmap_dep,
                const int fil_wid, const int fil_ht, const int num_fil,
                const int pool_wid, const int pool_ht, const int pool_stride,
                const int conv_stride, const int padding, const int use_relu )
{ 

  const int blk_dep       = BASE_PER_OBUS/BASE_PER_FBUS; 
  const int row_ddr_size  = ceil( (float)(fmap_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS ); 
  const int ofmap_wid     = ceil( (float)(fmap_wid)/conv_stride );
  const int ofmap_ht      = ceil( (float)(fmap_ht )/conv_stride );
  const int ofmap_dep     = num_fil;
  //const int ofmap_ddr_wid = ceil( (float)(ofmap_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS ); 

  const int fil_dep  = fmap_dep;
  const int fil_size = fil_wid * fil_ht * fil_dep;
  // calculate aligned size of a filter
  const int fil_ddr_size = ceil( (float)(fil_size)/BASE_PER_DDRBUS );
  const int fil_ailgned_size = fil_ddr_size * BASE_PER_DDRBUS;
  
  const int padded_wid = fmap_wid + padding * 2;
  const int padded_ht  = fmap_ht  + padding * 2;

  assert(ofmap_wid >= pool_wid);
  assert(ofmap_ht  >= pool_ht );
  const int out_wid     = (ofmap_wid - pool_wid) / pool_stride + 1;
  const int out_ht      = (ofmap_ht  - pool_ht ) / pool_stride + 1;
  const int out_dep     = ofmap_dep;
  const int out_nblk    = ofmap_dep/BASE_PER_OBUS;
  const int out_db_wid  = ceil( (float)(out_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS );


  base fmap_tmp[fmap_ht][fmap_wid][fmap_dep];
  base  wts_tmp[num_fil][fil_ht][fil_wid][fmap_dep];
  base   padded[padded_ht][padded_wid][fmap_dep];
  base  out_tmp[ofmap_ht][ofmap_wid][ofmap_dep]; // output fmap after conv
  base pool_tmp[out_ht][out_wid][out_dep];

  // Reshape fmap
  int nblk     = fmap_dep/BASE_PER_OBUS;
  int blk_size = fmap_ht * row_ddr_size*BASE_PER_DDRBUS; 
  for (int nb=0;nb<nblk;nb++){
    int blk_offset = nb * blk_size;
    for (int y=0;y<fmap_ht;y++){
      for (int x=0;x<fmap_wid;x++){
        for (int d=0;d<blk_dep;d++){
          for (int z=0;z<BASE_PER_FBUS;z++){
            int d_offset = d*BASE_PER_FBUS;
            int x_offset = x*BASE_PER_OBUS;
            int y_offset = y*row_ddr_size*BASE_PER_DDRBUS;
            int rd_addr = z + d_offset + x_offset + y_offset + blk_offset;
            fmap_tmp[y][x][z+d*BASE_PER_FBUS+nb*BASE_PER_OBUS] = fmap[rd_addr];
          } // z < BASE_PER_FBUS
        }
      } // famp_wid
    } // fmap_ht
  } // nblk

  // Reshape wts 
  // Assumes wts are 64B aligned!!!! Need to change this in the future
  for (int nf=0;nf<num_fil;nf++){
    for (int nb=0;nb<nblk;nb++){
      for (int y=0;y<fil_ht;y++) {
        for (int x=0;x<fil_wid;x++){
          for (int d=0;d<blk_dep;d++){
            for (int z=0;z<BASE_PER_FBUS;z++){
              int d_offset  = d*BASE_PER_FBUS;
              int x_offset  = x*BASE_PER_OBUS;
              int y_offset  = y*fil_wid*BASE_PER_OBUS;
              int blk_offset = nb * fil_ht * fil_wid * BASE_PER_OBUS;  
              int nf_offset = nf*fil_ailgned_size;
              int rd_addr   = z + d_offset + x_offset + y_offset + blk_offset + nf_offset;
              wts_tmp[nf][y][x][z+d*BASE_PER_FBUS+nb*BASE_PER_OBUS] = wts[rd_addr];     
            } // z < BASE_PER_FBUS
          } // d < blk_dep
        } // x < fil_wid
      } // y < fil_ht 
    } // nb < nblk
  } // nf < num_fil

  // Padding
  for (int z=0;z<fmap_dep;z++)
    for (int x=0;x<padded_wid;x++)
      for (int y=0;y<padded_ht;y++)
        padded[y][x][z] = 0;

  for (int y=0;y<fmap_ht;y++)
    for (int x=0;x<fmap_wid;x++)
      for (int z=0;z<fmap_dep;z++)
        padded[y+padding][x+padding][z] = fmap_tmp[y][x][z];

  // test reshape
  //for (int nf=0;nf<num_fil;nf++){
  //  for (int y=0;y<fil_ht;y++){
  //    for (int x=0;x<fil_wid;x++){
  //      for (int z=0;z<fmap_dep;z++){
  //        if (wts_tmp[nf][y][x][z] != 1){
  //          std::cout << "(" << x << "," << y << "," << z << ") " 
  //          << "wts_tmp not right!" << std::endl;
  //        }
  //      }
  //    }
  //  }
  //}

  //for (int y=0;y<fmap_ht;y++){
  //  for (int x=0;x<fmap_wid;x++){
  //    for (int z=0;z<fmap_dep;z++){
  //      if (fmap_tmp[y][x][z] != 1){
  //        std::cout << "(" << x << "," << y << "," << z << ") " 
  //        << "fmap_tmp not right!" << std::endl;
  //      }
  //    }
  //  }
  //}

  //for (int y=0;y<padded_ht;y++){
  //  for (int x=0;x<padded_wid;x++){
  //    for (int z=0;z<fmap_dep;z++){
  //      if ((x<padding || y<padding || x>=padded_wid-padding || y>=padded_ht-padding) && padded[y][x][z] != 0) {
  //        std::cout << "(" << y << "," << x << "," << z << ") " 
  //                  << " padding not right! " << padded[y][x][z] << std::endl;
  //      }
  //      else if (!(x<padding || y<padding || x>=padded_wid-padding || y>=padded_ht-padding) && padded[y][x][z] != 1) {
  //        std::cout << "(" << y << "," << x << "," << z << ") " 
  //                  << " fmap not right! " << padded[y][x][z] << std::endl;
  //      }
  //    }
  //  }
  //}


  // Compute - Assumes [fil_wid] is odd
  int X = padded_wid - fil_wid + 1;
  int Y = padded_ht  - fil_ht  + 1;
  for (int nf=0;nf<num_fil;nf++){
    for (int y=0;y<Y;y+=conv_stride){
      for (int x=0;x<X;x+=conv_stride){
        base sum = 0;
        for (int r=0;r<fil_ht;r++){
          for (int c=0;c<fil_wid;c++){
            for (int z=0;z<fmap_dep;z++){
              sum += padded[y+r][x+c][z] * wts_tmp[nf][r][c][z];
            }
          }
        }
        out_tmp[y/conv_stride][x/conv_stride][nf] = sum;
      }
    }
  }
 
  // ReLU
  for (int y=0;y<ofmap_ht;y++)
    for (int x=0;x<ofmap_wid;x++)
      for (int z=0;z<ofmap_dep;z++)
        if (use_relu)
          out_tmp[y][x][z] = std::max<base>(out_tmp[y][x][z], 0);
  
  // pooling
  X = ofmap_wid - pool_wid + 1;
  Y = ofmap_ht  - pool_ht  + 1;
  for (int z=0;z<out_dep;z++     ){  
    for (int y=0;y<Y;y+=pool_stride){
      for (int x=0;x<X;x+=pool_stride){
        // pooling wnd
        base cur_max = out_tmp[y][x][z];
        for (int r=0;r<pool_ht;r++){
          for(int c=0;c<pool_wid;c++){
            cur_max = std::max<base>(cur_max, out_tmp[y+r][x+c][z]);
          }
        }
        pool_tmp[y/pool_stride][x/pool_stride][z] = cur_max;
      }
    }
  }

  // Reshape output
  int out_blk_size = out_ht * out_db_wid*BASE_PER_DDRBUS;
  for (int nb=0;nb<out_nblk;nb++){
    for (int y=0;y<out_ht;y++){
      for (int x=0;x<out_wid;x++){
        for (int d=0;d<blk_dep;d++){
          for (int z=0;z<BASE_PER_FBUS;z++){
            int d_offset   = d*BASE_PER_FBUS;
            int x_offset   = x*BASE_PER_OBUS;
            int y_offset   = y*out_db_wid*BASE_PER_DDRBUS;
            int blk_offset = nb * out_blk_size;
            int wr_addr    = z + d_offset + x_offset + y_offset + blk_offset;
            ofmap[wr_addr] = pool_tmp[y][x][z+d*BASE_PER_FBUS+nb*BASE_PER_OBUS];
          } // z < BASE_PER_FBUS
        } // d < blk_dep
      } // famp_wid
    } // fmap_ht
  } // nblk
  printf("Test finished!\n");
}