#include <math.h> 
#include <iostream>
#include "krnl/configs.h"
// A c++ reference for convolution
template <class base>
void test_conv( base * fmap, base * wts, base * ofmap, 
                const int fmap_wid, const int fmap_ht, const int fmap_dep,
                const int fil_wid, const int fil_ht, const int num_fil,
                const int stride, const int padding )
{  
  const int ofmap_wid   = ceil( (float)(fmap_wid)/stride );
  const int ofmap_ht    = ceil( (float)(fmap_ht )/stride );
  const int ofmap_dep   = num_fil;
  
  const int padded_wid = fmap_wid + padding * 2;
  const int padded_ht  = fmap_ht  + padding * 2;

  base fmap_tmp[fmap_ht][fmap_wid][fmap_dep];
  base  wts_tmp[num_fil][fil_ht][fil_wid][fmap_dep];
  base   padded[padded_ht][padded_wid][fmap_dep];
  base  out_tmp[ofmap_ht][ofmap_wid][ofmap_dep];

  // Reshape fmap
  int nblk     = fmap_dep/BASE_PER_FBUS;
  int blk_size = fmap_wid * fmap_ht * BASE_PER_FBUS; 
  for (int nb=0;nb<nblk;nb++){
    int blk_offset = nb * blk_size;
    for (int y=0;y<fmap_ht;y++){
      for (int x=0;x<fmap_wid;x++){
        for (int z=0;z<BASE_PER_FBUS;z++){
          int x_offset = x*BASE_PER_FBUS;
          int y_offset = y*fmap_wid*BASE_PER_FBUS;
          int rd_addr = z + x_offset + y_offset + blk_offset;
          fmap_tmp[y][x][z+nb*BASE_PER_FBUS] = fmap[rd_addr];
        } // z < BASE_PER_FBUS
      } // famp_wid
    } // fmap_ht
  } // nblk

  // Reshape wts 
  // Assumes wts are 64B aligned!!!! Need to change this in the future
  int fil_size = fil_ht * fil_wid * fmap_dep;  
  for (int nf=0;nf<num_fil;nf++){
    for (int nb=0;nb<nblk;nb++){
      for (int y=0;y<fil_ht;y++) {
        for (int x=0;x<fil_wid;x++){
          for (int z=0;z<BASE_PER_FBUS;z++){
            int x_offset  = x*BASE_PER_FBUS;
            int y_offset  = y*fil_wid*BASE_PER_FBUS;
            int blk_offset = nb * fil_ht * fil_wid * BASE_PER_FBUS;  
            int nf_offset = nf*fil_size;
            int rd_addr   = z + x_offset + y_offset + blk_offset + nf_offset;
            wts_tmp[nf][y][x][z+nb*BASE_PER_FBUS] = wts[rd_addr];     
          } // z < BASE_PER_FBUS
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
        padded[y+1][x+1][z] = fmap_tmp[y][x][z];

  // test reshape
  for (int nf=0;nf<num_fil;nf++){
    for (int y=0;y<fil_ht;y++){
      for (int x=0;x<fil_wid;x++){
        for (int z=0;z<fmap_dep;z++){
          if (wts_tmp[nf][y][x][z] != 1){
            std::cout << "(" << x << "," << y << "," << z << ") " 
            << "wts_tmp not right!" << std::endl;
          }
        }
      }
    }

  }

  for (int y=0;y<fmap_ht;y++){
    for (int x=0;x<fmap_wid;x++){
      for (int z=0;z<fmap_dep;z++){
        if (fmap_tmp[y][x][z] != 1){
          std::cout << "(" << x << "," << y << "," << z << ") " 
          << "fmap_tmp not right!" << std::endl;
        }
      }
    }
  }

  for (int y=0;y<padded_ht;y++){
    for (int x=0;x<padded_wid;x++){
      for (int z=0;z<fmap_dep;z++){
        if ((x==0 || y==0 || x==padded_wid-1 || y==padded_wid-1) && padded[y][x][z] != 0) {
          std::cout << "(" << y << "," << x << "," << z << ") " 
                    << " padding not right! " << padded[y][x][z] << std::endl;
        }
        else if (!(x==0 || y==0 || x==padded_wid-1 || y==padded_wid-1) && padded[y][x][z] != 1) {
          std::cout << "(" << y << "," << x << "," << z << ") " 
                    << " fmap not right! " << padded[y][x][z] << std::endl;
        }
      }
    }
  }


  // Compute - Assumes [fil_wid] is odd
  int X = padded_wid - fil_wid + 1;
  int Y = padded_ht  - fil_ht  + 1;
  for (int nf=0;nf<num_fil;nf++){
    for (int y=0;y<Y;y+=stride){
      for (int x=0;x<X;x+=stride){
        base sum = 0;
        for (int r=0;r<fil_ht;r++){
          for (int c=0;c<fil_wid;c++){
            for (int z=0;z<fmap_dep;z++){
              sum += padded[y+r][x+c][z] * wts_tmp[nf][r][c][z];
            }
          }
        }
        out_tmp[y/stride][x/stride][nf] = sum;
      }
    }
  }

  // Reshape output
  int o_nblk     = ofmap_dep/BASE_PER_OBUS; //  OBUS = FBUS
  int o_blk_size = ofmap_ht * ofmap_wid * BASE_PER_OBUS;
  for (int nb=0;nb<o_nblk;nb++){
    for (int y=0;y<ofmap_ht;y++){
      for (int x=0;x<ofmap_wid;x++){
        for (int z=0;z<BASE_PER_OBUS;z++){
          int x_offset   = x*BASE_PER_OBUS;
          int y_offset   = y*ofmap_wid*BASE_PER_FBUS;
          int blk_offset = nb * o_blk_size;
          int wr_addr    = z + x_offset + y_offset + blk_offset;
          ofmap[wr_addr] = out_tmp[y][x][z+nb*BASE_PER_OBUS];
        } // z < BASE_PER_FBUS
      } // famp_wid
    } // fmap_ht
  } // nblk
  printf("finished!\n");
}