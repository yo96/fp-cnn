//#include "ap_int.h"
#include "def_helper.h"

#define N_PER_BUS 16

#define FIL_BUF_WID   3
#define FIL_BUF_HT    3
#define FIL_BUFF_SIZE 9
#define NUM_FIL_BUF   16
#define SYS_WID       16
#define SYS_HT        16
#define SHREG_SIZE    136 //1+2+3+...+16

typedef int base;

/******************************************************************************
 * [compute] kernel
 *****************************************************************************
  This kernel describe a 16x16 systolic array 
 */
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void compute( 
  int o_wid,
  int o_ht,
  int n_fil,
  int fil_size,
  int n_iter ) 
{
  bus_t wts_ram[NUM_FIL_BUF][FIL_BUFF_SIZE] 
  __attribute__((xcl_array_partition(complete, 1)));

  base sys[SYS_HT][SYS_WID] 
  __attribute__((xcl_array_partition(complete, 0)));

  base shreg_wts[NUM_FIL_BUF][SHREG_SIZE]  
  __attribute__((xcl_array_partition(complete, 0)));
  
  base shreg_fmap[SHREG_SIZE]
  __attribute__((xcl_array_partition(complete, 1)));

  //printf("comp: args: o_wid:%d, o_ht:%d, n_fil:%d, fil_size:%d, n_iter:%d.\n",
  //       o_wid, o_ht, n_fil, fil_size, n_iter);
  for (int ni=0;ni<n_iter;ni++){
    // read wts into BRAM
    LOAD_WTS: 
    for (int i=0;i<NUM_FIL_BUF;i++){
      for (int j=0;j<FIL_BUFF_SIZE;j++){
        bus_t wts_bus;
        read_pipe_block(pipe_wts, &wts_bus);
        wts_ram[i][j] = wts_bus;
      }
    }
    //printf("comp: wts loaded.\n");
    // iterating output fmap
    ITER_FMAP:
    for (int y=0;y<o_ht;y++){
      for (int x=0;x<o_wid;x++){
        //printf("comp: iterating ofmap(%d,%d).\n",y,x);
        
          __attribute__((xcl_pipeline_loop))
          COMPUTE:
          for (int f=0;f<fil_size;f++){
            //printf("comp: iterating ofmap(%d/%d,%d/%d), filter_idx:%d\n",y,o_ht-1,x,o_wid-1,f);
            int fil_idx = f;
            //TODO: filter depth maybe?
            
            // feed fmap into shreg
            bus_t fmap_bus;
            base fmap_base[N_PER_BUS];
            read_pipe_block(pipe_fmap, &fmap_bus);
            bus_to_int(fmap_bus, fmap_base);
            
            if(x==0 && y==0 && f==8){
              //printf("[comp]: fmap->shreg:");                  
            }
            __attribute__((opencl_unroll_hint))
            for (int i=0,j=0,nreg=0;j<N_PER_BUS;i+=nreg,j++){
              nreg ++;
              shreg_fmap[i] = fmap_base[j];
              if(x==0 && y==0 && f==8){
                //printf(" %d->shreg[%d]", fmap_base[j],i);                  
              }
            }
            if(x==0 && y==0 && f==8){
                //printf("\n");                  
            }

            // feed wts into shreg

            __attribute__((opencl_unroll_hint))
            for (int nf=0;nf<NUM_FIL_BUF;nf++){
              bus_t wts_bus = wts_ram[nf][fil_idx];
              base wts_base[N_PER_BUS];
              bus_to_int(wts_bus, wts_base);
              for (int i=0,j=0,nreg=0;j<N_PER_BUS;i+=nreg,j++){
                nreg++;
                shreg_wts[nf][i] = wts_base[j];
              }           

            } // NUM_FIL_BUF

            // systolic array
            __attribute__((opencl_unroll_hint))
            SYSTOLIC:
            for (int sr=0;sr<SYS_HT;sr++){
              sys[sr][0] = 0;
              int shreg_idx = 0;
              for (int sc=0;sc<SYS_WID;sc++){
                int nreg = sc+1;
                int last_reg = shreg_idx + nreg-1;
                base fmap_val, wts_val, acc;
                // get result from previous PE
                acc = (sc==0)? 0 : sys[sr][sc-1];

                // get fmap from shreg
                for (int i=shreg_idx+nreg,j=sc+1,nr=nreg;j<SYS_WID;i+=nr,j++){
                  nr ++;
                  shreg_fmap[i+sc+1] = shreg_fmap[i+sc];
                  if(x==0 && y==0 && f==8){
                    //printf("sc:%d ,%d<=%d\n",sc, i+sc+1, i+sc);                  
                  }
                } // nreg
                fmap_val = shreg_fmap[shreg_idx+nreg-1];

                // get wts from shreg
                for (int i=shreg_idx+nreg,j=sc+1,nr=nreg;j<SYS_WID;i+=nr,j++){
                  nr ++;
                  shreg_wts[sr][i+sc+1] = shreg_wts[sr][i+sc];
                } // nreg
                wts_val = shreg_wts[sr][shreg_idx+nreg-1];

                // MAC
                if(x==0 && y==0 && f==8){
                  //printf("[comp]: sys(%d,%d), fmap:%d, wts:%d, acc:%d\n",sr,sc,fmap_val, wts_val, acc);                  
                }
                sys[sr][sc] = fmap_val * wts_val + acc;
                shreg_idx += nreg;
              } // SYS_HT
            } // SYS_WID

            // get ouput from systolic array
            bus_t out_bus;
            base  out_base[N_PER_BUS];
            for (int i=0;i<N_PER_BUS;i++){
              out_base[i] = sys[i][SYS_WID-1];
            }
            out_bus = int_to_bus(out_base);
            write_pipe_block(pipe_out, &out_bus);
            
          } // fil_size

      } // o_wid
    } // o_ht
    //printf("comp: %d-th iteration.\n", ni);
  } // n_iter
  //printf("comp: done\n");
}