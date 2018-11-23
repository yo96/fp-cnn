#include "defs.h"
#include "configs.h"
#define SHREG_SIZE    (SYS_WID+1)*SYS_WID/2 //1+2+3+...+16

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
  int o_nblk,
  int n_fil,
  int fil_size, // in wts_bus_
  int n_iter ) 
{
  bool debug = false;

  // On-chip storage
  wts_bus_t wts_ram[NUM_FIL_BUF][FIL_BUF_SIZE]
  __attribute__((xcl_array_partition(complete, 1)))
  __attribute__((xcl_array_reshape(cyclic, WBUS_PER_DDRBUS, 2)));

  //wts_bus_t wts_ser[NUM_FIL_BUF][WBUS_PER_DDRBUS]
  //__attribute__((xcl_array_partition(complete, 0)));

  base sys[SYS_HT][SYS_WID] 
  __attribute__((xcl_array_partition(complete, 0)));

  base shreg_wts[NUM_FIL_BUF][SHREG_SIZE]  
  __attribute__((xcl_array_partition(complete, 0)));
  
  base shreg_fmap[SHREG_SIZE]
  __attribute__((xcl_array_partition(complete, 1)));

  for (int ni=0;ni<n_iter;ni++){
    // read wts into BRAM
    LOAD_WTS: 
    for (int i=0;i<NUM_FIL_BUF;i++){
      for (int j=0;j<fil_size;j+=WBUS_PER_DDRBUS){        
        ddr_bus from_wts_pipe;
        wts_bus to_wts_ram[WBUS_PER_DDRBUS];
        read_pipe_block(pipe_wts, &from_wts_pipe.bus_val);
        // Serialize ddr_bus into wts_bus
        __attribute__((opencl_unroll_hint))
        for (int k=0;k<WBUS_PER_DDRBUS;k++){
          for (int m=0;m<BASE_PER_WBUS;m++){
            int offset = k*BASE_PER_WBUS;
            to_wts_ram[k].vec[m] = from_wts_pipe.vec[m+offset];
            wts_ram[i][j+k] = to_wts_ram[k].bus_val;                                
          } // m
        } // k
      } // j - fil_size
    } // i - NUM_FIL_BUF
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
            
            // feed fmap into shreg
            fmap_bus fbus;
            read_pipe_block(pipe_fmap, &fbus.bus_val);
            if(debug && x==0 && y==0 && f==8){
              printf("[comp]: fmap->shreg:");                  
            }
            __attribute__((opencl_unroll_hint))
            for (int i=0,j=0,nreg=0;j<SYS_WID;i+=nreg,j++){
              nreg ++;
              shreg_fmap[i] = fbus.vec[j];
              if(debug && x==0 && y==0 && f==8){
                printf(" %d->shreg[%d]", fbus.vec[j],i);                  
              }
            }
            if(debug && x==0 && y==0 && f==8){
                printf("\n");                  
            }
            // feed wts into shreg
            __attribute__((opencl_unroll_hint))
            for (int nf=0;nf<NUM_FIL_BUF;nf++){
              wts_bus wbus;
              wbus.bus_val = wts_ram[nf][f];
              for (int i=0,j=0,nreg=0;j<SYS_WID;i+=nreg,j++){
                nreg++;               
                shreg_wts[nf][i] = wbus.vec[j];
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
                base fmap_val, wts_val, acc;
                // get result from previous PE
                acc = (sc==0)? 0 : sys[sr][sc-1];

                // get fmap from shreg
                for (int i=shreg_idx+nreg,j=sc+1,nr=nreg;j<SYS_WID;i+=nr,j++){
                  nr ++;
                  shreg_fmap[i+sc+1] = shreg_fmap[i+sc];
                  if(debug && x==0 && y==0 && f==8){
                    printf("sc:%d ,%d<=%d\n",sc, i+sc+1, i+sc);                  
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
                if(debug && x==0 && y==0 && f==8){
                  printf("[comp]: sys(%d,%d), fmap:%d, wts:%d, acc:%d\n",
                    sr,sc,fmap_val, wts_val, acc);                  
                }
                sys[sr][sc] = fmap_val * wts_val + acc;
                shreg_idx += nreg;
              } // SYS_HT
            } // SYS_WID

            // get ouput from systolic array
            out_bus obus;
            for (int i=0;i<SYS_HT;i++){
              obus.vec[i] = sys[i][SYS_WID-1];
            }
            if (debug && x==0 && y==0 && f==8){
              printf("compute(): writing pipe with\n    ");
              for (int di=0;di<BASE_PER_OBUS;di++)
                printf(" %d,", obus.vec[di]);
              printf("\n");
            }
            write_pipe_block(pipe_out, &obus.bus_val);
            
          } // fil_size

      } // o_wid
    } // o_ht
    //printf("comp: %d-th iteration.\n", ni);
  } // n_iter
  //printf("[comp]: done\n");
}