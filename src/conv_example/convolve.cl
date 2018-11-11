#define FMAP_PIPE_DEPTH 64
#define WTS_PIPE_DEPTH  16
#define OUT_PIPE_DEPTH  128

#define Tn 32
#define Tm 32
#define BUS_N_ELE 32

#define FILTER_WID   3
#define IMG_WID      56
#define N_FILTER     192
#define N_FILTER_BUF 32
#define PADDING      1

#define N_ITER       1
/******************************************************************************
 * Type definitions 
 ******************************************************************************/

//typedef short bus_t;  
typedef uint16 bus_t;
typedef union {
	bus_t bus_val;
	short ele[BUS_N_ELE] __attribute__((xcl_array_partition(complete,1)));
} short_bus; 

typedef union {
	bus_t bus_val;
	int ele[BUS_N_ELE/2] __attribute__((xcl_array_partition(complete,1)));
	} int_bus;

/*****************************************************************************
 * Helper functions
 *****************************************************************************/

/* [ bus_to_short( in, out) ] allocates the input bus [in] 
 * into a short array [out] */
void bus_to_short(bus_t in, short out[BUS_N_ELE]){
	short_bus tmp;
	tmp.bus_val = in;

	__attribute__((opencl_unroll_hint(BUS_N_ELE)))
	for(int i = 0; i < BUS_N_ELE; i++) {
		out[i] = tmp.ele[i];
	}
}

/* [ bus_to_short( in, out) ] allocates the input bus [in] 
 * into an int array [out] */
void bus_to_int(bus_t in, int out[BUS_N_ELE/2]){
	int_bus tmp;
	tmp.bus_val = in;

	__attribute__((opencl_unroll_hint(BUS_N_ELE/2)))
	for(int i = 0; i < BUS_N_ELE/2; i++) {
		out[i] = tmp.ele[i];
	}
}

/* [ short_to_bus(in) ] combines the 32 shorts in [in] into a 512-bit bus. */
bus_t short_to_bus(short in[BUS_N_ELE]){
	short_bus ret;

	__attribute__((opencl_unroll_hint(BUS_N_ELE)))
	for(int i = 0; i < BUS_N_ELE; i++) {
		ret.ele[i] = in[i];
	}
	return ret.bus_val;
}

/* [ int_to_bus(in) ] combines the 32 ints in [in] into a 512-bit bus. */
bus_t int_to_bus(int in[BUS_N_ELE/2]){
	int_bus ret;
	__attribute__((opencl_unroll_hint(BUS_N_ELE/2)))
	for(int i = 0; i < BUS_N_ELE/2; i++) {
		ret.ele[i] = in[i];
	}
	return ret.bus_val;
}


/*****************************************************************************
 * pipe definition
 *****************************************************************************/
pipe bus_t image_pipe    __attribute__((xcl_reqd_pipe_depth(FMAP_PIPE_DEPTH)));

pipe bus_t weight_pipe   __attribute__((xcl_reqd_pipe_depth(WTS_PIPE_DEPTH)));

pipe bus_t output_pipe0  __attribute__((xcl_reqd_pipe_depth(OUT_PIPE_DEPTH)));
pipe bus_t output_pipe1  __attribute__((xcl_reqd_pipe_depth(OUT_PIPE_DEPTH)));


/*============================================================================*
 * Kernel definitions
 *============================================================================*/

/*****************************************************************************
 * [load_img] loads the input feature map data.
 *****************************************************************************/

// no buffer, just streaming
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_img(
	__global bus_t * __restrict img,
	int img_wid,
	int img_ht,
	int img_depth,
	int filter_wid,
	int padding,
	int stride,
	int n_iter
	) 
{	
	//int num = 0;
	// load the image 3 times
	int rd_addr = 0;
	bus_t to_pipe;
	int bound_x   = img_wid+padding*2-filter_wid+1; // 56+2-3+1 = 56
	int bound_y   = img_ht +padding*2-filter_wid+1; // 56+2-3+1 = 56
	for(int idx=0;idx<n_iter;idx++){
		//printf(">>>reading image: %d...\n", n_iter);
		rd_addr = 0;
		// feed data into pipes
		feed_fmap:
		for(int i=0;i<bound_x;i+=stride){
			for(int k=0;k<bound_y;k+=stride){
				for(int n=0;n<filter_wid;n++){					
					__attribute__((xcl_pipeline_loop))
					for(int m=0;m<filter_wid*2;m++){
						if ((i+n)==0||(i+n)==img_wid+padding*2-1||(k+m/2)==0||(k+m/2)==img_wid+padding*2-1)
							to_pipe = 0;
						else
							to_pipe = img[(i+n-1)*img_wid*2+(k-1)*2+m];
						write_pipe_block(image_pipe, &to_pipe);					
					}// m
				}//n
			}//k
		}//i	
	} // end of n_iter
	//printf("load fmap done!\n");
}
/*
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_img(
	__global bus_t * __restrict img,
	int img_size 
	) 
{	
	bus_t buffer_img[FILTER_WID][IMG_WID+PADDING*2][2];
	// load the image 6 times
	for(int n_iter=0;n_iter<6;n_iter++){
		//printf(">>>reading image: %d...\n", n_iter);
		__attribute__((xcl_pipeline_loop))
		for(int i=0;i<IMG_WID+PADDING*2-FILTER_WID+1;i++){
				// fetch data from memory
				for(int m=0;m<FILTER_WID;m++){
					for(int n=0;n<IMG_WID+PADDING*2;n++){
						for(int j=0;j<2;j++)
							buffer_img[m][n][j] = img[(i+m)*(IMG_WID+PADDING*2)*2+n*2+j];
							// write
					}
				}
				// feed data into pipes
				for(int k=0;k<IMG_WID+PADDING*2-FILTER_WID+1;k++){
					for(int m=0;m<FILTER_WID;m++){
						for(int n=0;n<FILTER_WID;n++){
								bus_t to_pipe0 = buffer_img[m][k+n][0];
								write_pipe_block(image_pipe, &to_pipe0);
								bus_t to_pipe1 = buffer_img[m][k+n][1];
								write_pipe_block(image_pipe, &to_pipe1);		
						}						
					}
				}
		}	
	} // end of n_iter
	//printf("load_img is done!\n");
}
*/
/*****************************************************************************
 * [load_weight] loads the weights from global memory.
 *****************************************************************************/
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_weights(
	__global bus_t* __restrict weights,
	int w8_size 
	) 
{	
	bus_t w8_bus_buffer[Tn] __attribute__((xcl_array_partition(complete,1)));


	__attribute__((xcl_pipeline_loop))
	for(int k=0;k<w8_size;k++){

		// load buffer & write pipes 
		bus_t bus_to_pipe = weights[k];
		write_pipe_block( weight_pipe, &bus_to_pipe );
		//printf("(load_w8): writing weight[%d] to pipes\n",k);
	}
	//printf("Weights loaded!\n");
}

/*****************************************************************************
 * [convolve] is simply a functional level model of systolic array. 
 * It is highly un-optimized!
 *****************************************************************************/
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void convolve( 
	int fmap_width,
	int fmap_height,
	int n_filter,
	int filter_wid,
	int n_iter
	) 
{
	short img_buffer[32]     __attribute__((xcl_array_partition(complete, 1)));
	short w8_buffer [64][32] __attribute__((xcl_array_partition(complete, 1)));
	bus_t w8_bus_buffer[64]  __attribute__((xcl_array_partition(complete, 1)));

	short out_buffer[64]     __attribute__((xcl_array_partition(complete, 1)));
	
	bus_t img_bus;
	bus_t wts_bus;
	bus_t out_bus0;
	bus_t out_bus1;

	bus_t ram_wts[64][18] __attribute__((xcl_array_partition(complete,1)));

	//load data
	for(int i=0;i<n_iter;i++){
		printf("%d-th iteration!\n", i);
		// load 64 filters from pipe
		for(int n=0;n<64;n++){
			for(int m=0;m<filter_wid*filter_wid*2;m++){
				read_pipe_block( weight_pipe,&ram_wts[n][m] ); 
			}
		}
		iterate_img:
		for(int x=0;x<fmap_width;x++){
			for(int y=0;y<fmap_height;y++){
				compute:
				__attribute__(( xcl_pipeline_loop ))
				for(int n=0;n<filter_wid*filter_wid*2;n++){
					//load feature map
					read_pipe_block(image_pipe, &img_bus);
					bus_to_short(img_bus, img_buffer);
		
					// load weights data
					for(int k=0;k<64;k++){
						w8_bus_buffer[k] = ram_wts[k][n];
					}
					// decompose the loaded weight buses into short integers
					for(int j=0;j<64;j++)
						bus_to_short(w8_bus_buffer[j],w8_buffer[j]);

					//calculate
					for(int j=0;j<64;j++){
						int ret = 0;
						for(int k=0; k<32; k++){
							ret = ret + (img_buffer[k] * w8_buffer[j][k]);
						}
						//printf("(convolve): result is %d\n", ret);
						out_buffer[j] = ret;
					}		
					//output
					out_bus0 = short_to_bus(  out_buffer     );
					out_bus1 = short_to_bus( (out_buffer+32) );
					write_pipe_block(output_pipe0, &out_bus0);
					write_pipe_block(output_pipe1, &out_bus1);	
				} // compute-n, outputs 36

			} // iterate_img y
		} // iterate img x
	} //i
} // convolve

/*****************************************************************************
 * [load_output] loads the output from systolic.
 *****************************************************************************/
__attribute__((reqd_work_group_size(1,1,1)))
__kernel
void load_output(
	__global bus_t* __restrict img_out,
	int img_size
	) 
{
	bus_t out_bus0;
	bus_t out_bus1;
	bus_t out_reg[2];

	short out_bus_short0[32];
	short out_bus_short1[32];
	short out_reg_short0[32];
	short out_reg_short1[32];
	
	int n_trans;
	bus_t tmp[64];
	//__attribute__((xcl_pipeline_loop))
	output:
	for(int i=0;i<img_size;i+=64){

		n_trans = (img_size-i < 64? img_size-i : 64);
		for(int k=0;k<n_trans;k+=2){

			out_reg[0] = 0;
			out_reg[1] = 0;

			tmp_out:
			__attribute__((xcl_pipeline_loop))
			for(int j=0;j<FILTER_WID*FILTER_WID*2;j++) {
				read_pipe_block(output_pipe0, &out_bus0);	
				read_pipe_block(output_pipe1, &out_bus1);	

				bus_to_short(out_bus0,out_bus_short0);
				bus_to_short(out_bus1,out_bus_short1);

				bus_to_short(out_reg[0],out_reg_short0);
				bus_to_short(out_reg[1],out_reg_short1);

			// v-v add
				//__attribute__((opencl_unroll_hint))
				for(int m=0;m<32;m++){
					out_reg_short0[m] += out_bus_short0[m];
					out_reg_short1[m] += out_bus_short1[m];
				}

				out_reg[0] = short_to_bus( out_reg_short0 );
				out_reg[1] = short_to_bus( out_reg_short1 );

				tmp[k  ] = out_reg[0];
				tmp[k+1] = out_reg[1];
			}// j

		} // k

		// burst write
		__attribute__(( xcl_pipeline_loop ))
		for(int p=0;p<n_trans;p++){
			img_out[i+p] = tmp[p];
		}
	} // i

	//printf("remaining transactions: %d\n", img_size-acc);
	//printf("Output done!\n");
}	
