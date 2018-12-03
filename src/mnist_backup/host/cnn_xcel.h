#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <CL/cl_ext.h>
#include <math.h>
#include <assert.h>
#include "allocator.h"
#include "krnl/configs.h"
#include "test.h"
#include "args.h"
//TARGET_DEVICE macro needs to be passed from gcc command line
#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
    #define STR_VALUE(arg)      #arg
    #define GET_STRING(name) STR_VALUE(name)
    #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

#define BILLION 1000000000L
// assumes a square systolic array
typedef short base;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

class CnnAccelerator {

  public: 

    CnnAccelerator() :
      context( nullptr ), q( nullptr ), buf( nullptr ),
      program( nullptr ), krnl_load_fmap( nullptr ),
      krnl_load_wts( nullptr ), krnl_conv( nullptr ),
      krnl_relu( nullptr ), krnl_pool( nullptr ),
      dummy_vec0( MAX_MID_SIZE, 0 ), dummy_vec1( MAX_MID_SIZE, 0 ),
      wts_vec( NUM_LAYERS ), wts_buf_vec( NUM_LAYERS )
    {}

    ~CnnAccelerator() {}

    void initialize( int argc, char* argv[] ) {
      //TARGET_DEVICE macro needs to be passed from gcc command line
      const char *target_device_name = TARGET_DEVICE;

      if(argc != 2) {
          std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
          return;
      }

      char* xclbinFilename = argv[1];

      bool found_device = false;

      //traversing all Platforms To find Xilinx Platform and targeted
      //Device in Xilinx Platform
      cl::Platform::get(&platforms);
      for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
          devices.clear();
          platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);

          //Traversing All Devices of Xilinx Platform
          for (size_t j = 0 ; j < devices.size() ; j++){
            device = devices[j];
            std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
            if (deviceName == target_device_name){
                found_device = true;
                break;
            }
          }
        }
      }
      if (found_device == false){
         std::cout << "Error: Unable to find Target Device " 
                   << target_device_name << std::endl;
         return; 
      }

      // Creating Context and Command Queue for selected device
      context = new cl::Context(device);
      q = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE | 
                                          CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
      // Load xclbin 
      std::cout << "\nLoading: '" << xclbinFilename << "'\n";
      std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
      bin_file.seekg (0, bin_file.end);
      unsigned nb = bin_file.tellg();
      bin_file.seekg (0, bin_file.beg);
      buf = new char [nb];
      bin_file.read(buf, nb);
      
      // Creating Program from Binary File
      bins.push_back({buf,nb});
      devices.resize(1);
      program = new cl::Program(*context, devices, bins);
      
      krnl_load_fmap = new cl::Kernel(*program, "load_fmap");
      krnl_load_wts  = new cl::Kernel(*program, "load_wts" );
      krnl_conv      = new cl::Kernel(*program, "compute"  );
      krnl_relu      = new cl::Kernel(*program, "acc_relu" );    
      krnl_pool      = new cl::Kernel(*program, "pool_wb"  );

      // Initialize intermediate buffers
      mid_buf0 = new cl::Buffer( *context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
          MAX_MID_SIZE*sizeof(base)*BASE_PER_DDRBUS, dummy_vec0.data(), NULL );
      mid_buf1 = new cl::Buffer( *context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
          MAX_MID_SIZE*sizeof(base)*BASE_PER_DDRBUS, dummy_vec1.data(), NULL );
      q->enqueueMigrateMemObjects({*mid_buf0, *mid_buf1}, 0); /* 0 means from host*/
      q->finish(); 
    }

    void load_wts( int idx, base* ptr, size_t size ) {
      wts_vec[idx] = new std::vector<base, aligned_allocator<base>>( ptr, ptr+size );
      wts_buf_vec[idx] = new cl::Buffer( *context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                          size*sizeof(base), wts_vec[idx]->data(), NULL );
      q->enqueueMigrateMemObjects({*wts_buf_vec[idx]}, 0); /* 0 means from host*/
      q->finish();
    }

    cl::Buffer* create_buffer( cl_mem_flags flags, size_t size, base* ptr ) {
      return new cl::Buffer(*context, flags, size, ptr, NULL);
    }

    void run_inference( const cl::Buffer& buf_fmap, cl::Buffer& buf_out ) {
      q->enqueueMigrateMemObjects({buf_fmap, buf_out}, 0); /* 0 means from host*/
      q->finish();

      struct timespec start, end;
      double time;

      std::cout << "\nLaunching kernel..." <<std::endl;
      clock_gettime(CLOCK_MONOTONIC, &start);

      layer_arg_t* arg = new layer_arg_t();

      set_kernel_arg( *arg,
    FMAP_WID_0, FMAP_HT_0, FMAP_DEP_0, 
    FIL_WID_0, FIL_HT_0, NUM_FIL_0, 
    POOL_WID_0, POOL_HT_0, POOL_STRIDE_0, 
    CONV_STRIDE_0, TILE_WID_0, TILE_HT_0,
    LPADDING_0, RPADDING_0, UPADDING_0, DPADDING_0
  );
  exec_layer( *arg, buf_fmap, *mid_buf0, *wts_buf_vec[0], 1 );
  
  set_kernel_arg( *arg,
    FMAP_WID_1, FMAP_HT_1, FMAP_DEP_1, 
    FIL_WID_1, FIL_HT_1, NUM_FIL_1, 
    POOL_WID_1, POOL_HT_1, POOL_STRIDE_1, 
    CONV_STRIDE_1, TILE_WID_1, TILE_HT_1,
    LPADDING_1, RPADDING_1, UPADDING_1, DPADDING_1
  );
  exec_layer( *arg, *mid_buf0, *mid_buf1, *wts_buf_vec[1], 1 );
  
  set_kernel_arg( *arg,
    FMAP_WID_2, FMAP_HT_2, FMAP_DEP_2, 
    FIL_WID_2, FIL_HT_2, NUM_FIL_2, 
    POOL_WID_2, POOL_HT_2, POOL_STRIDE_2, 
    CONV_STRIDE_2, TILE_WID_2, TILE_HT_2,
    LPADDING_2, RPADDING_2, UPADDING_2, DPADDING_2
  );
  exec_layer( *arg, *mid_buf1, *mid_buf0, *wts_buf_vec[2], 1 );
  
  set_kernel_arg( *arg,
    FMAP_WID_3, FMAP_HT_3, FMAP_DEP_3, 
    FIL_WID_3, FIL_HT_3, NUM_FIL_3, 
    POOL_WID_3, POOL_HT_3, POOL_STRIDE_3, 
    CONV_STRIDE_3, TILE_WID_3, TILE_HT_3,
    LPADDING_3, RPADDING_3, UPADDING_3, DPADDING_3
  );
  exec_layer( *arg, *mid_buf0, buf_out, *wts_buf_vec[3], 0 );
  
  

      clock_gettime(CLOCK_MONOTONIC, &end);

      time = BILLION * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
      time = time / BILLION;
      std::cout << "Execution complete..." << std::endl
                << "Elapsed time:" << time*1000 << "ms" << std::endl << std::endl;

      std::cout << "Reading results from DDR..." << std::endl;

      q->enqueueMigrateMemObjects({buf_out},CL_MIGRATE_MEM_OBJECT_HOST);
      q->finish();
    }

  private:

    typedef struct{
      int fmap_wid, fmap_ht, fmap_dep;
      int fil_wid, fil_ht, num_fil;
      int pool_wid, pool_ht, pool_stride;
      int conv_stride, tile_wid, tile_ht;
      int lpadding, rpadding, upadding, dpadding;
    } layer_arg_t;

    void set_kernel_arg( layer_arg_t& arg,
        int fmap_wid, int fmap_ht, int fmap_dep, 
        int fil_wid, int fil_ht, int num_fil, 
        int pool_wid, int pool_ht, int pool_stride, 
        int conv_stride, int tile_wid, int tile_ht,
        int lpadding, int rpadding, int upadding, int dpadding
        ) {
      arg.fmap_wid = fmap_wid; arg.fmap_ht = fmap_ht; arg.fmap_dep = fmap_dep;
      arg.fil_wid = fil_wid; arg.fil_ht = fil_ht; arg.num_fil = num_fil;
      arg.pool_wid = pool_wid; arg.pool_ht = pool_ht; arg.pool_stride = pool_stride;
      arg.conv_stride = conv_stride; arg.tile_wid = tile_wid; arg.tile_ht = tile_ht;
      arg.lpadding = lpadding; arg.rpadding = rpadding;
      arg.upadding = upadding; arg.dpadding = dpadding;
    }

    void exec_layer( const layer_arg_t& arg, const cl::Buffer& buf_fmap, 
                                                   cl::Buffer& buf_out,
                                             const cl::Buffer& buf_wts, int use_relu ) {
      assert(arg.fmap_dep >= BASE_PER_OBUS);
      assert(arg.num_fil  >= BASE_PER_OBUS);

      // calculate aligned size for a fmap
      const int fmap_nblk = arg.fmap_dep/BASE_PER_OBUS;
      const int row_ddr_size = ceil( (float)(arg.fmap_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS ); 

      const int fil_dep  = arg.fmap_dep;
      const int fil_size = arg.fil_wid * arg.fil_ht * fil_dep;
      
      // calculate aligned size of a filter
      const int fil_ddr_size = ceil( (float)(fil_size)/BASE_PER_DDRBUS );
      const int fil_ailgned_size = fil_ddr_size * BASE_PER_DDRBUS;
      const int fil_fb_size = fil_size/BASE_PER_FBUS;
      const int wts_size = fil_ailgned_size * arg.num_fil;
      const int wts_ddr_size = wts_size/BASE_PER_DDRBUS;

      const int n_iter = arg.num_fil/NUM_FIL_BUF;

      // output fmap after conv
      const int ofmap_wid     = ceil( (float)(arg.fmap_wid)/arg.conv_stride );
      const int ofmap_ht      = ceil( (float)(arg.fmap_ht )/arg.conv_stride );
      const int ofmap_dep     = arg.num_fil;
      const int ofmap_nblk    = ofmap_dep/BASE_PER_OBUS;
      const int ofmap_ob_size = ofmap_wid * ofmap_ht * ofmap_dep/BASE_PER_OBUS;

      const int ofmap_ddr_wid = ceil( (float)(ofmap_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS ); 
        
      // pooling output - valid padding
      // Not sure what will happen if we need to drop some pixels...
      const int out_wid     = (ofmap_wid - arg.pool_wid) / arg.pool_stride + 1;
      const int out_ht      = (ofmap_ht  - arg.pool_ht ) / arg.pool_stride + 1;
      const int out_dep     = ofmap_dep;
      const int out_nblk    = ofmap_dep/BASE_PER_OBUS;
      const int out_db_wid  = ceil( (float)(out_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS );
      const int out_ob_size = out_nblk * out_wid * out_ht;

      const int pool_size     = arg.pool_ht  * arg.pool_wid;
      const int num_ddr_trans = out_nblk * out_ht * out_db_wid;

      // compute krnl
      const int fil_wnd_per_blk = out_wid * out_ht * pool_size;
      const int num_fil_wnd = out_wid * out_ht * out_nblk * pool_size;

      std::cout << "\nConfigurations:  " << std::endl 
                << "  systolic array : " << SYS_WID  << " x " << SYS_HT << std::endl
                << "         out_bus : " << BASE_PER_OBUS   << " x " << sizeof(base) << "B" << std::endl
                << "         fmap_bus: " << BASE_PER_FBUS   << " x " << sizeof(base) << "B" << std::endl
                << "         ddr_bus : " << BASE_PER_DDRBUS << " x " << sizeof(base) << "B" << std::endl;  

      std::cout << "\nInput feature map: " << std::endl
                << "             size: " << arg.fmap_wid << " x " << arg.fmap_ht << " x " << arg.fmap_dep << std::endl 
                << "             nblk: " << fmap_nblk << std::endl
                << "  ddr_bus per row: " << row_ddr_size << std::endl;
      
      std::cout << "\nFilters:" << std::endl
                << "             num_fil: " << arg.num_fil << std::endl 
                << "               niter: " << n_iter  << std::endl
                << "                size: " << arg.fil_wid << " x " << arg.fil_ht << " x " << fil_dep << std::endl
                << "  ddr_bus per filter: " << fil_ddr_size << std::endl;
      
      std::cout << "\nOut fmap:" << std::endl 
                << "             size: " << ofmap_wid << " x " << ofmap_ht << " x " << ofmap_dep << std::endl
                << "             nblk: " << ofmap_nblk << std::endl
                << "  ddr_bus per row: " << ofmap_ddr_wid << std::endl;
      
      std::cout << "\nCompute:" << std::endl
                << "        #fil_wnd: " << fil_wnd_per_blk << std::endl
                << "  fil_size in fb: " << fil_fb_size << std::endl;

      std::cout << "\nReLU:" << std::endl
                << "  fil_size in fmap_bus: " << fil_fb_size << std::endl
                << "  ofmap_size in outbus: " << ofmap_ob_size << std::endl;
      
      std::cout << "\nPooling:" << std::endl
                << "  pool_size in out_bus: " << pool_size << std::endl
                << "     output in out_bus: " << out_ob_size << std::endl
                << "     output size      : " << out_wid  << " x " << out_ht << " x " << out_dep << std::endl
                << "     output nblk      : " << out_nblk << std::endl
                << "     #DDR transactions: " << num_ddr_trans << std::endl; 

      std::cout << std::endl;

      //set the kernel Arguments
      std::cout << "Setting kernel arguments..." << std::endl;
      krnl_load_fmap->setArg( 0, buf_fmap    ); // fmap ptr
      krnl_load_fmap->setArg( 1, arg.fmap_wid    ); // fmap_wid
      krnl_load_fmap->setArg( 2, arg.fmap_ht     ); // fmap_ht
      krnl_load_fmap->setArg( 3, fmap_nblk   ); // fmap_nblk
      krnl_load_fmap->setArg( 4, arg.fil_wid     ); // fil_wid
      krnl_load_fmap->setArg( 5, arg.fil_ht      ); // fil_ht
      krnl_load_fmap->setArg( 6, arg.tile_wid     ); // tile_wid
      krnl_load_fmap->setArg( 7, arg.tile_ht      ); // tile_ht
      krnl_load_fmap->setArg( 8, arg.lpadding     ); // lpadding
      krnl_load_fmap->setArg( 9, arg.rpadding     ); // rpadding
      krnl_load_fmap->setArg(10, arg.upadding     ); // upadding
      krnl_load_fmap->setArg(11, arg.dpadding     ); // dpadding
      krnl_load_fmap->setArg(12, arg.pool_wid    ); // pool_wid
      krnl_load_fmap->setArg(13, arg.pool_ht     ); // pool_ht
      krnl_load_fmap->setArg(14, arg.pool_stride ); // pool_stride
      krnl_load_fmap->setArg(15, arg.conv_stride); // stride
      krnl_load_fmap->setArg(16, n_iter     ); // niter 

      krnl_load_wts->setArg(0, buf_wts     );
      krnl_load_wts->setArg(1, wts_ddr_size);

      krnl_conv->setArg(0, out_wid    ); // out_wid
      krnl_conv->setArg(1, out_ht     ); // out_ht 
      krnl_conv->setArg(2, pool_size  ); // pool_size
      krnl_conv->setArg(3, fil_fb_size); // fil_size
      krnl_conv->setArg(4, n_iter     ); // n_iter

      krnl_relu->setArg(0, fil_fb_size); // fil_size
      krnl_relu->setArg(1, num_fil_wnd); // o_size
      krnl_relu->setArg(2, use_relu); // use_relu

      krnl_pool->setArg(0, buf_out      ); // o_fmap
      krnl_pool->setArg(1, pool_size    ); // pool_size
      krnl_pool->setArg(2, out_wid      ); // o_wid
      krnl_pool->setArg(3, out_ob_size  ); // o_size
      krnl_pool->setArg(4, num_ddr_trans); // n_trans

      // Launch the Kernel
      q->enqueueTask(*krnl_load_fmap);
      q->enqueueTask(*krnl_load_wts );
      q->enqueueTask(*krnl_conv     );
      q->enqueueTask(*krnl_relu     );
      q->enqueueTask(*krnl_pool     );
      q->finish();
    }

    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;

    cl::Context* context;
    cl::CommandQueue* q;
    char *buf;

    cl::Program::Binaries bins;
    cl::Program* program;

    cl::Kernel *krnl_load_fmap, *krnl_load_wts, *krnl_conv, *krnl_relu, *krnl_pool;

    std::vector<base, aligned_allocator<base>> dummy_vec0, dummy_vec1;
    cl::Buffer *mid_buf0, *mid_buf1;

    std::vector<std::vector<base, aligned_allocator<base>>*> wts_vec;
    std::vector<cl::Buffer*> wts_buf_vec;
};
