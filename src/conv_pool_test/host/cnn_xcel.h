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

typedef struct{
  int fmap_wid, fmap_ht, fmap_dep;
  int fil_wid, fil_ht, num_fil;
  int pool_wid, pool_ht, pool_stride;
  int conv_stride, tile_wid, tile_ht;
  int lpadding, rpadding, upadding, dpadding;
} layer_arg_t;

cl::CommandQueue q;
cl::Context context;
cl::Kernel krnl_load_fmap;
cl::Kernel krnl_load_wts;
cl::Kernel krnl_conv;
cl::Kernel krnl_relu;
cl::Kernel krnl_pool;

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

void exec_layer( const layer_arg_t& arg ) {
  assert(arg.fmap_dep >= BASE_PER_OBUS);
  assert(arg.num_fil  >= BASE_PER_OBUS);
  // calculate aligned size for a fmap
  const int fmap_nblk = arg.fmap_dep/BASE_PER_OBUS;
  const int row_ddr_size = ceil( (float)(arg.fmap_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS ); 
  const int fmap_size = row_ddr_size * BASE_PER_DDRBUS * arg.fmap_ht * fmap_nblk;

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
  //const int ofmap_size    = ofmap_nblk * ofmap_ht * ofmap_ddr_wid * BASE_PER_DDRBUS;
    
  // pooling output - valid padding
  // Not sure what will happen if we need to drop some pixels...
  const int out_wid     = (ofmap_wid - arg.pool_wid) / arg.pool_stride + 1;
  const int out_ht      = (ofmap_ht  - arg.pool_ht ) / arg.pool_stride + 1;
  const int out_dep     = ofmap_dep;
  const int out_nblk    = ofmap_dep/BASE_PER_OBUS;
  const int out_db_wid  = ceil( (float)(out_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS );
  const int out_ob_size = out_nblk * out_wid * out_ht;
  const int out_size    = out_nblk * out_ht * out_db_wid * BASE_PER_DDRBUS;

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

  //std::cout << "\nComp:" << std::end 
  //          << "  " << std::end;

  std::cout << std::endl;
  size_t fmap_Bsize = fmap_size * sizeof(base);
  size_t wts_Bsize  = wts_size  * sizeof(base);
  size_t out_Bsize  = out_size  * sizeof(base);

  std::vector<base,aligned_allocator<base>> src_fmap(fmap_size, 1);
  std::vector<base,aligned_allocator<base>> src_wts (wts_size,  1);
  std::vector<base,aligned_allocator<base>> src_out (out_size,  0);
  std::vector<base,aligned_allocator<base>> ref     (out_size,  0);

  // Initiallize fmap
  //for (int i=0;i<fmap_size;i++)
    //src_fmap[i] = (rand() % 10);

  //for (int i=0;i<wts_size;i++)
    //src_wts[i] = (rand() %10);
  
  // Get CPU result
  test_conv<base>(src_fmap.data(), src_wts.data(), ref.data(),
            arg.fmap_wid, arg.fmap_ht, arg.fmap_dep,
            arg.fil_wid,  arg.fil_ht, arg.num_fil,
            arg.pool_wid, arg.pool_ht, arg.pool_stride,
            arg.conv_stride, arg.lpadding );
  std::cout << "Allocating buffers..." << std::endl;
  cl::Buffer buf_fmap(context, CL_MEM_USE_HOST_PTR  | CL_MEM_READ_ONLY,
                      fmap_Bsize, src_fmap.data(), NULL);
  cl::Buffer buf_wts (context, CL_MEM_USE_HOST_PTR  | CL_MEM_READ_ONLY,
                      wts_Bsize,  src_wts.data(), NULL);
  cl::Buffer buf_out (context, CL_MEM_USE_HOST_PTR  | CL_MEM_WRITE_ONLY,
                      out_Bsize,  src_out.data(), NULL);

  // Data will be transferred from system memory over PCIe to the FPGA on-board
  // DDR memory.
  std::cout << "Transferring data to DDR..." << std::endl;
  q.enqueueMigrateMemObjects({buf_fmap,buf_out}, 0); /* 0 means from host*/
  q.finish(); 
  //set the kernel Arguments
  krnl_load_fmap.setArg( 0, buf_fmap    ); // fmap ptr
  krnl_load_fmap.setArg( 1, arg.fmap_wid    ); // fmap_wid
  krnl_load_fmap.setArg( 2, arg.fmap_ht     ); // fmap_ht
  krnl_load_fmap.setArg( 3, fmap_nblk   ); // fmap_nblk
  krnl_load_fmap.setArg( 4, arg.fil_wid     ); // fil_wid
  krnl_load_fmap.setArg( 5, arg.fil_ht      ); // fil_ht
  krnl_load_fmap.setArg( 6, arg.tile_wid     ); // tile_wid
  krnl_load_fmap.setArg( 7, arg.tile_ht      ); // tile_ht
  krnl_load_fmap.setArg( 8, arg.lpadding     ); // lpadding
  krnl_load_fmap.setArg( 9, arg.rpadding     ); // rpadding
  krnl_load_fmap.setArg(10, arg.upadding     ); // upadding
  krnl_load_fmap.setArg(11, arg.dpadding     ); // dpadding
  krnl_load_fmap.setArg(12, arg.pool_wid    ); // pool_wid
  krnl_load_fmap.setArg(13, arg.pool_ht     ); // pool_ht
  krnl_load_fmap.setArg(14, arg.pool_stride ); // pool_stride
  krnl_load_fmap.setArg(15, arg.conv_stride); // stride
  krnl_load_fmap.setArg(16, n_iter     ); // niter 

  krnl_load_wts.setArg(0, buf_wts     );
  krnl_load_wts.setArg(1, wts_ddr_size);

  krnl_conv.setArg(0, out_wid    ); // out_wid
  krnl_conv.setArg(1, out_ht     ); // out_ht 
  krnl_conv.setArg(2, pool_size  ); // pool_size
  krnl_conv.setArg(3, fil_fb_size); // fil_size
  krnl_conv.setArg(4, n_iter     ); // n_iter

  krnl_relu.setArg(0, fil_fb_size); // fil_size
  krnl_relu.setArg(1, num_fil_wnd); // o_size

  krnl_pool.setArg(0, buf_out      ); // o_fmap
  krnl_pool.setArg(1, pool_size    ); // pool_size
  krnl_pool.setArg(2, out_wid      ); // o_wid
  krnl_pool.setArg(3, out_ob_size  ); // o_size
  krnl_pool.setArg(4, num_ddr_trans); // n_trans
  // Launch the Kernel
  struct timespec start, end;
  double time;
  
  std::cout << "\nLaunching kernel..." <<std::endl;
  clock_gettime(CLOCK_MONOTONIC, &start);

  q.enqueueTask(krnl_load_fmap);
  q.enqueueTask(krnl_load_wts );
  q.enqueueTask(krnl_conv     );
  q.enqueueTask(krnl_relu     );
  q.enqueueTask(krnl_pool     );
  q.finish();

  clock_gettime(CLOCK_MONOTONIC, &end);

  time = BILLION * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
  time = time / BILLION;
  std::cout << "Execution complete..." << std::endl
            << "Elapsed time:" << time*1000 << "ms" << std::endl << std::endl;


  std::cout << "Reading results from DDR..." << std::endl;
  q.enqueueMigrateMemObjects({buf_out},CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();

  bool match = true;
  std::cout << "Verifying results..." << std::endl;

  match = true;
  std::cout << "Comparing CPU resutls..." << std::endl;
  for (int i=0;i<out_size;i++){
    if (src_out[i] != ref[i]){
      std::cout << "cpu[" << i << "] = " << ref[i] 
                << ", device[" << i << "] = " << src_out[i] << std::endl;
      match = false;
    }
  }
  std::cout << "Test " << (match ? "passed!" : "failed...") << std::endl;
}

void initialize( int argc, char* argv[] ) {
  //TARGET_DEVICE macro needs to be passed from gcc command line
  const char *target_device_name = TARGET_DEVICE;

  if(argc != 2) {
      std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
      return;
  }

  char* xclbinFilename = argv[1];
  
  std::vector<cl::Device> devices;
  cl::Device device;
  std::vector<cl::Platform> platforms;
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
  //cl::Context context(device);
  context = cl::Context(device);
  //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | 
                                      //CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | 
                                      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  // Load xclbin 
  std::cout << "\nLoading: '" << xclbinFilename << "'\n";
  std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
  bin_file.seekg (0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg (0, bin_file.beg);
  char *buf = new char [nb];
  bin_file.read(buf, nb);
  
  // Creating Program from Binary File
  cl::Program::Binaries bins;
  bins.push_back({buf,nb});
  devices.resize(1);
  cl::Program program(context, devices, bins);
  
  // This call will get the kernel object from program.
  std::cout << "Creating kernels..." << std::endl;
  //cl::Kernel krnl_load_fmap(program, "load_fmap");
  //cl::Kernel krnl_load_wts (program, "load_wts" );
  //cl::Kernel krnl_conv     (program, "compute"  );
  //cl::Kernel krnl_relu     (program, "acc_relu" );    
  //cl::Kernel krnl_pool     (program, "pool_wb"  );
  krnl_load_fmap = cl::Kernel(program, "load_fmap");
  krnl_load_wts  = cl::Kernel(program, "load_wts" );
  krnl_conv      = cl::Kernel(program, "compute"  );
  krnl_relu      = cl::Kernel(program, "acc_relu" );    
  krnl_pool      = cl::Kernel(program, "pool_wb"  );
}

void run_inference() {
  layer_arg_t* arg = new layer_arg_t();

  set_kernel_arg( *arg, 
      FMAP_WID_1, FMAP_HT_1, FMAP_DEP_1, 
      FIL_WID_1, FIL_HT_1, NUM_FIL_1, 
      POOL_WID_1, POOL_HT_1, POOL_STRIDE_1, 
      CONV_STRIDE_1, 
      LPADDING_1, RPADDING_1, UPADDING_1, DPADDING_1
      );
  exec_layer( *arg  );
}
