#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <CL/cl_ext.h>
#include <math.h>
#include "allocator.h"
#include "krnl/configs.h"
#include "test.h"
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

int main(int argc, char* argv[]) {

    //TARGET_DEVICE macro needs to be passed from gcc command line
    const char *target_device_name = TARGET_DEVICE;

    if(argc != 2) {
        std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
        return EXIT_FAILURE;
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
       return EXIT_FAILURE; 
    }

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | 
                                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
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
    cl::Kernel krnl_load_fmap(program, "load_fmap");
    cl::Kernel krnl_load_wts (program, "load_wts" );
    cl::Kernel krnl_conv     (program, "compute"  );
    cl::Kernel krnl_output   (program, "load_out" );    
    

    /************************************************************************* 
     * HOST CODE AREA  
     ************************************************************************/
    const int fmap_wid  = 7;
    const int fmap_ht   = 7;
    const int fmap_dep  = 32;
    //const int fmap_size = fmap_wid * fmap_ht * fmap_dep;
    // calculate aligned size for a fmap
    const int fmap_nblk = fmap_dep/BASE_PER_OBUS;
    //const int blk_dep   = fmap_dep/BASE_PER_FBUS;
    const int row_ddr_size = ceil( (float)(fmap_wid * BASE_PER_OBUS)/BASE_PER_DDRBUS ); 
    const int fmap_size = row_ddr_size * BASE_PER_DDRBUS * fmap_ht * fmap_nblk;

    const int num_fil  = 32;
    const int fil_ht   = 3;
    const int fil_wid  = 3;
    const int fil_dep  = fmap_dep;
    const int fil_size = fil_wid * fil_ht * fil_dep;
    // calculate aligned size of a filter
    const int fil_ddr_size = ceil( (float)(fil_size)/BASE_PER_DDRBUS );
    const int fil_ailgned_size = fil_ddr_size * BASE_PER_DDRBUS;
    const int wts_size = fil_ailgned_size * num_fil;

    const int conv_stride = 1;
    const int padding  = 1; // to be changed 
    const int n_iter   = num_fil/NUM_FIL_BUF;

    const int ofmap_wid   = ceil( (float)(fmap_wid)/conv_stride );
    const int ofmap_ht    = ceil( (float)(fmap_ht )/conv_stride );
    const int ofmap_dep   = num_fil;
    const int ofmap_nblk  = ofmap_dep/BASE_PER_OBUS;
    const int ofmap_size  = ofmap_wid * ofmap_ht * ofmap_dep;
    
    size_t fmap_Bsize = fmap_size  * sizeof(base);
    size_t wts_Bsize  = wts_size   * sizeof(base);
    size_t out_Bsize  = ofmap_size * sizeof(base);

    std::vector<base,aligned_allocator<base>> src_fmap(fmap_size,  1);
    std::vector<base,aligned_allocator<base>> src_wts (wts_size,   1);
    std::vector<base,aligned_allocator<base>> src_out (ofmap_size, 0);
    std::vector<base,aligned_allocator<base>> ref     (ofmap_size, 0);

    // Initiallize fmap
    for (int i=0;i<fmap_size;i++)
      src_fmap[i] = (rand() % 10) - 6;

    for (int i=0;i<wts_size;i++)
      src_wts[i] = (rand() %10) - 5;
    // Get CPU result
    test_conv<base>(src_fmap.data(), src_wts.data(), ref.data(),
              fmap_wid, fmap_ht, fmap_dep,
              fil_wid,  fil_ht, num_fil,
              conv_stride, padding );

    // produces weird bugs if using multiple DDR...
    //cl_mem_ext_ptr_t ext_a;
    //ext_a.flags = XCL_MEM_DDR_BANK0;
    //ext_a.obj   = source_a.data();
    //ext_a.param = 0;

    //cl_mem_ext_ptr_t ext_b;
    //ext_b.flags = XCL_MEM_DDR_BANK0;
    //ext_b.obj   = source_b.data();
    //ext_b.param = 0;

    //cl_mem_ext_ptr_t ext_c;
    //ext_b.flags = XCL_MEM_DDR_BANK1;
    //ext_b.obj   = source_c.data();
    //ext_b.param = 0;

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
    krnl_load_fmap.setArg(0, buf_fmap   ); // fmap ptr
    krnl_load_fmap.setArg(1, fmap_wid   ); // fmap_wid
    krnl_load_fmap.setArg(2, fmap_ht    ); // fmap_ht
    krnl_load_fmap.setArg(3, fmap_nblk  ); // fmap_nblk
    krnl_load_fmap.setArg(4, fil_wid    ); // fil_wid
    krnl_load_fmap.setArg(5, fil_ht     ); // fil_ht
    krnl_load_fmap.setArg(6, padding    ); // padding
    krnl_load_fmap.setArg(7, conv_stride); // stride
    krnl_load_fmap.setArg(8, n_iter     ); // niter 

    const int wts_ddr_size = wts_size/BASE_PER_DDRBUS;
    krnl_load_wts.setArg(0, buf_wts     );
    krnl_load_wts.setArg(1, wts_ddr_size);

    const int fil_out_size = fil_size/BASE_PER_OBUS;
    krnl_conv.setArg(0, ofmap_wid   ); // o_wid
    krnl_conv.setArg(1, ofmap_ht    ); // o_ht 
    krnl_conv.setArg(2, ofmap_nblk  ); // o_blk
    krnl_conv.setArg(3, num_fil     ); // n_fil
    krnl_conv.setArg(4, fil_out_size); // fil_size
    krnl_conv.setArg(5, n_iter      ); // n_iter

    const int ofmap_out_size = ofmap_size/BASE_PER_OBUS;
    krnl_output.setArg(0, buf_out       ); // o_fmap ptr
    krnl_output.setArg(1, fil_out_size  ); // fil_size
    krnl_output.setArg(2, ofmap_out_size); // o_size
    // Launch the Kernel
    struct timespec start, end;
    double time;
    
    std::cout << "\nLaunching kernel..." <<std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);

    q.enqueueTask(krnl_load_fmap);
    q.enqueueTask(krnl_load_wts );
    q.enqueueTask(krnl_conv     );
    q.enqueueTask(krnl_output   );
    q.finish();

    clock_gettime(CLOCK_MONOTONIC, &end);


    time = BILLION * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    time = time / BILLION;
    std::cout << "Execution complete..." << std::endl
              << "Elapsed time:" << time*1000 << "ms" << std::endl << std::endl;


    std::cout << "Reading results from DDR..." << std::endl;
    q.enqueueMigrateMemObjects({buf_out},CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    std::cout << "Verifying results..." << std::endl;
    //for (int i=0;i<ofmap_size;i++){
    //  if (src_out[i] != 288 && src_out[i] != 128 && src_out[i] != 192){
    //    match = false;  
    //    std::cout << "src_out["<< i << "] =  "<< src_out[i] << std::endl;
    //    break;
    //  }
    //  else {
    //    //std::cout << i << ": "<< src_out[i] << std::endl;
    //  }
    //}

    bool match = true;
    match = true;
    std::cout << "Comparing CPU resutls..." << std::endl;
    for (int i=0;i<ofmap_size;i++){
      if (src_out[i] != ref[i]){
        std::cout << "cpu[" << i << "] = " << ref[i] 
                  << ", device[" << i << "] = " << src_out[i] << std::endl;
        match = false;
      }
    }
    std::cout << "Test " << (match ? "passed!" : "failed...") << std::endl;

    return 0;
}
