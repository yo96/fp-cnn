#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "allocator.h"
#include <CL/cl_ext.h>

//TARGET_DEVICE macro needs to be passed from gcc command line
#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
    #define STR_VALUE(arg)      #arg
    #define GET_STRING(name) STR_VALUE(name)
    #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

#define BILLION 1000000000L

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

    /************************************************************************* 
     * HOST CODE AREA  
     ************************************************************************/
    // Set it to be 4KB so that hw_emu run faster. 
    // Change this to a large number when running on board
    const int DATA_SIZE = 1024; 
    size_t read_Bsize = DATA_SIZE * sizeof(int);
    std::vector<int,aligned_allocator<int>> source_data(DATA_SIZE, 1);

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | 
                                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    // Load xclbin 
    std::cout << "Loading: '" << xclbinFilename << "'\n";
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
    cl::Kernel krnl_load (program, "load_data");
    cl::Kernel krnl_dummy(program, "dummy");
    
    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    cl_mem_ext_ptr_t input_ext;
    input_ext.flags = XCL_MEM_DDR_BANK1;
    input_ext.obj = source_data.data();
    input_ext.param = 0;

    cl::Buffer buffer_data(context, CL_MEM_USE_HOST_PTR   | CL_MEM_READ_ONLY,
                           read_Bsize, &input_ext, NULL);
    //cl_mem buffer_data = clCreateBuffer(
    //                      context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
    //                               CL_MEM_EXT_PTR_XILINX,  
    //                      read_Bsize, input_ext);
    
    // Data will be transferred from system memory over PCIe to the FPGA on-board
    // DDR memory.
    q.enqueueMigrateMemObjects({buffer_data},0); /* 0 means from host*/
    q.finish();
    //set the kernel Arguments
    krnl_load.setArg(0, buffer_data);
    krnl_load.setArg(1, DATA_SIZE);

    krnl_dummy.setArg(0, DATA_SIZE);
    // Launch the Kernel
    // TODO: time this?
    struct timespec start, end;
    double time;
    double bw;
    
    std::cout << "Launching kernel..." <<std::endl;

    clock_gettime(CLOCK_MONOTONIC, &start);

    q.enqueueTask(krnl_load  );
    q.enqueueTask(krnl_dummy );
    q.finish();

    clock_gettime(CLOCK_MONOTONIC, &end);
    
    time = BILLION * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    time = time / BILLION;
    bw = read_Bsize / time / BILLION;
    std::cout << "Execution complete..." << std::endl
              << "Elapsed time:" << time*1000 << "ms" << std::endl
              << std::setprecision(2)
              << "DDR Bandwidth (read):" << bw << " GB/s" << std::endl;

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    return 0;

}
