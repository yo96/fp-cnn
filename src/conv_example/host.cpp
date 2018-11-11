#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "allocator.h"

//TARGET_DEVICE macro needs to be passed from gcc command line
#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
    #define STR_VALUE(arg)      #arg
    #define GET_STRING(name) STR_VALUE(name)
    #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

// parameters
static const int in_fmap_wid  = 56;
static const int in_fmap_ht   = 56;
static const int in_fmap_depth= 64;
static const int padding      = 1;
static const int filter_wid   = 3;
static const int filter_ht    = 3;
static const int stride       = 2;
static const int n_filter     = 256;

static const int out_fmap_wid  = (in_fmap_wid+padding*2-filter_wid+stride)/stride;
static const int out_fmap_ht   = (in_fmap_ht +padding*2-filter_ht +stride)/stride;
static const int out_fmap_depth= n_filter;

static const int n_iterations  = n_filter/64;

// this size means number of elements in image and weight
static const int IMAGE_SIZE  = in_fmap_wid * in_fmap_ht * in_fmap_depth;
static const int WEIGHT_SIZE = filter_wid * filter_ht * in_fmap_depth * n_filter;
static const int OUTPUT_SIZE = out_fmap_wid * out_fmap_ht * n_filter;

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
    
    // Compute the size of array in bytes
    size_t image_Bsize  = IMAGE_SIZE  * sizeof(short);
    size_t weight_Bsize = WEIGHT_SIZE * sizeof(short);
    size_t output_Bsize = OUTPUT_SIZE * sizeof(short);

    /*********************************************************************************
     * Prepare data
     *********************************************************************************/
    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary
    std::vector<short,aligned_allocator<short>> source_image  (IMAGE_SIZE , 0);
    std::vector<short,aligned_allocator<short>> source_weight (WEIGHT_SIZE, 0);
    std::vector<short,aligned_allocator<short>> source_results(OUTPUT_SIZE, 0);
    
    short std_result[OUTPUT_SIZE];  
    srand(time(NULL));
    // prepare data
    for(int i=0;i<in_fmap_wid;i++){
      for(int j=0;j<in_fmap_ht;j++){
        for(int k=0;k<in_fmap_depth;k++){
          source_image[i*in_fmap_ht*in_fmap_depth+j*in_fmap_depth+k] = 1; //(k==0 || k==32? i : 0);//(rand() % 10) - 5;//
        }
      }
    }

    // prepare wts
    int one_filter_size = filter_wid*filter_ht*(in_fmap_depth/32); // in short32
    for (int i=0;i<WEIGHT_SIZE;i+=one_filter_size*32){
      for(int j=0;j<one_filter_size;j++){
        for(int k=0;k<32;k++){
          source_weight[i+j*32+k] = 1;//(k==0? j : 0);//(rand() % 10) - 5; //          
        }
      }
    }
    
    //// Calculate Cpu Result ////
    // re-arrange image data
    short img_tmp[in_fmap_wid+padding*2][in_fmap_ht+padding*2][in_fmap_depth] = {0};
    // image data
    for(int i=0;i<in_fmap_wid;i++){
      for(int j=0;j<in_fmap_ht;j++){
        for(int k=0;k<in_fmap_depth;k++){
          img_tmp[i+1][j+1][k] = source_image[i*in_fmap_ht*in_fmap_depth+j*in_fmap_depth+k];
        }
      }
    }

    for(int n=0;n<n_filter;n++){
      short w8_tmp[filter_wid][filter_ht][in_fmap_depth];
      int cur_idx = n*filter_wid*filter_ht*in_fmap_depth;
      // fetch a filter 
      for(int i=0;i<in_fmap_depth;i++){
        for(int j=0;j<filter_ht;j++){
          for(int k=0;k<filter_wid;k++){
            w8_tmp[k][j][i] = source_weight[cur_idx+k*filter_ht*in_fmap_depth+j*in_fmap_depth+i];
          }
        }
      }

      // convolve
      int n_block = n/64;
      int n_layer = n%64;
      int wid_max = in_fmap_wid+padding*2-filter_wid;
      int ht_max  = in_fmap_ht +padding*2-filter_ht;

      int out_x = 0;
      int out_y = 0;
      for(int i=0;i<=ht_max;i+=stride){
        for(int j=0;j<=wid_max;j+=stride){
          short ret = 0;
          for(int k=0;k<in_fmap_depth;k++){
            // element-wise mult-acc with the filter
            for(int m=0;m<filter_wid;m++){
              for(int n=0;n<filter_ht;n++){
                ret += img_tmp[i+m][j+n][k] * w8_tmp[m][n][k];
              }
            }
          }
          std_result[n_block*out_fmap_wid*out_fmap_ht*64 + n_layer + out_x*out_fmap_ht*64 + out_y*64] = ret;
          out_x++;
        }
        out_x = 0;
        out_y++;
      }
    }

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

    /******************** HOST CODE AREA  ********************/
    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
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
    cl::Kernel krnl_load_img     (program,"load_img"        );
    cl::Kernel krnl_load_weights (program,"load_weights"    );
    cl::Kernel krnl_load_output  (program,"load_output");
    cl::Kernel krnl_convolve     (program,"convolve"        );
    
    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    cl::Buffer buffer_img(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  
            image_Bsize, source_image.data());
    cl::Buffer buffer_w8 (context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  
            weight_Bsize, source_weight.data());
    cl::Buffer buffer_result(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            output_Bsize, source_results.data());
    
    // Data will be transferred from system memory over PCIe to the FPGA on-board
    // DDR memory.
    q.enqueueMigrateMemObjects({buffer_img,buffer_w8},0/* 0 means from host*/);
    q.finish();
    
    //set the kernel Arguments
    krnl_load_img.setArg(0,buffer_img    );
    krnl_load_img.setArg(1,in_fmap_wid   ); // fmap_wid
    krnl_load_img.setArg(2,in_fmap_ht    ); // fmap_ht
    krnl_load_img.setArg(3,in_fmap_depth ); // fmap_wid
    krnl_load_img.setArg(4,filter_wid    ); // filter_wid
    krnl_load_img.setArg(5,padding       ); // padding
    krnl_load_img.setArg(6,stride        ); // stride
    krnl_load_img.setArg(7,n_iterations  ); // n_iterations

    krnl_load_weights.setArg(0,buffer_w8);
    krnl_load_weights.setArg(1,WEIGHT_SIZE/32);

    krnl_convolve.setArg( 0,out_fmap_wid  ); // fmap_wid
    krnl_convolve.setArg( 1,out_fmap_ht   ); // fmap_ht
    krnl_convolve.setArg( 2,n_filter      ); // n_filter
    krnl_convolve.setArg( 3,filter_wid    ); // filter_wid
    krnl_convolve.setArg( 4,n_iterations  ); // n_iter

    krnl_load_output.setArg(0, buffer_result);
    krnl_load_output.setArg(1, OUTPUT_SIZE/32);   

    //Launch the Kernel
    std::cout << "Launching kernel..." <<std::endl;
    q.enqueueTask(krnl_load_img    );
    q.enqueueTask(krnl_load_weights);
    q.enqueueTask(krnl_convolve    );
    q.enqueueTask(krnl_load_output );    
    q.finish();
    std::cout << "Execution complete..." << std::endl;
    std::cout << "Output feature map size: " << out_fmap_wid << " x " 
                                             << out_fmap_ht  << " x "
                                             << out_fmap_depth << std::endl;

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();
    std::cout << "Verifying results..." << std::endl;
    //Verify the result
    int match = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int host_result = std_result[i];
        if (source_results[i] != host_result) {
            if(i<OUTPUT_SIZE/3 && i%64==0)
                printf(error_message.c_str(), i, host_result, source_results[i]);
            match = 1;
            //break;
        }
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl; 
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);

}
