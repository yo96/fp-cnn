#include "cnn_xcel.h"
#include "wts.h"

int main(int argc, char* argv[]) {
  class CnnAccelerator cnnxcel;

  const int fmap_size = 28*28*32;
  const int out_size  = 1*1*32;
  size_t fmap_Bsize = fmap_size * sizeof(base);
  size_t out_Bsize  = out_size  * sizeof(base);

  std::cout << "Intializing CnnAccelerator..." << std::endl;

  cnnxcel.initialize( argc, argv );

  cnnxcel.load_wts( 0, conv1_w, _conv1_w_size );
  cnnxcel.load_wts( 1, conv2_w, _conv2_w_size );
  cnnxcel.load_wts( 2, fc1_w, _fc1_w_size );
  cnnxcel.load_wts( 3, last_w, _last_w_size );

  std::vector<base,aligned_allocator<base>> src_fmap(conv1_in[0], conv1_in[0] + _conv1_in_size);
  std::vector<base,aligned_allocator<base>> src_out (out_size,  0);
  std::vector<base,aligned_allocator<base>> ref (last_out[0], last_out[0] + _last_out_size);

  // Initiallize fmap
  //for (int i=0;i<fmap_size;i++)
    //src_fmap[i] = (rand() % 10);

  cl::Buffer* buf_fmap = cnnxcel.create_buffer( CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
      fmap_Bsize, src_fmap.data()
    );

  cl::Buffer* buf_out = cnnxcel.create_buffer( CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
      out_Bsize, src_out.data()
    );

  std::cout << "Running inference..." << std::endl;

  cnnxcel.run_inference( *buf_fmap, *buf_out );

  double src_mean = 0;
  double dev_mean = 0;
  double rmse = 0;

  for ( int i=0;i<out_size;i++){
    rmse += sqrt( (src_out[i]-ref[i])*(src_out[i]-ref[i]) );
    src_mean += ref[i];
    dev_mean += src_out[i];
    std::cout << "cpu[" << i << "] = " << ref[i] 
              << " dev[" << i << "] = " << src_out[i] 
              << std::endl; 

  }
  rmse = rmse/out_size;
  src_mean = src_mean/out_size;
  dev_mean = dev_mean/out_size;

  std::cout << "device out mean:" << dev_mean << std::endl
            << "   ref out mean:" << src_mean << std::endl
            << "          error:" << rmse     << std::endl;

  printf("Done!\n");

  return EXIT_SUCCESS;
}
