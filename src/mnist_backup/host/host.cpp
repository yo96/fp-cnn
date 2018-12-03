#include "cnn_xcel.h"
#include "wts.h"

int arg_max( const std::vector<base, aligned_allocator<base>>& vec, size_t size ) {
  int _idx = 0;
  base _max = vec[0];

  for ( int i = 1; i < size; ++i ) {
    if ( vec[i] > _max ) {
      _max = vec[i];
      _idx = i;
    }
  }

  return _idx;
}

int main(int argc, char* argv[]) {
  int total = 50, match = 0;

  CnnAccelerator cnnxcel;

  int ref_res, dev_res;
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

  for ( int img_idx = 0; img_idx < total; img_idx++ ) {
    

    // Takes the first image of the whole batch of input feature maps
    std::vector<base,aligned_allocator<base>> src_fmap(conv1_in[img_idx], conv1_in[img_idx] + _conv1_in_size);
    std::vector<base,aligned_allocator<base>> src_out (out_size,  0);
    std::vector<base,aligned_allocator<base>> ref (last_out[img_idx], last_out[img_idx] + _last_out_size);

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
      //std::cout << "cpu[" << i << "] = " << ref[i] 
                //<< " dev[" << i << "] = " << src_out[i] 
                //<< std::endl; 
    }

    rmse = rmse/out_size;
    src_mean = src_mean/out_size;
    dev_mean = dev_mean/out_size;

    std::cout << "device out mean:" << dev_mean << std::endl
              << "   ref out mean:" << src_mean << std::endl
              << "          error:" << rmse     << std::endl;

    ref_res = arg_max( ref, out_size );
    dev_res = arg_max( src_out, out_size );
    if ( ref_res == dev_res )
      ++match;

    std::cout << "TF predicted result = " << ref_res << std::endl;
    std::cout << "Device predicted result = " << dev_res << std::endl;
  }

  std::cout << "Accuracy = " << float(match)/total << std::endl;

  printf("Done!\n");

  return EXIT_SUCCESS;
}
