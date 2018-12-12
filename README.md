# fp-cnn
An end-to-end framework to map a CNN (TensorFlow) onto FPGA (OpenCL).   
Link to our repo: https://github.com/yo96/fp-cnn.

## How to generate the accelerator
- Specify the network configuration

   The network configuration file is `model/<your_model_name>/nn_configs.txt` where `<your_model_name>` is the 
   name of the model. We have provided a reference network configuration for MNIST under `model/MNIST`.
   
   `nn_configs.txt` should be generated by our Tensorflow API wrapper. During our project, we used the `nn-quant`
   library from Ritchie to build the model and dump weights, therefore we did not include that part in this repo. 
   We will provide the wrapper after nn-quant, which is currently private, becomes public. 
   
- Provide the weights file
   
   The network configuration file is `model/<your_model_name>/_dump.npy` where `<your_model_name>` is the 
   name of the model. We have provided a reference binary weight file for MNIST under `model/MNIST`.
   
   `_dump.npy` should be generated by `--save-activations` option of the nn-quant library. 
   It is basically a dictionary of numpy arrays. The key for an array that stores the weights for a specific layer 
   should end with '/w'. For exampe, `conv1/w`, `last/w` are valid keys for weights.
   During our project, we got the permission from Ritchie to use this library but it is private. You can access
   this library after it becomes public.
   
   **Please note that due to CMSX file size limit, we cannot upload the _dump.npy file to the system. You can download**
   **the complete project at https://github.com/yo96/fp-cnn.**
   
- Run the generation framework
 
    `source utils/gen.sh`
    
    The above command will generate a new SDAccel project under `src/<you_model_name>/`. All files should be ready
    and you can directly run software/hardware emulation there.

## Other projects in this repo    
under `src` directory there are also a number of projects that we created when we are developing this framework:

 - `conv_example`: an example project that contains only functional-level kernels.
 - `conv`: initial project created when developing the compute kernel. It has functional-level `load_fmap`, `load_wts`, and a 
 dummy `output` kernel.
 - `load_fmap`: initial project created when developing the `load_fmap` kernel. It has a dummy `output` kernel.
 - `conv_pool`: a project that tests the integration of `conv`, `acc_relu`, and `pooling` kernels.
 - `conv_pool_test`: a project that tests the integration of `load_fmap`, `conv`,`acc_relu`, and `pooling` kernels.
 - `verif_conv`: a projcet that verifies the device result of a conv layer against the intermediate result directly dumped 
 from tensorflow.
 - `verif_fc`: a projcet that verifies the device result of a fully-connected layer against the intermediate result
 directly dumped from tensorflow.
 - `vadd`: a hello world SDAccel project.
 - `measure_mem`: a testing project that measures the effective DDR bandwidth.
 
## How to use the makefile inside the SDAccel project directory
- Software emulation   
~~~
make -f sdaccel.mk run_cpu_em
~~~

- Hardware emulation   
~~~
make -f sdaccel.mk run_hw_emu
~~~

- Run on board   
~~~
make -f sdaccel.mk run_hw
~~~

- Clean   
~~~
make -f sdaccel.mk cleanall
~~~
