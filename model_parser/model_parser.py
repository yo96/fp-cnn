# model_parser.py
# Takes the nn_configs.txt and _dump.npy file as input
# Generate a new SDAccel project that contains the CNN accelerator

# Model parser will parse the nn_configs.txt file into an IR that represents how the OpenCL kernels are configured (the arguments)
# It also generates the weight file that are ready to be loaded into the host DRAM

# TODO List:
# 0. Wrappers for Tensorflow APIs
# 1. Generate correct _dump.npy and nn_configs.txt
# 2. Parse nn_configs.txt and calculate the parameters (use simple parameters for now, write to defs.h and configs.h)
# 3. Generate the IR (determine the arguments for each kernel)
# 4. Generate the driver code (cnn_xcel.h which contains initialize() and runInference())
# 5. Pad _dump.npy and dump it into another weight file so that host.cpp can easily load it
