
#===============================================================================
# Host Area
#===============================================================================
	# Specify source files 
	# HOST_SRCS   : source files for host
	# HOST_HDRS   : header files for host
	# HOST_EXE_DIR: directory to place the compiled exe object
	# HOST_EXE    : name for the compiiled exe 
HOST_SRCS = host/host.cpp
HOST_HDRS = host/*.h
HOST_EXE_DIR= .
HOST_EXE = bin_host

	# Specify flags for compiler and linker 
	# HOST_CFLAGS : compiler flags
	# HOST_LFLAGS : linker flags
HOST_CFLAGS =  -g -Wall -DFPGA_DEVICE -DC_KERNEL 
HOST_CFLAGS += -DTARGET_DEVICE=\"${XDEVICE}\"
HOST_CFLAGS += -I${XILINX_SDX}/runtime/include/1_2 -std=c++0x

HOST_LFLAGS =  -L${XILINX_SDX}/runtime/lib/x86_64 -lxilinxopencl -lrt -pthread

#===============================================================================
# Kernel Area
#===============================================================================
	#>CL Kernel Template:
	# base_SRCS   = kernel source file (.cl)
	# base_HDRS   = kernel header files(.h )
	# base_CFLAGS = compiler flags specially for this kernel
	# base_LFLAGS = linker   flags specially for this kernel
	#
	#>RTL Kernel Template:
	# base_HDLSRCS = kernel source files (.v .xml .tcl)
	# base_TCL     = tcl script for generating xo object

conv_SRCS   = krnl/*.cl
conv_HDRS   = krnl/*.h
conv_CFLAGS = -g --memory_port_data_width all:512 #--max_memory_ports all 
conv_CFLAGS += -I krnl
conv_LFLAGS =

# Not used for pure HLS design
sys_array_HDLSRCS = hdl/*.v kernel.xml scripts/package_kernel.tcl scripts/gen_xo.tcl
sys_array_TCL     = scripts/gen_xo.tcl

#===============================================================================
# xos and xclbin
#===============================================================================
CLXOS = conv
RTLXOS = 
XOS = $(CLXOS) $(RTLXOS)

XCLBIN_NAME   =bin_conv
XCLBIN_DIR    =xclbin
XCLBIN_LFLAGS =#--xp param:compiler.useHlsGpp=1#--optimize 3

#===============================================================================
# Device 
#===============================================================================
# Available devices for ece-linux are:
#  - xilinx_kcu1500_dynamic_5_0
#  - xilinx_vcu1525_dynamic_5_1
#  - zc702
#  - zc706
#  - zcu102
#  - zcu104
#  - zcu106
#  - zed
#XDEVICE=xilinx_adm-pcie-8k5_2ddr_4_0
XDEVICE=xilinx_kcu1500_dynamic_5_0
include ../../util/common.mk
