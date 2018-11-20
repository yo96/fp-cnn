# check environment variable
ifndef XILINX_SDX
$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)
endif

# CC  : compiler for host code
# CLCC: compiler for CL kernel code
#supported flow: cpu_emu, hw_emu, hw
CC = xcpp
CLCC = xocc

# Environment setup
# XDEVICE : Target device for XCLBIN
# BOARD_SETUP_FILE needs to point to setup.sh generated by xbinst command
# XDEVICE=xilinx:adm-pcie-7v3:1ddr:3.0 (old convention)
BOARD_SETUP_FILE=setup.sh
# XDEVICE=xilinx_adm-pcie-7v3_1ddr_3_0
ifndef XDEVICE
	XDEVICE=xilinx_adm-pcie-8k5_2ddr_4_0
endif

VIVADO=$(XILINX_VIVADO)/bin/vivado
#KERNEL_DEBUG=
#XDEVICE_REPO_PATH=

SDA_FLOW := cpu_emu
TARGET   := sw_emu
# CLCC_OPT : flags for kernel compiler(xocc)
# EMU_MODE : emulation mode (sw_emu | hw_emu)
# -s : Do not delete intermediate files
# -g : Generate code for debugging
ifeq (${SDA_FLOW},cpu_emu)
	TARGET = sw_emu
  CLCC_OPT += -t sw_emu
  XCLBIN = ${XCLBIN_NAME}_sw_emu.xclbin
else ifeq (${SDA_FLOW},hw_emu)
	TARGET = hw_emu
  CLCC_OPT += -t hw_emu
  XCLBIN = ${XCLBIN_NAME}_hw_emu.xclbin
else ifeq (${SDA_FLOW},hw)
	TARGET = hw
  CLCC_OPT += -t hw
  XCLBIN = ${XCLBIN_NAME}_hw.xclbin
endif

ifeq (${SDA_FLOW},cpu_emu)
	EMU_MODE=sw_emu
else ifeq (${SDA_FLOW},hw_emu)
	EMU_MODE=hw_emu
endif

CLCC_OPT += 
CLCC_OPT += $(CLCC_OPT_LEVEL) ${DEVICE_REPO_OPT} --platform ${XDEVICE} -o ${XCLBIN}
CLCC_OPT += -s

# specify compiling objects
OBJECTS := $(HOST_SRCS:.cpp=.o)
HOST_ARGS = ${XCLBIN_DIR}/${XCLBIN}

#====================================================================
#                          Helper Functions 
#====================================================================
CLFLAGS:= -I/$(XILINX_SDX)/Vivado_HLS/include/ \
--xp "param:compiler.preserveHlsOutput=1" \
--xp "param:compiler.generateExtraRunData=true" \
-s --platform ${XDEVICE}

# mk_clxo - create an xo from a set of cl kernel sources
#  CLC - kernel compiler to use
#  CLFLAGS - flags to pass to the compiler
#  $(1) - base name for this kernel
#  $(1)_SRCS - set of source kernel
#  $(1)_HDRS - set of header kernel
#  $(1)_CLFLAGS - set clflags per kernel 
#  $(2) - compilation target (i.e. hw, hw_emu, sw_emu), which is SDA_FLOW
define mk_clxo
$(XCLBIN_DIR)/$(1).$(2).xo: $($(1)_SRCS) $($(1)_HDRS) 
	mkdir -p $(XCLBIN_DIR)
	$(CLCC) -o xclbin/$(1).$(2).xo -c $(CLFLAGS) $($(1)_CFLAGS) -t $(TARGET) $($(1)_SRCS)
endef

# mk_rtlxo - create an xo from a tcl and RTL sources
#   VIVADO - version of Vivado to use
#   $(1) - base name for this kernel
#   $(1)_HDLSRCS - source files used in compilation
#   $(1)_TCL - tcl file to use for build
#   $(2) - target to build for
define mk_rtlxo
$(XCLBIN_DIR)/$(1).$(2).xo: $($(1)_HDLSRCS)
	mkdir -p $(XCLBIN_DIR)
	$(VIVADO) -mode batch -source $($(1)_TCL) -tclargs $(XCLBIN_DIR)/$(1).$(2).xo $(1) $(2) ${XDEVICE}
endef

# mk_xclbin - create an xclbin from a set of krnl sources
#  CLCC - kernel linker to use
#  XCLBIN_LFLAGS - flags to pass to the linker
#  $(1) - base name for this xclbin
#  $XOS - list of xos to link
#  $(2) - compilation target (i.e. hw, hw_emu, sw_emu)
define mk_xclbin
$(XCLBIN_DIR)/$(XCLBIN): $(addprefix $(XCLBIN_DIR)/,$(addsuffix .$(2).xo, $(XOS))) 
	mkdir -p ${XCLBIN_DIR}
	$(CLCC) -l $(CLFLAGS) $(XCLBIN_LFLAGS)  -o $$@\
			-t $(2) $(addprefix $(XCLBIN_DIR)/,$(addsuffix .$(2).xo, $(XOS)))
endef

####################### End of Helper Functions ######################

${HOST_EXE_DIR}/${HOST_EXE} : ${OBJECTS} ${HOST_HDRS}
	${CC} -o $@ ${OBJECTS} ${HOST_LFLAGS}  


# Rules to make CLXOs:
$(foreach clxo,$(CLXOS),$(eval $(call mk_clxo,$(clxo),$(TARGET))))
# Rules to make RTLXOs :
$(foreach rtlxo,$(RTLXOS),$(eval $(call mk_rtlxo,$(rtlxo),$(TARGET))))
# Rule to make xclbin
$(eval $(call mk_xclbin,$(XCLBIN),$(TARGET)))

%.o: %.cpp
	${CC}  -c $< -o $@ ${HOST_CFLAGS} 

# Commands
.PHONY: all

all: run

host: ${HOST_EXE_DIR}/${HOST_EXE}

xbin_cpu_em:
	make SDA_FLOW=cpu_emu xbin -f sdaccel.mk

xbin_hw_em:
	make SDA_FLOW=hw_emu xbin -f sdaccel.mk

xbin_hw :
	make SDA_FLOW=hw xbin -f sdaccel.mk

xos : $(addprefix $(XCLBIN_DIR)/,$(addsuffix .$(TARGET).xo, $(XOS)))
	@echo $(XOS)

xbin: $(XCLBIN_DIR)/${XCLBIN}

run_cpu_em: 
	make SDA_FLOW=cpu_emu run_em -f sdaccel.mk

run_hw_em: 
	make SDA_FLOW=hw_emu run_em -f sdaccel.mk

run_hw : 
	make SDA_FLOW=hw run_hw_int -f sdaccel.mk

run_em: xconfig host xbin
	XCL_EMULATION_MODE=${EMU_MODE} ${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}

run_hw_int : host xbin_hw
	source ${BOARD_SETUP_FILE};${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}

estimate : 
	${CLCC} -c -t hw_emu --platform ${XDEVICE} --report estimate ${KERNEL_SRCS}

xconfig : emconfig.json

emconfig.json :
	emconfigutil --platform ${XDEVICE} ${DEVICE_REPO_OPT} --od .

clean:
	${RM} -rf ${HOST_EXE} ${OBJECTS} ${XCLBIN} emconfig.json _xocc_${XCLBIN_NAME}_*.dir .Xil\
						 packaged_kernel_* tmp_kernel_pack_* _xocc_*.dir

cleanall: clean
	${RM} -rf *.xclbin sdaccel_profile_summary.* _xocc_* TempConfig *.log *.jou *.csv *.wcfg *.wdb *.html
	${RM} -r ${XCLBIN_DIR}


help:
	@echo "Compile and run CPU emulation using default xilinx:adm-pcie-7v3:1ddr:3.0 DSA"
	@echo "make -f sdaccel.mk run_cpu_em"
	@echo ""
	@echo "Compile and run hardware emulation using default xilinx:adm-pcie-7v3:1ddr:3.0 DSA"
	@echo "make -f sdaccel.mk run_hw_em"
	@echo ""
	@echo "Compile host executable only"
	@echo "make -f sdaccel.mk host"
	@echo ""
	@echo "Compile XCLBIN file for system run only"
	@echo "make -f sdaccel.mk xbin_hw"
	@echo ""
	@echo "Compile and run CPU emulation using xilinx:tul-pcie3-ku115:2ddr:3.0 DSA"
	@echo "make -f sdaccel.mk XDEVICE=xilinx:tul-pcie3-ku115:2ddr:3.0 run_cpu_em"
	@echo ""
	@echo "Clean working diretory"
	@echo "make -f sdaccel.mk clean"
	@echo ""
	@echo "Super clean working directory"
	@echo "make -f sdaccel.mk cleanall"
