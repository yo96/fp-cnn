# !bin/sh
MODEL_NAME="mnist"
MODEL_DIR=../model/${MODEL_NAME}
DEST_DIR=../src/${MODEL_NAME}
TEMPLATE_DIR=../src/conv_pool_test

mkdir -p $DEST_DIR/host
mkdir -p $DEST_DIR/krnl

# Begin copying files...
cp $TEMPLATE_DIR/sdaccel.ini $DEST_DIR/sdaccel.ini
cp $TEMPLATE_DIR/sdaccel.mk $DEST_DIR/sdaccel.mk
cp $TEMPLATE_DIR/host/allocator.h $DEST_DIR/host/allocator.h
cp $TEMPLATE_DIR/host/test.h $DEST_DIR/host/test.h
cp $TEMPLATE_DIR/host/host.cpp $DEST_DIR/host/host.cpp
cp $TEMPLATE_DIR/krnl/acc_relu.cl $DEST_DIR/krnl/acc_relu.cl
cp $TEMPLATE_DIR/krnl/compute.cl $DEST_DIR/krnl/compute.cl
cp $TEMPLATE_DIR/krnl/load_fmap.cl $DEST_DIR/krnl/load_fmap.cl
cp $TEMPLATE_DIR/krnl/load_wts.cl $DEST_DIR/krnl/load_wts.cl
cp $TEMPLATE_DIR/krnl/pooling_wb.cl $DEST_DIR/krnl/pooling_wb.cl
