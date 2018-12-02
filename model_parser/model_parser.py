# model_parser.py
# Takes the nn_configs.txt and _dump.npy file as input
# Generate a new SDAccel project that contains the CNN accelerator

# Model parser will parse the nn_configs.txt file into an IR that represents how the OpenCL kernels are configured (the arguments)
# It also generates the weight file that are ready to be loaded into the host DRAM

# TODO List:
# 0. Wrappers for Tensorflow APIs [DONE]
# 1. Generate correct _dump.npy and nn_configs.txt [DONE]
# 2. Parse nn_configs.txt and calculate the parameters (use simple parameters for now, write to defs.h and configs.h)
# 3. Generate the IR (determine the arguments for each kernel)
# 4. Generate the driver code (cnn_xcel.h which contains initialize() and runInference())
# 5. Pad _dump.npy and dump it into another weight file so that host.cpp can easily load it
import re
from math import ceil

MODEL_NAME = 'mnist'
NN_CONFIGS = '../model/' + MODEL_NAME + '/nn_configs.txt'
KRNL_DIR = '../src/' + MODEL_NAME + '/krnl/'
HOST_DIR = '../src/' + MODEL_NAME + '/host/'

class IR( object ):
  def __init__( s, conv, relu, pooling ):
    s.conv = conv
    s.relu = relu
    s.pooling = pooling
  def gen_outshape( s ):
    o_w = int(ceil( float(s.conv.fmap_w)/float(s.conv.stride) ))
    o_h = int(ceil( float(s.conv.fmap_h)/float(s.conv.stride) ))
    s.out_shape = [
      int(ceil( float(o_w-s.pooling.pool_w+1)/float(s.pooling.pool_stride))), 
      int(ceil( float(o_h-s.pooling.pool_h+1)/float(s.pooling.pool_stride))), 
      s.conv.filter_num
    ]
  def gen_tile( s ):
    s.conv.tile_w = s.conv.fmap_w + s.conv.lpadding + s.conv.rpadding
    s.conv.tile_h = \
      ( s.pooling.pool_h - 1 ) * s.conv.stride + s.conv.filter_h
  def _print( s ):
    print( 'IR:\n' )
    print( '\tconv:\n' )
    print( s.conv.nn_type + ', fmap_w = ' + str(s.conv.fmap_w) + ', fmap_h = ' + str(s.conv.fmap_h) + ', fmap_chnl = ' + str(s.conv.fmap_chnl) + '\n' )
    print( 'filter_w = ' + str(s.conv.filter_w) + ', filter_h = ' + str(s.conv.filter_h) + ', filter_num = ' + str(s.conv.filter_num) + '\n')
    print( 'stride = ' + str(s.conv.stride) + ', tile_w = ' + str(s.conv.tile_w) + ', tile_h = ' + str(s.conv.tile_h) + '\n')
    print( '\trelu: ' + str(s.relu) + '\n' )
    print( '\tpooling: w = ' + str(s.pooling.pool_w) + ', h = ' + str(s.pooling.pool_h) + ', stride = ' + str(s.pooling.pool_stride) + '\n' )
    print( '\tout_shape: ' + str( s.out_shape ) + '\n' )

class conv( object ):
  def __init__( s, nn_type, fmap_w, fmap_h, fmap_chnl, filter_w, filter_h,
      filter_num, stride ):
    s.nn_type = nn_type
    s.fmap_w = fmap_w
    s.fmap_h = fmap_h
    s.fmap_chnl = fmap_chnl
    s.filter_w = filter_w
    s.filter_h = filter_h
    s.filter_num = filter_num
    s.stride = stride

  def gen_padding( s, nn_type ):
    if nn_type is 'conv2d':
      # same padding for conv
      if s.filter_w % 2 == 1:
        s.lpadding = s.rpadding = (s.filter_w-1)/2
      else:
        s.rpadding = s.filter_w/2
        s.lpadding = s.rpadding-1

      if s.filter_h % 2 == 1:
        s.upadding = s.dpadding = (s.filter_h-1)/2
      else:
        s.dpadding = s.filter_h/2
        s.upadding = s.dpadding-1

    elif nn_type is 'fc':
      # valid padding for fc
      s.lpadding = 0
      s.rpadding = 0
      s.upadding = 0
      s.dpadding = 0


class pooling( object ):
  def __init__( s, pool_w, pool_h, pool_stride ):
    s.pool_w = pool_w
    s.pool_h = pool_h
    s.pool_stride = pool_stride

def get_layer_type( s ):
  return s.split()[0]

def get_shape( s ):
  # Drop the -1 in the first position
  return [ int(x) for x in s.split(',') ][ 1: ]

def get_pooling_info( s ):
  ss = s.split( ' ' )
  return ( int(ss[2]), int(ss[3]), int(ss[4]) )

def get_conv_info( s ):
  ss = s.split( ' ' )
  return map( int, ss[2:] )

def get_fc_info( s ):
  ss = s.split( ' ' )
  return int(ss[2])

def gen_macro( out, idx, body, value ):
  s = '#define {} ({})\n'.format(
      body+'_'+str(idx), str(value)
    )
  out.write( s )

def _i( idx, s ):
  return s+str(idx)

def parse_nn_configs():
  with open( NN_CONFIGS, 'r' ) as config_file:
    configs = config_file.read().splitlines()

    IRs = []
    idx = 0

    while idx < len( configs ):
      # Try to extract the (optional) reshape layer
      if idx is 0: 
        assert get_layer_type( configs[idx] ) == 'reshape'

      if get_layer_type( configs[idx] ) == 'reshape':
        in_shape = get_shape( re.search( '\[([^]]+)', configs[idx] ).group( 1 ) )
        idx += 1
        # Do not allow reshaping the output of the last layer
        assert idx < len( configs )
      else:
        in_shape = None

      # Try to extract the convolution/fc layer
      assert ( get_layer_type( configs[idx] ) == 'conv2d' ) or\
             ( get_layer_type( configs[idx] ) == 'fc' )

      if get_layer_type( configs[idx] ) == 'conv2d':
        conv_info = get_conv_info( configs[idx] )
        if in_shape:
          # apply reshaped size to this layer
          assert len(in_shape) is 3
          _conv = conv( 'conv2d', in_shape[0], in_shape[1], in_shape[2],
            conv_info[1], conv_info[2], conv_info[3], conv_info[0]
            )
        else:
          # infer input size from previous layer
          assert len(IRs) != 0
          out_shape = IRs[-1].out_shape
          _conv = conv( 'conv2d', out_shape[0], out_shape[1], out_shape[2],
            conv_info[1], conv_info[2], conv_info[3], conv_info[0]
            )
        _conv.gen_padding( 'conv2d' )

      if get_layer_type( configs[idx] ) == 'fc':
        fc_info = get_fc_info( configs[idx] )
        if in_shape:
          # apply reshaped size to this layer
          assert len(in_shape) == 1
          _conv = conv( 'fc', 1, 1, in_shape[0],
            1, 1, fc_info, 1
            )
        else:
          # infer input size from previous layer
          assert len(IRs) != 0
          out_shape = IRs[-1].out_shape
          _conv = conv( 'fc', out_shape[0], out_shape[1], out_shape[2],
            out_shape[0], out_shape[1], fc_info, 
            max( out_shape[0], out_shape[1] )
            )
        _conv.gen_padding( 'fc' )

      idx += 1

      # Try to extract the (optional) relu layer
      if idx < len(configs) and get_layer_type( configs[idx] ) == 'relu':
        _relu = True
        idx += 1
      else:
        _relu = False

      # Try to extract the (optional) pooling layer
      if idx < len(configs) and \
          get_layer_type( configs[idx] ) == 'max_pooling2d':
        ( pw, ph, ps ) = get_pooling_info( configs[idx] )
        _pooling = pooling( pw, ph, ps )
        idx += 1
      else:
        _pooling = pooling( 1, 1, 1 )

      # Calculate the output shape of the IR
      ir = IR( _conv, _relu, _pooling )
      ir.gen_outshape()
      ir.gen_tile()
      IRs += [ ir ]
  return IRs

def generate_params( IRs ):
  ret = {}
  BASE = 16
  DDR = 512

  wts_buf_size = 0
  tile_buf_size = 0

  for ir in IRs:
    _wts_buf_size =\
      ir.conv.filter_w*ir.conv.filter_h*ir.conv.fmap_chnl*BASE/DDR
    _tile_buf_size =\
      ir.conv.tile_w*ir.conv.tile_h*ir.conv.fmap_chnl*BASE/DDR
    wts_buf_size = max( wts_buf_size, _wts_buf_size )
    tile_buf_size = max( tile_buf_size, _tile_buf_size )

  ret[ 'BASE' ] = BASE
  ret[ 'DDR' ] = DDR
  ret[ 'BASE_PER_DDR' ] = DDR/BASE
  ret[ 'sys_w' ] = 16
  ret[ 'sys_h' ] = 32
  ret[ 'wts_buf_size' ] = wts_buf_size
  ret[ 'tile_buf_size' ] = tile_buf_size

  return ret

def generate_project( IRs, params ):
  with open( KRNL_DIR+'defs.h', 'w' ) as out:
    s = open( 'defs.h.tmplt', 'r' ).read()

    if params['BASE'] is 32:
      base_t = 'int'
    elif params['BASE'] is 16:
      base_t = 'short'
    else:
      raise 'Unsupported BASE type'

    out.write( s.format(
        BASE_T = base_t,
        FBUS_T = 'int' + str( params['sys_w']*params['BASE']/32 ),
        OBUS_T = 'int' + str( params['sys_h']*params['BASE']/32 )
      )
    )


  with open( KRNL_DIR+'configs.h', 'w' ) as out:
    s = open( 'configs.h.tmplt', 'r' ).read()

    out.write( s.format(
        BASE_PER_DBUS = params['DDR']/params['BASE'],
        BASE_PER_FBUS = params['sys_w'],
        BASE_PER_OBUS = params['sys_h'],
        TILE_BUF_SIZE = params['tile_buf_size'],
        FIL_BUF_SIZE = params['wts_buf_size']
      )
    )

  arg_dict = {
      'FMAP_WID':'conv.fmap_w', 'FMAP_HT':'conv.fmap_h',
      'FMAP_DEP':'conv.fmap_chnl', 'FIL_WID':'conv.filter_w',
      'FIL_HT':'conv.filter_h', 'NUM_FIL':'conv.filter_num',
      'POOL_WID':'pooling.pool_w', 'POOL_HT':'pooling.pool_h',
      'POOL_STRIDE':'pooling.pool_stride', 'CONV_STRIDE':'conv.stride',
      'TILE_WID':'conv.tile_w', 'TILE_HT':'conv.tile_h',
      'LPADDING':'conv.lpadding', 'RPADDING':'conv.rpadding',
      'UPADDING':'conv.upadding', 'DPADDING':'conv.dpadding'
    }

  with open( HOST_DIR+'args.h', 'w' ) as out:
    for i, ir in enumerate(IRs):
      for arg, val in arg_dict.iteritems():
        # exec( '_val = ir.' + val )
        # gen_macro( out, i, arg, _val )
        gen_macro( out, i, 'FMAP_WID', ir.conv.fmap_w )
        gen_macro( out, i, 'FMAP_HT', ir.conv.fmap_h )
        gen_macro( out, i, 'FMAP_DEP', ir.conv.fmap_chl )
        gen_macro( out, i, 'FIL_WID', ir.conv.filter_w )
        gen_macro( out, i, 'FIL_HT', ir.conv.filter_h )
        gen_macro( out, i, 'NUM_FIL', ir.conv.filter_num )
        gen_macro( out, i, 'POOL_WID', ir.pooling.pool_w )
        gen_macro( out, i, 'POOL_HT', ir.pooling.pool_h )
        gen_macro( out, i, 'POOL_STRIDE', ir.pooling.pool_stride )
        gen_macro( out, i, 'CONV_STRIDE', ir.conv.stride )
        gen_macro( out, i, 'TILE_WID', ir.conv.tile_w )
        gen_macro( out, i, 'TILE_HT', ir.conv.tile_h )
        gen_macro( out, i, 'LPADDING', ir.conv.lpadding )
        gen_macro( out, i, 'RPADDING', ir.conv.rpadding )
        gen_macro( out, i, 'UPADDING', ir.conv.upadding )
        gen_macro( out, i, 'DPADDING', ir.conv.dpadding )

  with open( HOST_DIR+'cnn_xcel.h', 'w' ) as out:
    s = open( 'cnn_xcel.h.tmplt', 'r' ).read()
    exec_s = ''
    code_s =\
"""set_kernel_arg( *arg,
    FMAP_WID_{n}, FMAP_HT_{n}, FMAP_DEP_{n}, 
    FIL_WID_{n}, FIL_HT_{n}, NUM_FIL_{n}, 
    POOL_WID_{n}, POOL_HT_{n}, POOL_STRIDE_{n}, 
    CONV_STRIDE_{n}, TILE_WID_{n}, TILE_HT_{n},
    LPADDING_{n}, RPADDING_{n}, UPADDING_{n}, DPADDING_{n}
  );
  exec_layer( *arg  );
  """
    for i, ir in enumerate(IRs):
      exec_s += code_s.format(
        n = str(i)
      )
    out.write(
        s.format( EXEC = exec_s )
      )

if __name__ == '__main__':
  IRs = parse_nn_configs()
  params = generate_params( IRs )
  generate_project( IRs, params )
  for ir in IRs:
    ir._print()
