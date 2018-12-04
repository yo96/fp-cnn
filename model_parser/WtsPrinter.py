import numpy as np
import math

"""
Example use

dumper = WtsPrinter( '../model/mnist/_dump.npy', float_nbits=8, base_per_obus=32 )
#dumper.dump_conv_wts( 'conv1/w' )
dumper.dump_fc_wts( 'fc1/w', 7, 7, 64  )
dumper.dump_fc_wts( 'last/w', 1, 1, 256 )
dumper.dump_fc( 'fc1/in', 7, 7, 64 )
dumper.dump_fc( 'fc1/out', 1, 1, 256 )

"""

class WtsPrinter(  ):
  """
  Helper class to dump out weights 
  """
  def __init__( s, npy_file, wts_file, float_nbits=8, dtype='short',
                   base_per_obus=32, base_per_ddrbus=32 ):
    # Sanity check
    assert( base_per_ddrbus % base_per_obus == 0 )

    # Load data from npy file
    npy_data = np.load( npy_file )

    s.float_nbits = float_nbits
    s.dtype       = dtype
    s._bpo         = base_per_obus
    s._bpd         = base_per_ddrbus
    s._opd         = base_per_ddrbus / base_per_obus

    # Get the wts dictionary 
    s.wts_dict = npy_data.item()
    s.wts_keys = s.wts_dict.keys()

    # Create a file to dump to 
    s._dump_file = open( wts_file, "w+" )

  def dump_conv_wts( s, key ):

    # Wts must be fixed-point numbers
    fl_wts = s.wts_dict[key]
    fp_wts = np.rint( fl_wts / 2 ** ( -1 * ( s.float_nbits ) ) ) 
    assert ( len(fp_wts.shape) == 4 )

    # Calclulate dimensions
    num_fil = fp_wts.shape[3]
    fil_wid = fp_wts.shape[0]
    fil_ht  = fp_wts.shape[1]
    dep     = fp_wts.shape[2]
    nblk    = int( math.ceil( float(dep)/s._bpo ) )
    
    actual_wid = int( math.ceil( float( fil_wid ) / s._opd ) ) * s._opd
    actual_nf  = int( math.ceil( float( num_fil ) / s._bpo ) ) * s._bpo
    
    print ( "Dumping wts %s... (%d filters, %d blocks) " % (key, num_fil, nblk) )
    print "Numpy shape:", fp_wts.shape
    
    # Print array definition
    var_name = key.replace( '/', '_' );
    s._dump_file.write( "\nconst int _%s_size = %d * %d * %d * %d;\n" % 
                         ( var_name, 
                           actual_nf, actual_wid, fil_ht, nblk * s._bpo ) )
    s._dump_file.write( "%s %s[_%s_size] = {\n" % 
                        ( s.dtype, var_name, var_name ) )
    
    wts_str = ""
    for nf in range( actual_nf ):
      #wts_str += ( "//filter%d:\n" % nf )
      for nb in range ( nblk ):
        wts_str += ( "//filter%d blk%d\n" % (nf, nb) )
        for y in range( fil_ht ):
          for x in range( actual_wid ):
            for z in range( s._bpo ):
              chnl = z + nb * s._bpo
              to_file = 0 if chnl >= dep else (
                        0 if x >= fil_wid else (
                        0 if nf >= num_fil else fp_wts[y][x][chnl][nf] ) )
              wts_str += ( "%d, " % to_file ) 
            wts_str += ( "\n" )

    # Print wts to file 
    s._dump_file.write( wts_str[:-3] + "\n};\n" )

  def dump_fc_wts( s, key, wid, ht, dep ):
    
    # Convert to fixed point integer
    fl_wts = s.wts_dict[key]
    fp_wts = np.rint( fl_wts / 2 ** ( -1 * ( s.float_nbits ) ) )
    assert( len(fp_wts.shape) == 2 )
    assert( wid * ht * dep == fp_wts.shape[0] )

    # Calculate dimensions..
    num_fil = fp_wts.shape[1]
    fil_wid = wid  
    fil_ht  = ht  
    fil_dep = dep 
    nblk    = int( math.ceil( float(fil_dep)/s._bpo ) )
    
    actual_wid = int( math.ceil( float( fil_wid ) / s._opd ) ) * s._opd
    actual_nf  = int( math.ceil( float( num_fil ) / s._bpo ) ) * s._bpo

    print ( "Dumping wts %s... (%d filters, %d blocks) " % (key, num_fil, nblk) )
    print "Numpy shape:", fp_wts.shape
    
    # Print array definition
    var_name = key.replace( '/', '_' );
    s._dump_file.write( "\nconst int _%s_size = %d * %d * %d * %d;\n" % 
                         ( var_name, 
                           actual_nf, actual_wid, fil_ht, nblk * s._bpo ) )
    s._dump_file.write( "%s %s[_%s_size] = {\n" % 
                        ( s.dtype, var_name, var_name ) )
    
    # Iterate through the fmap and dump data
    wts_str = ""
    for nf in range( actual_nf ):
      for nb in range ( nblk ):
        wts_str += ( "//filter%d blk%d\n" % ( nf, nb ) )
        for y in range( fil_ht ):
          for x in range( actual_wid ):
            for z in range( s._bpo ):
              y_offset   =  y * fil_wid * fil_dep
              x_offset   =  x * fil_dep
              chnl       =  z + nb * s._bpo
              idx        =  chnl + x_offset + y_offset
              to_file = 0 if chnl >= dep else (
                        0 if x >= fil_wid  else ( 
                        0 if nf >= num_fil else fp_wts[idx][nf] ) )
              wts_str += ( "%d, " % to_file ) 
            wts_str += ( "\n" )
    
    # Prints wts to file
    s._dump_file.write( wts_str[:-3] + "\n};\n" )
  
  def dump_fmap( s, key ):

    # Convert wts to fixed point integers
    fl_fmap = s.wts_dict[key]
    fp_fmap = np.rint( fl_fmap / 2 ** ( -1 * ( s.float_nbits ) ) )
    assert( len( fp_fmap.shape ) == 4 )

    num_fmap = fp_fmap.shape[0]
    fmap_ht  = fp_fmap.shape[1]
    fmap_wid = fp_fmap.shape[2]
    fmap_dep = fp_fmap.shape[3]
    nblk     = int( math.ceil( float( fmap_dep ) / s._bpo ) )

    # Actual sidze in order to be 64B aligned
    actual_wid = int( math.ceil( float( fmap_wid ) / s._opd ) ) * s._opd

    # Print array definition
    var_name = key.replace( '/', '_' )
    s._dump_file.write( "\nconst int _%s_size = %d * %d *%d;\n" % 
                         ( var_name, fmap_ht, fmap_wid, nblk*s._bpo ) )
    s._dump_file.write( "%s %s[%d][_%s_size] = {\n" % 
                         ( s.dtype, var_name, num_fmap, var_name) )

    print ( "Dumping fmap %s... (%d fmaps, %d blocks) " % (key, num_fmap, nblk) )
    print "Numpy shape:", fp_fmap.shape
    
    # Iterate the fmap and dump data
    str_to_file = ""
    for nf in range( num_fmap ):
      str_to_file += ( "{ // fmap%d\n" % nf )
      fmap_str = ""
      for nb in range( nblk ):
        fmap_str += ( "//fmap%d blk%d\n" % ( nf, nb ) )
        for y in range( fmap_ht ):
          for x in range( actual_wid ):
            for z in range( s._bpo ):
              chnl = z + nb * s._bpo
              to_file = 0 if chnl >= fmap_dep else (
                        0 if x >= fmap_wid else  fp_fmap[nf][y][x][chnl] )
              fmap_str += ( "%d, " % to_file )
            fmap_str += ( "\n" )
      str_to_file += ( fmap_str[:-3] + "},\n" )
    s._dump_file.write( str_to_file[:-2] + "\n};\n" )

  def dump_fc( s, key, wid, ht, dep ):
    
    # Conver data to fixed point integer
    fl_wts = s.wts_dict[key]
    fp_wts = np.rint( fl_wts / 2 ** ( -1 * ( s.float_nbits ) ) )
    assert( len(fp_wts.shape) == 2 )
    assert( wid * ht * dep == fp_wts.shape[1] )
    num_fmap = fp_wts.shape[0]
    fmap_wid  = wid  
    fmap_ht   = ht  
    fmap_dep  = dep 
    nblk     = int( math.ceil( float(fmap_dep)/s._bpo ) )
    
    actual_wid = int( math.ceil( float( fmap_wid ) / s._opd ) ) * s._opd

    print ( "Dumping wts %s... (%d filters, %d blocks) " % (key, num_fmap, nblk) )
    print "Numpy shape:", fp_wts.shape
    
    # Print array definition
    var_name = key.replace( '/', '_' );
    s._dump_file.write( "\nconst int _%s_size = %d * %d * %d;\n" % 
                         ( var_name, 
                           actual_wid, fmap_ht, nblk * s._bpo ) )
    s._dump_file.write( "%s %s[%d][_%s_size] = {\n" % 
                        ( s.dtype, var_name, num_fmap, var_name ) )

    str_to_file = ""
    for nf in range( num_fmap ):
      str_to_file += ( "{ // fc%d\n" % nf )
      fc_str = ""
      for nb in range ( nblk ):
        fc_str += ( "//fc%d blk%d\n" % ( nf, nb ) )
        for y in range( fmap_ht ):
          for x in range( actual_wid ):
            for z in range( s._bpo ):
              y_offset   =  y * fmap_wid * fmap_dep
              x_offset   =  x * fmap_dep
              chnl       =  z + nb * s._bpo
              idx        =  chnl + x_offset + y_offset
              to_file = 0 if chnl >= dep else (
                        0 if x >= fmap_wid  else fp_wts[nf][idx] )
              fc_str += ( "%d, " % to_file ) 
            fc_str += ( "\n" )
      str_to_file += ( fc_str[:-3] + "},\n" )
    
    # Prints fc data to file
    s._dump_file.write( str_to_file[:-2] + "\n};\n" )
