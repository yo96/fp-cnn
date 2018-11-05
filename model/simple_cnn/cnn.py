
print ('importing matplotlib, numpy, tensorflow...')
# import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import time
from datetime import timedelta
# from plot_helper import *
print("tensorflow version: ",tf.__version__)

#====================================================================
# CNN configurations
#====================================================================

# conv1
filter_wid1 = 3  # 3x3 filter for first conv layer
num_filters1 = 16 # 16 filters for first conv layer

# conv2
filter_wid2 = 5
num_filters2 = 32 

# fc 
fc_size = 256  # 256 neurons for fully connected layer

# always 2x2 max pooling
pool_wid = 2

# TODO: explore more data types
data_type = tf.float32

#====================================================================
# Load data
#====================================================================
print ("loading MNIST data...")
from tensorflow.examples.tutorials.mnist import input_data
# pops up a lot of warnings... 
# TODO: Please use tf.data to implement this functionality
#       Figure out how that works...
# TODO: Don't download the data again when re-run it?
# The class labels are one-hot encoded.
# 0010000000 means a 3
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
print("\n MNIST data successfully loaded! Fix the warnings int the future... \n")

print("Size of:")
print("- Training-set:\t\t{}".format( len( data.train.labels      ) ) )
print("- Test-set:\t\t{}".    format( len( data.test.labels       ) ) )
print("- Validation-set:\t{}".format( len( data.validation.labels ) ) )

#=====================================================================
# Input data dimensions
#=====================================================================

img_wid      = 28 # MNIST images are 28x28
img_size     = img_wid * img_wid
img_shape    = ( img_wid, img_wid )
num_channels = 1     # only one input channel
num_classes  = 10    # 10 classes in total

# get first 9 iamges from the test set
images = data.test.images[0:9]
# get the classes for those images
cls_true = data.test.cls[0:9]

# #plot these images and true labels
# print("\nPlotting some inputs from testing set...")
# print("\nClose the figure to continue...")
# plot_images(images=images, cls_true=cls_true)

#=====================================================================
# Construct CNN
#=====================================================================
x      = tf.placeholder( data_type, shape=[None, img_size   ], name='x' )
y_true = tf.placeholder( data_type, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax( y_true, axis=1 )

"""
  the CNN model :
  
  input 28 x 28
    |
  conv1 3x3x16
    |----------28x28x16
  pool1 2x2
    |----------14x14x16
  conv2 5x5x32
    |----------14x14x32
  pool2 2x2
    |----------7x7x32
  reshape(flatten)
    |----------1568x1
   fc1
    |----------256x1
   fc2
    |----------10x1
  argmax -> y_pred_cls

"""


def cnn_model( x ):
  """ Model Function for CNN """
  x_image = tf.reshape(x, [-1, img_wid, img_wid, 1] )

  print ("building first conv layer...")
  conv1 = tf.layers.conv2d(
    inputs      = x_image,
    filters     = num_filters1,
    kernel_size = [ filter_wid1, filter_wid1 ],
    padding     = "same",
    activation  = tf.nn.relu 
    )

  pool1 = tf.layers.max_pooling2d(
    inputs    = conv1,
    pool_size = [ pool_wid, pool_wid ],
    strides   = 2
  )

  print ("building second conv layer...")
  conv2 = tf.layers.conv2d(
    inputs      = pool1,
    filters     = num_filters2,
    kernel_size = [ filter_wid2, filter_wid2 ],
    padding     = "same",
    activation  = tf.nn.relu
  )

  pool2 = tf.layers.max_pooling2d(
    inputs    = conv2,
    pool_size = [ pool_wid, pool_wid ],
    strides   = 2 
  )

  # reshape the layer
  layer_shape  = pool2.get_shape()
  print( "output shape:\t",layer_shape )
  num_features = layer_shape[1:4].num_elements() 
  layer_flat   = tf.reshape(pool2, [ -1, num_features ] )

  # fully connected layer
  print ("building fully connected layer...")
  fc1 = tf.layers.dense(
    inputs = layer_flat,
    units  = fc_size,
    activation = tf.nn.relu
  )

  fc2 = tf.layers.dense(
    inputs = fc1,
    units  = num_classes
  )

  return fc2

#============================================================================
# Cost function 
#============================================================================
y_pred = cnn_model( x )

# cost function
cost = tf.losses.mean_squared_error(
  labels      = y_true,
  predictions = y_pred
)

optimizer = tf.train.AdamOptimizer( learning_rate=1e-3 ).minimize( cost )

# Calculate accuracy
y_pred_cls         = tf.argmax( y_pred, 1 )
correct_prediction = tf.equal ( y_pred_cls, tf.argmax( y_true, 1 )  )
accuracy = tf.reduce_mean( tf.cast( correct_prediction, data_type ) )

#===========================================================================
# Training
#===========================================================================
sess = tf.Session()
sess.run( tf.global_variables_initializer() )

# batch size. Larger batch should run faster
test_batch_size  = 100
train_batch_size = 100

# Helper function to run the test set and print out test accuracy
def print_test_accuracy():
  num_batches = len(data.test.labels)//100
  test_acc = []
  for _ in range(num_batches):
      x_batch, y_true_batch = data.test.next_batch(train_batch_size)
      feed_dict_test = {
          x: x_batch,
          y_true: y_true_batch
      }
      
      batch_acc = sess.run(accuracy, feed_dict=feed_dict_test)
      test_acc.append(batch_acc)
  print("Test accuracy: {:>6.1%}".format(np.mean(test_acc)))

# Training with batch SGD
def optimize( num_epochs ):
  num_batches = len( data.train.labels )//100
  print("\nTraining starts...")
  start_time  = time.time()

  for i in range( num_epochs ):
    train_acc = []
    for _ in range( num_batches ):
      train_acc = []
      x_batch, y_true_batch = data.train.next_batch(train_batch_size)

      feed_dict_train = {
        x      : x_batch,
        y_true : y_true_batch
      }
      batch_acc, _ = sess.run( [ accuracy, optimizer ], feed_dict=feed_dict_train )
      train_acc.append( batch_acc )
    print( "Epoch {}, Train accuracy: {:>6.1%}".format(i, np.mean(train_acc) ) )
    if i%2 == 0:
      print_test_accuracy()

  end_time = time.time()
  time_diff = end_time - start_time
  print("Time elapsed: " + str(timedelta(seconds=int(round(time_diff)))))

#============================================================================
# Run training / testing
#============================================================================

print ("\nUntrained model:" )
print_test_accuracy()

# larger [num_epochs] runs slower but should produce more accurate model 
optimize( num_epochs=4 )
print('='*30)
print("\nTraining completed! Now testing the model against validation set...")
print_test_accuracy()


# TODO: output the weights of the model HERE? 
# Perhaps tensorflow already have api for this...
tvars = tf.trainable_variables()
tvars_val = sess.run( tvars )
for var, val in zip( tvars, tvars_val ):
  print( var.name, val )

print("Session finished!")
