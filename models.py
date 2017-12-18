import tensorflow as tf
from layers import *
import pdb

def photo_to_sketch_generator(x, batch_size, is_train, reuse):

  with tf.variable_scope('GEN', reuse=reuse) as vs:

    with tf.variable_scope('Encoder', reuse=reuse) as vs_enc:
      with tf.variable_scope('conv1', reuse=reuse):
        hidden_num = 64
        x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('conv2', reuse=reuse):
        hidden_num *= 2
        x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('conv3', reuse=reuse):
        hidden_num *= 2
        x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('conv4', reuse=reuse):
        hidden_num *= 2
        x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('conv5', reuse=reuse):
        hidden_num *= 2
        x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('conv6', reuse=reuse):
        x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
        print (x.shape)


      with tf.variable_scope('dropout', reuse=reuse):
        x = tf.nn.dropout(x, keep_prob=0.5)
        print ('Dropout layer : ', x.shape)

    with tf.variable_scope('Decoder', reuse=reuse) as vs_dec:
      with tf.variable_scope('deconv1', reuse=reuse):
        out_channels = hidden_num
        x = t_conv_factory(x, hidden_num, [batch_size,8,7,out_channels], 3, 1, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('deconv2', reuse=reuse):
        hidden_num /= 2
        out_channels = hidden_num
        x = t_conv_factory(x, hidden_num, [batch_size,16,13,out_channels], 3, 1, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('deconv3', reuse=reuse):
        hidden_num /= 2
        out_channels = hidden_num
        x = t_conv_factory(x, hidden_num, [batch_size,32,25,out_channels], 3, 1, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('deconv4', reuse=reuse):
        hidden_num /= 2
        out_channels = hidden_num
        x = t_conv_factory(x, hidden_num, [batch_size,63,50,out_channels], 3, 1, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('deconv5', reuse=reuse):
        hidden_num /= 2
        out_channels = hidden_num
        x = t_conv_factory(x, hidden_num, [batch_size,125,100,out_channels], 3, 1, is_train, reuse)
        print (x.shape)

      with tf.variable_scope('deconv6', reuse=reuse):
        hidden_num = 1
        out_channels = hidden_num
        x = t_conv_factory_noact(x, hidden_num, [batch_size,250,200,out_channels], 3, 1, is_train, reuse)
        print (x.shape)


    variables = tf.contrib.framework.get_variables(vs)
    return x,variables


def discriminator(x, batch_size, is_train, reuse):
  
  with tf.variable_scope('DIS', reuse=reuse) as vs:
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32
      x = conv_factory_leaky_nbn(x, hidden_num, 3, 2, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num *= 2
      x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num *= 2
      x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num *= 2
      x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv5', reuse=reuse):
      hidden_num *= 2
      x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv6', reuse=reuse):
      hidden_num *= 2
      x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

    with tf.variable_scope('conv7', reuse=reuse):
      x = conv_factory_sig(x, 1, 1, 1, is_train, reuse)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
      print (x.shape)

      # x = tf.nn.sigmoid(x)

  variables = tf.contrib.framework.get_variables(vs)
  return x, variables
