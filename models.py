import tensorflow as tf
from layers import *
import pdb

def generator(x, batch_size, is_train, reuse):

  with tf.variable_scope('GEN', reuse=reuse) as vs:
    with tf.variable_scope('fc1', reuse=reuse):
      hidden_num = 1024
      x = fc_factory(x, 4*hidden_num, is_train, reuse)
      x = tf.reshape(x, shape = [batch_size,2,2,1024])
      print (x.shape)

    with tf.variable_scope('deconv1', reuse=reuse):
      hidden_num /= 4
      x = t_conv_factory(x, hidden_num,[batch_size,4,4,256] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv2', reuse=reuse):
      hidden_num /= 4
      x = t_conv_factory(x, hidden_num,[batch_size,8,8,64] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv3', reuse=reuse):
      hidden_num /= 4
      x = t_conv_factory(x, hidden_num,[batch_size,16,16,16] ,3, 1, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('deconv4', reuse=reuse):
      hidden_num /= 4
      x = t_conv_factory(x, hidden_num,[batch_size,32,32,4] ,3, 1, is_train, reuse)
      print (x.shape)

    # with tf.variable_scope('deconv5', reuse=reuse):
    #   hidden_num /= 2
    #   x = t_conv_factory(x, hidden_num,[batch_size,32,32,4] ,3, 1, is_train, reuse)
    #   print (x.shape)

    with tf.variable_scope('deconv6', reuse=reuse):
      hidden_num = 1
      x = t_conv_factory_tanh(x, hidden_num,[batch_size,64,64,1] ,3, 1, is_train, reuse)
      print (x.shape)

  variables = tf.contrib.framework.get_variables(vs)
  return x,variables


def discriminator(x, batch_size, is_train, reuse):
  
  with tf.variable_scope('DIS', reuse=reuse) as vs:
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 4
      x = conv_factory_leaky(x, hidden_num, 3, 2, is_train, reuse)
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

    with tf.variable_scope('fc1', reuse=reuse):
      x = tf.reshape(x, shape = [batch_size, -1])
      x = fc_factory_leaky(x, 100, is_train, reuse)
      print (x.shape)

    with tf.variable_scope('fc_out', reuse=reuse):
      # x = tf.reshape(x, shape = [batch_size, -1])
      x = fc_factory_noact(x, 1, is_train, reuse)
      print (x.shape)
      # x = tf.nn.sigmoid(x)

  variables = tf.contrib.framework.get_variables(vs)
  return x, variables