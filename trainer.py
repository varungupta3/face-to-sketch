from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from models_v1 import *
import cv2

class Trainer(object):
  def __init__(self, config, img_loader, sketch_loader, img_loader_test, sketch_loader_test):
    self.config = config
    self.img_loader = img_loader
    self.sketch_loader = sketch_loader
    self.img_loader_test = img_loader_test
    self.sketch_loader_test = sketch_loader_test

    self.mode = config.mode

    self.batch_size = config.batch_size
    self.batch_size_eval = config.batch_size_eval

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.save_step = config.save_step
    self.wd_ratio = config.wd_ratio

    self.g_lr = tf.Variable(config.g_lr, name='g_lr')
    self.d_lr = tf.Variable(config.d_lr, name='d_lr')

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    g_decay_factor = (config.g_min_lr / config.g_lr)**(1./(self.epoch_num-1.))
    self.g_lr_update = tf.assign(self.g_lr, self.g_lr*g_decay_factor, name='g_lr_update')

    d_decay_factor = (config.d_min_lr / config.d_lr)**(1./(self.epoch_num-1.))
    self.d_lr_update = tf.assign(self.d_lr, self.d_lr*d_decay_factor, name='d_lr_update')

    self.model_dir = config.model_dir
    self.load_path = config.load_path

    if self.mode == 'photo_to_sketch_generator':
        self.generator = photo_to_sketch_generator
    elif self.mode == 'photo_to_sketch_GAN':
        self.generator = photo_to_sketch_generator
        self.discriminator = discriminator
    elif self.mode == 'sketch_to_photo_GAN':
        self.generator = sketch_to_photo_generator
        self.discriminator = discriminator
    elif self.mode == 'photo_to_sketch_GAN_UNET':
        self.generator = photo_to_sketch_generator_UNET
        self.discriminator = discriminator
    else:
        print('Wrong mode selected. Select one of 4 available choices')


    self.build_model()
    self.build_gen_eval_model()

    self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60,
                             global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def build_model(self):

    if self.mode == 'photo_to_sketch_generator':
        # Use only L1 loss or both L1 and discriminator loss
        self.x = self.img_loader
        x = self.x
        self.y = self.sketch_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y)) # L1 loss
        # self.D_loss = tf.zeros(self.G_loss.shape)
        gen_optimizer = tf.train.AdamOptimizer(self.g_lr, beta1 = 0.5, beta2=0.999)
        wd_optimizer = tf.train.GradientDescentOptimizer(self.g_lr)
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))
        self.G_optim = gen_optimizer.minimize(self.G_loss, var_list=self.G_var)
        self.wd_optim = wd_optimizer.minimize(wd_loss)


        self.summary_op = tf.summary.merge([
            tf.summary.scalar("g_lr", self.g_lr),
            tf.summary.scalar("d_lr", self.d_lr),
            tf.summary.image("gen_sketch", self.G_x),
            tf.summary.image('train_image',self.x),
            tf.summary.image('train_sketch',self.y),
            tf.summary.scalar("G_loss", self.G_loss)
            # tf.summary.scalar('D_loss', self.D_loss)
            ])

    elif self.mode == 'photo_to_sketch_GAN':
        self.x = self.img_loader
        x = self.x
        self.y = self.sketch_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)

        G_x = self.G_x
        D_G_x_in = tf.concat([G_x,x], axis=3) # Concatenates image and sketch along channel axis for generated image
        D_y_in = tf.concat([y,x], axis=3) # Concatenates image and sketch along channel axis for ground truth image
        D_in = tf.concat([D_G_x_in, D_y_in], axis=0) # Batching ground truth and generator output as input for discriminator
        D_out, self.D_var = self.discriminator(D_in, self.batch_size*2,
            is_train=True, reuse=False)
        self.D_G_x = D_out[0:self.batch_size]
        self.D_y = D_out[self.batch_size:]
        D_loss_real = tf.reduce_mean(tf.log(self.D_y))
        D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x))
        self.D_loss = D_loss_fake + D_loss_real
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y))*0.5 - self.D_loss # L1 loss

        gen_optimizer = tf.train.AdamOptimizer(self.g_lr, beta1 = 0.5, beta2=0.999)
        disc_optimizer = tf.train.AdamOptimizer(self.d_lr, beta1 = 0.5, beta2=0.999)
        wd_optimizer = tf.train.GradientDescentOptimizer(self.g_lr)
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))
        self.G_optim = gen_optimizer.minimize(self.G_loss, var_list=self.G_var)
        self.D_optim = disc_optimizer.minimize(self.D_loss, var_list=self.D_var)
        self.wd_optim = wd_optimizer.minimize(wd_loss)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("g_lr", self.g_lr),
            tf.summary.scalar("d_lr", self.d_lr),
            tf.summary.image("gen_sketch", self.G_x),
            tf.summary.image('train_image',self.x),
            tf.summary.image('train_sketch',self.y),
            tf.summary.scalar("G_loss", self.G_loss),
            tf.summary.scalar('D_loss', self.D_loss),
            tf.summary.image('D_G_x', self.D_G_x),
            tf.summary.image('D_y', self.D_y),
            ])


    elif self.mode == 'sketch_to_photo_GAN':
        self.x = self.sketch_loader
        x = self.x
        self.y = self.img_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)
        D_G_x_in = tf.concat([self.G_x,x], axis=3) # Concatenates image and sketch along channel axis for generated image
        D_y_in = tf.concat([y,x], axis=3) # Concatenates image and sketch along channel axis for ground truth image
        D_in = tf.concat([D_G_x_in, D_y_in], axis=0) # Batching ground truth and generator output as input for discriminator
        D_out, self.D_var = self.discriminator(D_in, self.batch_size*2,
            is_train=True, reuse=False)
        self.D_G_x = D_out[0:self.batch_size]
        self.D_y = D_out[self.batch_size:]
        D_loss_real = tf.reduce_mean(tf.log(self.D_y))
        D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x))
        self.D_loss = D_loss_fake + D_loss_real
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y))*0.1 - self.D_loss
        gen_optimizer = tf.train.AdamOptimizer(self.g_lr, beta1 = 0.5, beta2=0.999)
        disc_optimizer = tf.train.AdamOptimizer(self.d_lr, beta1 = 0.5, beta2=0.999)
        wd_optimizer = tf.train.GradientDescentOptimizer(self.g_lr)
        for var in tf.trainable_variables():
            print(var)
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))
        self.G_optim = gen_optimizer.minimize(self.G_loss, var_list=self.G_var)
        self.D_optim = disc_optimizer.minimize(self.D_loss, var_list=self.D_var)
        self.wd_optim = wd_optimizer.minimize(wd_loss)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("g_lr", self.g_lr),
            tf.summary.scalar("d_lr", self.d_lr),
            tf.summary.image("gen_sketch", self.G_x),
            tf.summary.image('train_image',self.x),
            tf.summary.image('train_sketch',self.y),
            tf.summary.scalar("G_loss", self.G_loss),
            tf.summary.scalar('D_loss', self.D_loss),
            tf.summary.image('D_G_x', self.D_G_x),
            tf.summary.image('D_y', self.D_y),
            ])


    elif self.mode == 'photo_to_sketch_GAN_UNET':
        self.x = self.img_loader
        x = self.x
        self.y = self.sketch_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)
        D_G_x_in = tf.concat([self.G_x,x], axis=3) # Concatenates image and sketch along channel axis for generated image
        D_y_in = tf.concat([y,x], axis=3) # Concatenates image and sketch along channel axis for ground truth image
        D_in = tf.concat([D_G_x_in, D_y_in], axis=0) # Batching ground truth and generator output as input for discriminator
        D_out, self.D_var = self.discriminator(D_in, self.batch_size*2,
            is_train=True, reuse=False)
        self.D_G_x = D_out[0:self.batch_size]
        self.D_y = D_out[self.batch_size:]
        D_loss_real = tf.reduce_mean(tf.log(self.D_y))
        D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x))
        self.D_loss = D_loss_fake + D_loss_real
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y))*0.5 - self.D_loss
        gen_optimizer = tf.train.AdamOptimizer(self.g_lr, beta1 = 0.5, beta2=0.999)
        disc_optimizer = tf.train.AdamOptimizer(self.d_lr, beta1 = 0.5, beta2=0.999)
        wd_optimizer = tf.train.GradientDescentOptimizer(self.g_lr)
        for var in tf.trainable_variables():
            print(var)
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))
        self.G_optim = gen_optimizer.minimize(self.G_loss, var_list=self.G_var)
        self.D_optim = disc_optimizer.minimize(self.D_loss, var_list=self.D_var)
        self.wd_optim = wd_optimizer.minimize(wd_loss)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("g_lr", self.g_lr),
            tf.summary.scalar("d_lr", self.d_lr),
            tf.summary.image("gen_sketch", self.G_x),
            tf.summary.image('train_image',self.x),
            tf.summary.image('train_sketch',self.y),
            tf.summary.scalar("G_loss", self.G_loss),
            tf.summary.scalar('D_loss', self.D_loss),
            tf.summary.image('D_G_x', self.D_G_x),
            tf.summary.image('D_y', self.D_y),
            ])

    else:
        print('Wrong mode selected. Choose from available 4 choices.')



  def build_gen_eval_model(self):
    if self.mode == 'photo_to_sketch_generator':
      self.test_x = self.img_loader_test
      # self.test_x = tf.placeholder(shape=[self.batch_size_eval,256,256,3], dtype=tf.float32)
      test_x = self.test_x
      self.test_y = self.sketch_loader_test
      test_y = self.test_y

      self.G_x_test, G_var = self.generator(test_x, self.batch_size_eval, 
        is_train = False, reuse = True)

      self.G_loss_test = tf.reduce_mean(tf.abs(self.G_x_test-test_y)) # L1 loss

      self.summary_op_test = tf.summary.merge([
        tf.summary.image("gen_test_sketch", self.G_x_test),
        tf.summary.image('test_image',self.test_x),
        tf.summary.image('test_sketch',self.test_y),
        tf.summary.scalar("G_loss", self.G_loss_test)
        ])
    elif self.mode == 'photo_to_sketch_GAN':
      self.test_x = self.img_loader_test
      test_x = self.test_x
      self.test_y = self.sketch_loader_test
      test_y = self.test_y

      self.G_x_test, G_var = self.generator(test_x, self.batch_size_eval, 
        is_train = False, reuse = True)

      G_x_test = self.G_x_test

      D_G_x_in = tf.concat([G_x_test,test_x], axis=3) # Concatenates image and sketch along channel axis for generated image
      D_y_in = tf.concat([test_y,test_x], axis=3) # Concatenates image and sketch along channel axis for ground truth image

      self.D_G_x_test, D_Var = self.discriminator(D_G_x_in, self.batch_size_eval, 
        is_train = False, reuse = True)

      self.D_y_test, D_Var = self.discriminator(D_y_in, self.batch_size_eval, 
        is_train = False, reuse = True)

      D_loss_real = tf.reduce_mean(tf.log(self.D_y_test))
      D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x_test))
      self.D_loss_test = D_loss_fake + D_loss_real
      self.G_loss_test = tf.reduce_mean(tf.abs(self.G_x_test-test_y))*0.5 - self.D_loss_test # L1 loss
      self.G_loss_test_L1 = tf.reduce_mean(tf.abs(self.G_x_test-test_y)) # L1 loss

      self.summary_op_test = tf.summary.merge([
        tf.summary.image("gen_test_sketch", self.G_x_test),
        tf.summary.image('test_image',self.test_x),
        tf.summary.image('test_sketch',self.test_y),
        tf.summary.scalar("G_loss", self.G_loss_test),
        tf.summary.scalar("G_loss_L1", self.G_loss_test_L1),
        tf.summary.image("D_G_x_test", self.D_G_x_test),
        tf.summary.image("D_y_test", self.D_y_test),
        tf.summary.scalar("D_loss_test", self.D_loss_test)
        ])

    elif self.mode == 'sketch_to_photo_GAN':
      self.test_x = self.sketch_loader_test
      test_x = self.test_x
      self.test_y = self.img_loader_test
      test_y = self.test_y

      self.G_x_test, G_var = self.generator(test_x, self.batch_size_eval, 
        is_train = False, reuse = True)

      G_x_test = self.G_x_test

      D_G_x_in = tf.concat([G_x_test,test_x], axis=3) # Concatenates image and sketch along channel axis for generated image
      D_y_in = tf.concat([test_y,test_x], axis=3) # Concatenates image and sketch along channel axis for ground truth image

      self.D_G_x_test, D_Var = self.discriminator(D_G_x_in, self.batch_size_eval, 
        is_train = False, reuse = True)

      self.D_y_test, D_Var = self.discriminator(D_y_in, self.batch_size_eval, 
        is_train = False, reuse = True)

      D_loss_real = tf.reduce_mean(tf.log(self.D_y_test))
      D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x_test))
      self.D_loss_test = D_loss_fake + D_loss_real
      self.G_loss_test = tf.reduce_mean(tf.abs(self.G_x_test-test_y))*0.5 - self.D_loss_test # L1 loss
      self.G_loss_test_L1 = tf.reduce_mean(tf.abs(self.G_x_test-test_y)) # L1 loss

      self.summary_op_test = tf.summary.merge([
        tf.summary.image("gen_test_image", self.G_x_test),
        tf.summary.image('test_sketch',self.test_x),
        tf.summary.image('test_image',self.test_y),
        tf.summary.scalar("G_loss", self.G_loss_test),
        tf.summary.scalar("G_loss_L1", self.G_loss_test_L1),
        tf.summary.image("D_G_x_test", self.D_G_x_test),
        tf.summary.image("D_y_test", self.D_y_test),
        tf.summary.scalar("D_loss_test", self.D_loss_test)
        ])

    elif self.mode == 'photo_to_sketch_GAN_UNET':
      self.test_x = self.img_loader_test
      test_x = self.test_x
      self.test_y = self.sketch_loader_test
      test_y = self.test_y

      self.G_x_test, G_var = self.generator(test_x, self.batch_size_eval, 
        is_train = False, reuse = True)

      G_x_test = self.G_x_test

      D_G_x_in = tf.concat([G_x_test,test_x], axis=3) # Concatenates image and sketch along channel axis for generated image
      D_y_in = tf.concat([test_y,test_x], axis=3) # Concatenates image and sketch along channel axis for ground truth image

      self.D_G_x_test, D_Var = self.discriminator(D_G_x_in, self.batch_size_eval, 
        is_train = False, reuse = True)

      self.D_y_test, D_Var = self.discriminator(D_y_in, self.batch_size_eval, 
        is_train = False, reuse = True)

      D_loss_real = tf.reduce_mean(tf.log(self.D_y_test))
      D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x_test))
      self.D_loss_test = D_loss_fake + D_loss_real
      self.G_loss_test = tf.reduce_mean(tf.abs(self.G_x_test-test_y))*0.5 - self.D_loss_test # L1 loss
      self.G_loss_test_L1 = tf.reduce_mean(tf.abs(self.G_x_test-test_y)) # L1 loss

      self.summary_op_test = tf.summary.merge([
        tf.summary.image("gen_test_sketch", self.G_x_test),
        tf.summary.image('test_image',self.test_x),
        tf.summary.image('test_sketch',self.test_y),
        tf.summary.scalar("G_loss", self.G_loss_test),
        tf.summary.scalar("G_loss_L1", self.G_loss_test_L1),
        tf.summary.image("D_G_x_test", self.D_G_x_test),
        tf.summary.image("D_y_test", self.D_y_test),
        tf.summary.scalar("D_loss_test", self.D_loss_test)
        ])



  def train(self):
    for step in trange(self.start_step, self.max_step):
      if self.config.mode == 'photo_to_sketch_generator':
        fetch_dict_gen = {
          'gen_optim': self.G_optim,
          'x': self.x,
          'y': self.y,
          'G_loss': self.G_loss,
          'G_x': self.G_x}

        # fetch_dict_disc = {
        #   'disc_optim': self.D_optim,
        #   # 'wd_optim': self.wd_optim,
        #   'D_loss': self.D_loss,
        #   # 'D_x': self.D_x,
        #   # 'G_loss': self.G_loss,
        #   # 'D_G_z':self.D_G_z,
        #   # 'G_z': self.G_z
        #   }

        if step % self.log_step == self.log_step - 1:
          fetch_dict_gen.update({
            'g_lr': self.g_lr,
            # 'd_lr': self.d_lr,
            'summary': self.summary_op })

        result = self.sess.run(fetch_dict_gen)
        G_loss = result['G_loss']
        G_x = result['G_x']

        # print("\n[{}/{}] Gen_Loss: {:.6f} " . \
        #     format(step, self.max_step, G_loss))
        # D_x = result['D_x']
        # D_G_z = result['D_G_z']
        # G_z = result['G_z']

        if step % self.log_step == self.log_step - 1:
          self.summary_writer.add_summary(result['summary'], step)
          self.summary_writer.flush()
          #pdb.set_trace()

          g_lr = result['g_lr']
          print("\n[{}/{}] Gen_Loss: {:.6f} " . \
                format(step, self.max_step, G_loss))

          sys.stdout.flush()

        if step % self.save_step == self.save_step - 1:
          self.saver.save(self.sess, self.model_dir + '/model')

          G_loss_test = 0
          for i in range(100):
            fetch_dict_gen = {
              'x': self.test_x,
              'y': self.test_y,
              'G_loss': self.G_loss_test,
              'G_x': self.G_x_test,
              'summary_test': self.summary_op_test}

            result_test = self.sess.run(fetch_dict_gen)
            G_loss_test += result_test['G_loss']

          G_loss_test /= 100

          print ('\ntest_loss = %.4f'%(G_loss_test))

          self.summary_writer.add_summary(result_test['summary_test'], step)
          self.summary_writer.flush()

        if step % self.epoch_step == self.epoch_step - 1:
          self.sess.run([self.g_lr_update])
          self.sess.run([self.d_lr_update])
      
      # elif self.config.mode == 'photo_to_sketch_GAN':
      else:
        fetch_dict_gen = {
          'gen_optim': self.G_optim,
          'x': self.x,
          'y': self.y,
          'G_loss': self.G_loss,
          'G_x': self.G_x}

        fetch_dict_disc = {
          'disc_optim': self.D_optim,
          # 'wd_optim': self.wd_optim,
          'D_loss': self.D_loss,
          'D_y': self.D_y,
          'G_loss': self.G_loss,
          'D_G_x':self.D_G_x,
          'G_x': self.G_x
          }

        if step % self.log_step == self.log_step - 1:
          fetch_dict_disc.update({
            'g_lr': self.g_lr,
            'd_lr': self.d_lr,
            'summary': self.summary_op })

        result = self.sess.run(fetch_dict_gen)
        G_loss = result['G_loss']
        x = result['x']
        y = result['y']
        G_x = result['G_x']

        result = self.sess.run(fetch_dict_disc)
        result = self.sess.run(fetch_dict_disc)
        D_y = result['D_y']
        D_G_x = result['D_G_x']
        G_x = result['G_x']
        D_loss = result['D_loss']



        if step % self.log_step == self.log_step - 1:
          self.summary_writer.add_summary(result['summary'], step)
          self.summary_writer.flush()
          #pdb.set_trace()

          g_lr = result['g_lr']
          print("\n[{}/{}] Gen_Loss: {:.6f} Disc_Loss: {:.6f} " . \
                format(step, self.max_step, G_loss, D_loss))

          sys.stdout.flush()

        if step % self.save_step == self.save_step - 1:
          self.saver.save(self.sess, self.model_dir + '/model')

          G_loss_test = 0
          for i in range(100):
            fetch_dict_gen = {
              'x': self.test_x,
              'y': self.test_y,
              'G_loss_L1': self.G_loss_test_L1,
              'G_x': self.G_x_test,
              'D_g_x': self.D_G_x_test,
              'summary_test': self.summary_op_test}

            result_test = self.sess.run(fetch_dict_gen)
            G_loss_test += result_test['G_loss_L1']

          G_loss_test /= 100

          print ('\nG_test_loss_L1 = %.4f'%(G_loss_test))

          self.summary_writer.add_summary(result_test['summary_test'], step)
          self.summary_writer.flush()

        if step % self.epoch_step == self.epoch_step - 1:
          self.sess.run([self.g_lr_update])
          self.sess.run([self.d_lr_update])



  def test(self):
    # x_image = cv2.imread(self.config.data_dir + 'images/Anand.jpeg')
    # x_image = cv2.resize(x_image, (256,256))
    # x_image = x_image[:,:,::-1]
    # x_image = (x_image - np.mean(x_image))/np.std(x_image)
    # x_image = np.repeat(x_image[np.newaxis,:,:,:], 10, axis=0)
    # pdb.set_trace()
    self.saver.restore(self.sess, self.model_dir + '/model.ckpt-0')

    if self.mode == 'photo_to_sketch_generator':
      G_loss = 0
      for i in range(1):
        fetch_dict_gen = {
          'x': self.test_x,
          'y': self.test_y,
          'G_loss': self.G_loss_test,
          'G_x': self.G_x_test}

        # feed_dict = {
        #   self.test_x : x_image
        # }

        # result = self.sess.run(fetch_dict_gen,feed_dict=feed_dict)
        result = self.sess.run(fetch_dict_gen)
        # pdb.set_trace()

        G_loss += result['G_loss']

      G_loss /= 100

    # elif self.mode == 'photo_to_sketch_GAN':
    else:
      G_loss = 0
      for i in range(1000):
        fetch_dict_gen = {
          'x': self.test_x,
          'y': self.test_y,
          'G_loss': self.G_loss_test,
          'G_x': self.G_x_test}

        # feed_dict = {
        #   self.test_x : x_image
        # }

        # result = self.sess.run(fetch_dict_gen,feed_dict=feed_dict)
        result = self.sess.run(fetch_dict_gen)
        # pdb.set_trace()

        G_loss += result['G_loss']

      G_loss /= 100

