from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Trainer(object):
  def __init__(self, config, img_loader, sketch_loader):
    self.config = config
    self.img_loader = img_loader
    self.sketch_loader = sketch_loader
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
    elif self.mode == 'sketch_to_photo_GAN':
        self.generator = sketch_to_photo_generator
    elif self.mode == 'photo_to_sketch_GAN_UNET':
        self.generator = photo_to_sketch_generator_UNET
    else:
        print('Wrong mode selected. Select one of 4 available choices')

    self.discriminator = discriminator

    self.build_model()
    # self.build_gen_eval_model()

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
        self.D_loss = tf.zeros(self.G_loss.shape)
        gen_optimizer = tf.train.AdamOptimizer(self.g_lr, beta1 = 0.5, beta2=0.999)
        wd_optimizer = tf.train.GradientDescentOptimizer(self.g_lr)
        for var in tf.trainable_variables():
            print(var)
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))
        self.G_optim = gen_optimizer.minimize(self.G_loss, var_list=self.G_var)
        self.wd_optim = wd_optimizer.minimize(wd_loss)

    elif self.mode == 'photo_to_sketch_GAN':
        self.x = self.img_loader
        x = self.x
        self.y = self.sketch_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)

        G_x = self.G_x
        pdb.set_trace()
        D_G_x_in = tf.concat([G_x,x], axis=3) # Concatenates image and sketch along channel axis for generated image
        D_y_in = tf.concat([y,x], axis=3) # Concatenates image and sketch along channel axis for ground truth image
        D_in = tf.concat([D_G_x_in, D_y_in], axis=0) # Batching ground truth and generator output as input for discriminator
        D_out, self.D_var = self.discriminator(D_in, self.batch_size*2,
            is_train=True, reuse=False)
        self.D_G_x = D_out[0:self.batch_size]
        self.D_y = D_out[self.batch_size:]
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y)) # L1 loss
        D_loss_real = tf.reduce_mean(tf.log(self.D_y))
        D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x))
        self.D_loss = D_loss_fake + D_loss_real
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

    elif self.mode == 'sketch_to_photo_GAN':
        self.x = self.sketch_loader
        x = self.x
        self.y = self.img_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)
        D_G_x_in = tf.concat([G_x,x], axis=3) # Concatenates image and sketch along channel axis for generated image
        D_y_in = tf.concat([y,x], axis=3) # Concatenates image and sketch along channel axis for ground truth image
        D_in = tf.concat([D_G_x_in, D_y_in], axis=0) # Batching ground truth and generator output as input for discriminator
        D_out, self.D_var = self.discriminator(D_in, self.batch_size*2,
            is_train=True, reuse=False)
        self.D_G_x = D_out[0:self.batch_size]
        self.D_y = D_out[self.batch_size:]
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y)) # L1 loss
        D_loss_real = tf.reduce_mean(tf.log(self.D_y))
        D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x))
        self.D_loss = D_loss_fake + D_loss_real
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

    elif self.mode == 'photo_to_sketch_GAN_UNET':
        self.x = self.img_loader
        x = self.x
        self.y = self.sketch_loader
        y = self.y
        self.G_x, self.G_var = self.generator(x, self.batch_size, 
          is_train = True, reuse = False)
        D_G_x_in = tf.concat([G_x,x], axis=3) # Concatenates image and sketch along channel axis for generated image
        D_y_in = tf.concat([y,x], axis=3) # Concatenates image and sketch along channel axis for ground truth image
        D_in = tf.concat([D_G_x_in, D_y_in], axis=0) # Batching ground truth and generator output as input for discriminator
        D_out, self.D_var = self.discriminator(D_in, self.batch_size*2,
            is_train=True, reuse=False)
        self.D_G_x = D_out[0:self.batch_size]
        self.D_y = D_out[self.batch_size:]
        self.G_loss = tf.reduce_mean(tf.abs(self.G_x-y)) # L1 loss
        D_loss_real = tf.reduce_mean(tf.log(self.D_y))
        D_loss_fake = tf.reduce_mean(tf.log(tf.constant([1],dtype=tf.float32) - self.D_G_x))
        self.D_loss = D_loss_fake + D_loss_real
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

    else:
        print('Wrong mode selected. Choose from available 4 choices.')


    self.summary_op = tf.summary.merge([
      tf.summary.scalar("g_lr", self.g_lr),
      tf.summary.scalar("d_lr", self.d_lr),
      tf.summary.image("gen_image", self.G_x),
      tf.summary.image('train_image',self.x),
      tf.summary.scalar("G_loss", self.G_loss),
      tf.summary.scalar('D_loss', self.D_loss)
    ])

  # def build_gen_eval_model(self):
  #   self.z_test = tf.placeholder(dtype = tf.float32, shape = [self.batch_size_eval,100])
  #   z_test = self.z_test

  #   self.G_z_test, var = self.generator(z_test, self.batch_size_eval, 
  #     is_train=False, reuse=True)

  #   G_z_test = self.G_z_test

  #   self.D_G_z_test, var = self.discriminator(G_z_test, self.batch_size_eval, 
  #     is_train=False, reuse=True)


  def train(self):
    flag = False
    for step in trange(self.start_step, self.max_step):
      #print (step)
      # z = np.random.uniform(-1.0,1.0,
      #   size=[self.batch_size,100]).astype(np.float32)

      # feed_dict = {self.z : z}

      fetch_dict_gen = {
        'gen_optim': self.G_optim,
        # 'wd_optim': self.wd_optim,
        'x': self.x,
        'y': self.y,
        'G_loss': self.G_loss,
        # 'D_x': self.D_x,
        # 'D_loss': self.D_loss,
        # 'D_G_x':self.D_G_x,
        'G_x': self.G_x
        }

      # fetch_dict_disc = {
      #   'disc_optim': self.D_optim,
      #   # 'wd_optim': self.wd_optim,
      #   'D_loss': self.D_loss,
      #   # 'D_x': self.D_x,
      #   # 'G_loss': self.G_loss,
      #   # 'D_G_z':self.D_G_z,
      #   # 'G_z': self.G_z
      #   }

      # if step % self.log_step == self.log_step - 1:
      #   fetch_dict_gen.update({
      #     'g_lr': self.g_lr,
      #     'd_lr': self.d_lr,
      #     'summary': self.summary_op })

      result = self.sess.run(fetch_dict_gen)
      G_loss = result['G_loss']
      G_x = result['G_x']
      #pdb.set_trace()

      # if (step > 2000 and step <= 3000 and D_loss < 0.1) or step < 10:
        # result = self.sess.run(fetch_dict_gen)#, feed_dict =feed_dict )
        # G_loss = result['G_loss']
        # D_loss = result['D_loss']
      # else:
        # result = self.sess.run(fetch_dict_disc)#, feed_dict =feed_dict )
        # result = self.sess.run(fetch_dict_disc)#, feed_dict =feed_dict )
        # D_loss = result['D_loss']
        # result = self.sess.run(fetch_dict_gen)#, feed_dict =feed_dict )
        # G_loss = result['G_loss']

      # if flag or step < 100:
      #   result = self.sess.run(fetch_dict_gen)#, feed_dict =feed_dict )
      #   G_loss = result['G_loss']
      #   D_loss = result['D_loss']
      # else:
      #   result = self.sess.run(fetch_dict_disc)#, feed_dict =feed_dict )
      #   result = self.sess.run(fetch_dict_disc)#, feed_dict =feed_dict )
      #   D_loss = result['D_loss']
      #   result = self.sess.run(fetch_dict_gen)#, feed_dict =feed_dict )
      #   G_loss = result['G_loss']

      print("\n[{}/{}] Gen_Loss: {:.6f} " . \
          format(step, self.max_step, G_loss))
      # D_x = result['D_x']
      # D_G_z = result['D_G_z']
      # G_z = result['G_z']

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()
        #pdb.set_trace()

        # if D_loss < 0.2:
        #   flag = True
        # else:
        #   flag = False

        # plt.figure(1)
        # plt.subplot(221)
        # plt.title('img')
        # plt.imshow(G_z[0,:,:,0], cmap='gray')
        # plt.subplot(222)
        # plt.title('img2')
        # plt.imshow(G_z[3,:,:,0], cmap='gray')  
        # plt.subplot(223)
        # plt.title('img3')
        # plt.imshow(G_z[7,:,:,0], cmap='gray')
        # plt.subplot(224)
        # plt.title('img4')
        # plt.imshow(G_z[10,:,:,0], cmap='gray')  

        # plt.savefig(self.config.model_dir + '/gen_out_iter_iter_%d'%(step) +'.png' )
        # plt.close('all')

        #g_lr = result['g_lr']
        print("\n[{}/{}] Gen_Loss: {:.6f} " . \
              format(step, self.max_step, G_loss))


        # print("\n[{}/{}:{:.6f}] Gen_Loss: {:.6f} Disc_Loss: {:.6f}" . \
              # format(step, self.max_step, lr, G_loss, D_loss))

        sys.stdout.flush()

      # if step % self.save_step == self.save_step - 1:
      #   self.saver.save(self.sess, self.model_dir + '/model')


      #   images = np.zeros((64*10,64*10))
      #   gen_images = np.zeros((64*10,64*10))

      #   for i in range(10):
      #     for j in range(10):
      #       z_test = np.random.uniform(-1.0,1.0,
      #       size=[self.batch_size_eval,100]).astype(np.float32)
        
      #       feed_dict_gen = {self.z_test : z_test}

      #       fetch_dict_gen = {'G_z': self.G_z_test,
      #                         'D_G_z': self.D_G_z_test,
      #                         'image': self.x}
      #       result = self.sess.run(fetch_dict_gen,feed_dict=feed_dict_gen)

      #       idx = i*10 + j
      #       im = result['image'][idx,:,:,0]
      #       images[i*64:(i+1)*64,j*64:(j+1)*64] = im 
      #       gen_im = result['G_z'][idx,:,:,0]
      #       gen_images[i*64:(i+1)*64,j*64:(j+1)*64] = gen_im

      #   plt.figure(1)
      #   plt.title('org_imgs_iter_%d'%(step))
      #   plt.imshow(images, cmap='gray')
      #   plt.savefig(self.config.model_dir + '/org_imgs_iter_%d'%(step) +'.png' )
      #   plt.close('all')

      #   plt.figure(1)
      #   plt.title('gen_imgs_iter_%d'%(step))
      #   plt.imshow(gen_images, cmap='gray')
      #   plt.savefig(self.config.model_dir + '/gen_imgs_iter_%d'%(step) +'.png' )
      #   plt.close('all')

      #   # D_G_z = np.mean(result['D_G_z'])

      #   # sys.stdout.flush()
      #   # print("Disc_score: {:.6f}" . \
      #   #   format(D_G_z))
      #   # sys.stdout.flush()


      if step % self.epoch_step == self.epoch_step - 1:
        self.sess.run([self.g_lr_update])
        self.sess.run([self.d_lr_update])


  def gen_image(self):
    self.saver.restore(self.sess, self.model_dir + '/model.ckpt-0')
    z1 = np.random.normal(0,1,100).reshape(self.batch_size_eval,100)
    z2 = np.random.normal(0,1,100).reshape(self.batch_size_eval,100)

    for i,alpha in enumerate(np.linspace(0, 1,20)):
      noise_in =  np.random.normal(0,alpha,(self.batch_size_eval,100)).reshape(1,100)
      feed_dict_gen = {self.z_test : noise_in}
      fetch_dict_gen = {'G_z': self.G_z_test}
      result = self.sess.run(fetch_dict_gen,feed_dict=feed_dict_gen)
      result = result['G_z']

      # pdb.set_trace()
      plt.figure(1)
      plt.title('img%d'%(i))
      plt.imshow(result[0,:,:,0], cmap='gray')
      plt.savefig(self.config.model_dir + '/gen_out_im_%d'%(i) +'.png' )
      plt.close('all')
