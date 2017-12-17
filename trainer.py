from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from models import *


class Trainer(object):
  def __init__(self, config, data_loader):
    self.config = config
    self.data_loader = data_loader

    self.batch_size = config.batch_size
    self.batch_size_eval = config.batch_size_eval

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.save_step = config.save_step
    self.wd_ratio = config.wd_ratio

    self.lr = tf.Variable(config.lr, name='lr')

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*decay_factor, name='lr_update')

    self.model_dir = config.model_dir
    self.load_path = config.load_path

    self.generator = generator_v2
    self.discriminator = discriminator

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
    self.x = self.data_loader
    x = self.x
    # self.z = tf.placeholder(dtype = tf.float32, shape = [self.batch_size,100])
    self.z = tf.random_uniform(shape = [self.batch_size,100],minval=-1.0,maxval=1.0)

    z = self.z

    self.G_z, self.G_var = self.generator(z, self.batch_size, 
      is_train = True, reuse = False)

    G_z = self.G_z

    D_in = tf.concat([G_z,x], axis=0)

    D_out, self.D_var = self.discriminator(D_in, self.batch_size*2, 
      is_train=True, reuse=False) 

    self.D_G_z = D_out[0:self.batch_size]
    self.D_x = D_out[self.batch_size:]    

    # self.D_G_z, D_var = discriminator(G_z, self.batch_size, 
    #   is_train=False, reuse=True)



    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits = self.D_x,labels=tf.ones_like(self.D_x)))

    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits = self.D_G_z,labels=tf.zeros_like(self.D_G_z)))

    self.D_loss = D_loss_fake + D_loss_real

    self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits = self.D_G_z,labels=tf.ones_like(self.D_G_z)))
    

    wd_optimizer = tf.train.GradientDescentOptimizer(self.lr)
    gen_optimizer = tf.train.AdamOptimizer(self.lr*10, beta1 = 0.5)
    disc_optimizer = tf.train.AdamOptimizer(self.lr, beta1 = 0.5)


    for var in tf.trainable_variables():
      weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
      tf.add_to_collection('losses', weight_decay)
    wd_loss = tf.add_n(tf.get_collection('losses'))

    self.G_optim = gen_optimizer.minimize(self.G_loss, var_list=self.G_var)   
    self.D_optim = disc_optimizer.minimize(self.D_loss, var_list=self.D_var)
    self.wd_optim = wd_optimizer.minimize(wd_loss)

    # pdb.set_trace()

    D_G_z_mean = tf.reduce_mean(tf.nn.sigmoid(self.D_G_z))
    D_x_mean = tf.reduce_mean(tf.nn.sigmoid(self.D_x))

    self.summary_op = tf.summary.merge([
      tf.summary.scalar("lr", self.lr),
      tf.summary.image("gen_image", self.G_z),
      tf.summary.histogram('input_noise',self.z),
      tf.summary.image("train_image", self.x),
      tf.summary.scalar("G_loss", self.G_loss),
      tf.summary.scalar('D_loss', self.D_loss),
      tf.summary.scalar('D_G_z', D_G_z_mean),
      tf.summary.scalar('D_x', D_x_mean)

    ])

  def build_gen_eval_model(self):
    self.z_test = tf.placeholder(dtype = tf.float32, shape = [self.batch_size_eval,100])
    z_test = self.z_test

    self.G_z_test, var = self.generator(z_test, self.batch_size_eval, 
      is_train=False, reuse=True)

    G_z_test = self.G_z_test

    self.D_G_z_test, var = self.discriminator(G_z_test, self.batch_size_eval, 
      is_train=False, reuse=True)


  def train(self):
    flag = False
    for step in trange(self.start_step, self.max_step):
      # z = np.random.uniform(-1.0,1.0,
      #   size=[self.batch_size,100]).astype(np.float32)

      # feed_dict = {self.z : z}

      fetch_dict_gen = {
        'gen_optim': self.G_optim,
        # 'wd_optim': self.wd_optim,
        'G_loss': self.G_loss,
        # 'D_x': self.D_x,
        'D_loss': self.D_loss,
        # 'D_G_z':self.D_G_z,
        # 'G_z': self.G_z
        }

      fetch_dict_disc = {
        'disc_optim': self.D_optim,
        # 'wd_optim': self.wd_optim,
        'D_loss': self.D_loss,
        # 'D_x': self.D_x,
        # 'G_loss': self.G_loss,
        # 'D_G_z':self.D_G_z,
        # 'G_z': self.G_z
        }

      if step % self.log_step == self.log_step - 1:
        fetch_dict_gen.update({
          'lr': self.lr,
          'summary': self.summary_op })

      if (step > 2000 and step <= 3000 and D_loss < 0.1) or step < 10:
        result = self.sess.run(fetch_dict_gen)#, feed_dict =feed_dict )
        G_loss = result['G_loss']
        D_loss = result['D_loss']
      else:
        result = self.sess.run(fetch_dict_disc)#, feed_dict =feed_dict )
        result = self.sess.run(fetch_dict_disc)#, feed_dict =feed_dict )
        D_loss = result['D_loss']
        result = self.sess.run(fetch_dict_gen)#, feed_dict =feed_dict )
        G_loss = result['G_loss']

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


      # D_x = result['D_x']
      # D_G_z = result['D_G_z']
      # G_z = result['G_z']

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()

        if D_loss < 0.2:
          flag = True
        else:
          flag = False

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

        lr = result['lr']


        print("\n[{}/{}:{:.6f}] Gen_Loss: {:.6f} Disc_Loss: {:.6f}" . \
              format(step, self.max_step, lr, G_loss, D_loss))
        sys.stdout.flush()

      if step % self.save_step == self.save_step - 1:
        self.saver.save(self.sess, self.model_dir + '/model')


        images = np.zeros((64*10,64*10))
        gen_images = np.zeros((64*10,64*10))

        for i in range(10):
          for j in range(10):
            z_test = np.random.uniform(-1.0,1.0,
            size=[self.batch_size_eval,100]).astype(np.float32)
        
            feed_dict_gen = {self.z_test : z_test}

            fetch_dict_gen = {'G_z': self.G_z_test,
                              'D_G_z': self.D_G_z_test,
                              'image': self.x}
            result = self.sess.run(fetch_dict_gen,feed_dict=feed_dict_gen)

            idx = i*10 + j
            im = result['image'][idx,:,:,0]
            images[i*64:(i+1)*64,j*64:(j+1)*64] = im 
            gen_im = result['G_z'][idx,:,:,0]
            gen_images[i*64:(i+1)*64,j*64:(j+1)*64] = gen_im

        plt.figure(1)
        plt.title('org_imgs_iter_%d'%(step))
        plt.imshow(images, cmap='gray')
        plt.savefig(self.config.model_dir + '/org_imgs_iter_%d'%(step) +'.png' )
        plt.close('all')

        plt.figure(1)
        plt.title('gen_imgs_iter_%d'%(step))
        plt.imshow(gen_images, cmap='gray')
        plt.savefig(self.config.model_dir + '/gen_imgs_iter_%d'%(step) +'.png' )
        plt.close('all')

        # D_G_z = np.mean(result['D_G_z'])

        # sys.stdout.flush()
        # print("Disc_score: {:.6f}" . \
        #   format(D_G_z))
        # sys.stdout.flush()


      if step % self.epoch_step == self.epoch_step - 1:
        self.sess.run([self.lr_update])


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
