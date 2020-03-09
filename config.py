#-*- coding: utf-8 -*-                                                                                
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='cufs_photos', choices=['cufs_photos'])
data_arg.add_argument('--batch_size', type=int, default=10)
data_arg.add_argument('--batch_size_eval', type=int, default=10)
data_arg.add_argument('--mode', type=str, default='photo_to_sketch_GAN', choices=['photo_to_sketch_generator',
	'photo_to_sketch_GAN', 'sketch_to_photo_GAN', 'photo_to_sketch_GAN_UNET'])

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--max_step', type=int, default=1500)
train_arg.add_argument('--epoch_step', type=int, default=100)
train_arg.add_argument('--g_lr', type=float, default=1e-3)
train_arg.add_argument('--d_lr', type=float, default=1e-4)
train_arg.add_argument('--g_min_lr', type=float, default=1e-4)
train_arg.add_argument('--d_min_lr', type=float, default=1e-5)
train_arg.add_argument('--wd_ratio', type=float, default=5e-2)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='logs/pts_GAN_best')
misc_arg.add_argument('--log_step', type=int, default=1)
misc_arg.add_argument('--save_step', type=int, default=1500)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='/home/varun/Courses/CIS680/vision_and_learning/HW4/datasets/')
misc_arg.add_argument('--random_seed', type=int, default=0)


def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed    
