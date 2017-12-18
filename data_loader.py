import os
import numpy as np
import tensorflow as tf
import pdb
import glob

def read_labeled_image_list(img_dir):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_dir: path of directory that contains images
  Returns:
    List with all filenames
  """
  files = glob.glob(img_dir+'/*.jpg')
  img_paths = []
  for file in files:
    img_paths.append(file)
  return img_paths

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string
  Args:
    filename_and_label_tensor: A scalar string tensor
  Returns:
    Two tensors: the decoded image, and the string label
  """
  img_path = tf.read_file(input_queue[0])
  sketch_path = tf.read_file(input_queue[1])
  img = tf.image.decode_png(img_path, channels=3)
  sketch = tf.image.decode_png(sketch_path, channels=1)
  return img, sketch

def get_loader(root, batch_size, split='train', shuffle=True):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
  """

  img_paths_np = read_labeled_image_list(root+'/'+split+'/'+'photos')
  sketch_paths_np = read_labeled_image_list(root+'/'+split+'/'+'sketches')
  	
  with tf.device('/cpu:0'):
    img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)
    sketch_paths = tf.convert_to_tensor(sketch_paths_np, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([img_paths,sketch_paths], shuffle=shuffle, 
      capacity=10*batch_size)

    img, sketch = read_images_from_disk(input_queue)

    img.set_shape([250, 200, 3])
    img = tf.cast(img, tf.float32)
    img = tf.image.resize_images(img, (256,256))
    img = tf.image.per_image_standardization(img)

    sketch.set_shape([250, 200, 1])
    sketch = tf.cast(sketch, tf.float32)
    sketch = tf.image.resize_images(sketch, (256,256))
    sketch = tf.image.per_image_standardization(sketch)

    img_batch, sketch_batch = tf.train.batch([img, sketch], num_threads=1, batch_size=batch_size, capacity=10*batch_size)

  return img_batch, sketch_batch
