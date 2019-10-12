#Task 2: Alpha blending

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util

import matplotlib.pyplot as plt

def hard_blending(im1, im2):
  """
  return an image that consist of the left-half of im1
  and right-half of im2
  """
  assert(im1.shape == im2.shape)
  h, w, c = im1.shape
  new_im = im1.copy()
  new_im[:,:(w//2),:] = im2[:,:(w//2),:]
  return new_im

def alpha_blending(im1, im2, window_size=0.5):
  """
  return a new image that smoothly combines im1 and im2
  im1: np.array image of the dimensions: height x width x channels; values: 0-1 
  im2: np.array same dim as im1
  window_size: what fraction of image width to use for the transition (0-1)
  """
  # useful functions: np.linspace and np.concatenate
  assert(im1.shape == im2.shape)
  # TODO: Put your code below
  

if __name__ == "__main__":
  # TODO: Replace with your own images
  im1 = io.imread(path.join('images','cat_aligned.png'))
  im1 = util.img_as_float(im1[:,:,:3])
  im2 = io.imread(path.join('images','dog_aligned.png'))
  im2 = util.img_as_float(im2[:,:,:3])
  plt.figure(figsize=(15, 12))


  plt.subplot(221)
  plt.title('left image')
  plt.axis('off')
  plt.imshow(im1)

  plt.subplot(222)
  plt.title('right image')
  plt.axis('off')
  plt.imshow(im2)

  plt.subplot(223)
  plt.title('hard blending')
  plt.axis('off')
  plt.imshow(hard_blending(im1, im2))

  plt.subplot(224)
  plt.title('alpha blending')
  plt.axis('off')
  plt.imshow(alpha_blending(im1, im2, window_size=0.2))

  plt.show()
