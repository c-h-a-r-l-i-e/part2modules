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
  new_im = im2.copy()
  new_im[:,:(w//2),:] = im1[:,:(w//2),:]
  return new_im

def alpha_blending(im1, im2, window_size=0.5):
  """
  return a new image that smoothly combines im1 and im2
  im1: np.array image of the dimensions: height x width x channels; values: 0-1 
  im2: np.array same dim as im1
  window_size: what fraction of image width to use for the transition (0-1)
  """
  assert(im1.shape == im2.shape)

  columns = im1.shape[1]
  rows = im1.shape[0]
  transition_size = int(columns * window_size)
  im1_size = (columns - transition_size) // 2
  im2_size = columns - transition_size - im1_size
  
  # alpha is a matrix which describes how much of im1 we want to display
  alpha = np.concatenate((np.ones((im1_size)), np.linspace(1, 0, transition_size), np.zeros((im2_size))))
  
  new_im = im1.copy()
  for x in range(rows):
    # Calculates Iblend(x,y) =α(x,y)Ileft(x,y) + (1−α(x,y))Iright(x,y)
    new_im[x] = (im1[x] * alpha[:, None]) + ((np.ones([columns]) - alpha)[:, None] * im2[x])

  return new_im
  

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
