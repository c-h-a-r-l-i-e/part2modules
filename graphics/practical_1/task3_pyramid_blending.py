#Task 3: Pyramid blending

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt

from task2_alpha_blending import alpha_blending

def laplacian_pyramid(img, levels=4, sigma=1):
  """
  Decompose an image into a laplcaian pyramid without decimation (reducing resolution)
  img - greyscale image to decompose
  levels - how many levels of the pyramid should be created
  sigma - the standard deviation to use for the Gaussian low-pass filter
  return array of the pyramid levels, each the same resolution as input. The sum of these 
         images should produce the input image.
  """
  pyramid = []
  #TODO: Implement decomposition into a laplacian pyramid
  
  return pyramid

def pyramid_blending(im1, im2, levels=4, sigma=1, window_size=0.3):
  #TODO: Implement pyramid blending
  return im1
  

if __name__ == "__main__":

  #Part 1: Laplacian pyramid decomposition
  #TODO: Replace with your own image
  im = io.imread(path.join('images','cat_aligned.png'))
  im = util.img_as_float(im[:,:,:3])
  im = color.rgb2grey(im)
  pyramid = laplacian_pyramid(im, levels=4)

  plt.figure(figsize=(3*len(pyramid), 3))
  grid = len(pyramid) * 10 + 121
  for i, layer in enumerate(pyramid):
    plt.subplot(grid+i)
    plt.title('level {}'.format(i))
    plt.axis('off')
    if i == len(pyramid)-1:
      io.imshow(layer)
    else:
      plt.imshow(layer)
    
  plt.subplot(grid+len(pyramid))
  plt.title('reconstruction')
  plt.axis('off')
  io.imshow(sum(pyramid))

  plt.subplot(grid+len(pyramid)+1)
  plt.title('differences')
  plt.axis('off')
  plt.imshow(im - sum(pyramid))
  plt.show()  

  # Part 2: Pyramid blending
  #TODO: Replace with your own images
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
  plt.title('alpha blend')
  plt.axis('off')
  plt.imshow(alpha_blending(im1, im2, window_size=0.3))

  plt.subplot(224)
  plt.title('pyramid blend')
  plt.axis('off')
  plt.imshow(pyramid_blending(im1, im2, window_size=0.3))
  plt.show()
