# Task 1: Image enhancement
import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util

import matplotlib.pyplot as plt

# useful functions: np.bincount and np.cumsum

def equalise_hist(image, bin_count=256):
  """
  Perform histogram equalization on an image and return as a new image.

  Arguments:
  image -- a numpy array of shape height x width, dtype float, range between 0 and 1
  bin_size -- how many bins to use
  """
  print(image)
  # Split the image into bins
  image = np.multiply(image, bin_count - 1)
  image = np.floor(image).astype(int)

  # Calculate the histogram - a 2D array where the ith element is the number of pixels of value i
  histogram = np.bincount(image.flatten())

  cum_histogram = np.cumsum(histogram)

  normalized = np.divide(cum_histogram, image.size, dtype=float)


  print(normalized)




  
  
  #return new_image


# TODO: Change the file to your own image
img_name = 'snow.jpg'

test_im = io.imread(path.join('images',img_name))

test_im_gray = color.rgb2gray(test_im)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.title('Original image')
plt.axis('off')
plt.imshow(test_im_gray,cmap='gray')

plt.subplot(122)
plt.title('Histogram equalised image')
plt.axis('off')
plt.imshow(equalise_hist(test_im_gray),cmap='gray')

plt.show()

def he_per_channel(img):
  # Perform histogram equalization separately on each colour channel. 
  # TODO: put your code below
  return img
  

def he_colour_ratio(img):
  # Perform histogram equalization on a gray-scale image and transfer colour using colour ratios.
  # TODO: put your code below
  return img
  

def he_hsv(img):
  # Perform histogram equalization by processing channel V in the HSV colourspace.
  # TODO: put your code below
  return img
  


test_im = io.imread(path.join('images',img_name))
test_im = util.img_as_float(test_im)
plt.figure(figsize=(15, 12))

plt.subplot(221)
plt.title('Original image')
plt.axis('off')
io.imshow(test_im)

plt.subplot(222)
plt.title('Each channel processed seperately')
plt.axis('off')
io.imshow(he_per_channel(test_im))

plt.subplot(223)
plt.title('Gray-scale + colour ratio')
plt.axis('off')
io.imshow(he_colour_ratio(test_im))

plt.subplot(224)
plt.title('Processed V in HSV')
plt.axis('off')
io.imshow(he_hsv(test_im))

plt.show()
