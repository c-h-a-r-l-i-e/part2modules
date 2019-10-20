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
  # Split the pixels into bins
  image = np.multiply(image, bin_count - 1)
  image = np.floor(image).astype(int)

  # Calculate the histogram - a 2D array where the ith element is the number of pixels of value i, then the 
  # normalized cumulative histogram
  histogram = np.bincount(image.flatten())
  cum_histogram = np.cumsum(histogram)
  normalized = np.divide(cum_histogram, image.size, dtype=float)
  
  # Iterate through pixels, calculating new pixel value by using the value of the old pixel to index into the
  # cumulative normalized histogram
  new_image = np.zeros(image.shape)
  rows = image.shape[0]
  cols = image.shape[1]
  for x in range(rows):
    for y in range(cols):
      new_image[x][y] = normalized[image[x][y]]

  return new_image

# TODO: Change the file to your own image
img_name = 'house.jpg'

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
  # Seperate out the r, g and b channels
  img_r = img[:,:,0]
  img_g = img[:,:,1]
  img_b = img[:,:,2]

  # Run histogram equalization on each channel
  img_r = equalise_hist(img_r)
  img_g = equalise_hist(img_g)
  img_b = equalise_hist(img_b)

  # Merge the channels
  new_img = np.dstack((img_r, img_g, img_b))
  return new_img
  

def he_colour_ratio(img):
  # Perform histogram equalization on a gray-scale image and transfer colour using colour ratios.
  img_gray_old = color.rgb2gray(img)
  img_grey_new = equalise_hist(img_gray_old)

  rows = img.shape[0]
  cols = img.shape[1]
  new_img = np.zeros(img.shape)
  for x in range(rows):
    for y in range(cols):
      # Calculate scale for pixel, then scale pixel r, g and b by that value
      scale = img_grey_new[x][y] / img_gray_old[x][y]
      new_img[x][y] = np.multiply(img[x][y], scale)
      new_img[x][y] = np.clip(new_img[x][y], 0, 1)
  
  return new_img
  

def he_hsv(img):
  # Perform histogram equalization by processing channel V in the HSV colourspace.
  img_hsv = color.rgb2hsv(img)
  img_hsv_v = img_hsv[:,:,2]
  img_hsv_v = equalise_hist(img_hsv_v)
  new_img = np.dstack((img_hsv[:,:,0], img_hsv[:,:,1], img_hsv_v))
  new_img = color.hsv2rgb(new_img)
  return new_img
  


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
