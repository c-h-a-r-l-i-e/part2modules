#Task 5: Gradient domain image enhancement
import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import signal
import skimage
from scipy.interpolate import interp1d
from skimage import color

from task4_grad_domain import img2grad_field, reconstruct_grad_field

if __name__ == "__main__":
    #TODO: Replace with your own image
    #im = io.imread(path.join('images','rubberduck.jpg'))
    im = io.imread(path.join('images','mountain.jpg'))
    im = skimage.img_as_float(im)

    im_gray = color.rgb2gray(im)

    G = img2grad_field(im_gray)

    #TODO: Implement gradient domain enhancement on the greyscale image, then recover colour
    #Hint: Use reconstruct_grad_field from the previous task
    

    plt.figure(figsize=(9, 3))

    plt.subplot(121)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(im)

    plt.subplot(122)
    plt.title('Enhanced')
    plt.axis('off')
    plt.imshow(imr_color)

    plt.show()

    io.imsave(path.join('results','gd_enhanced.jpg'), imr_color)
