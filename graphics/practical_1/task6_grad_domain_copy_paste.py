#Task 6: Gradient domain copy & paste

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
from scipy import ndimage
from sksparse.cholmod import cholesky
import time


if __name__ == "__main__":

    # Read background image
    bg = io.imread(path.join('images','nantes_river_sm.jpg'))[:,:,:3]
    bg = skimage.img_as_float(bg)
    # Read foreground image
    fg = io.imread(path.join('images','whale.png'))
    fg = skimage.img_as_float(fg)
    # Calculate alpha mask
    mask = (fg[:,:,3] > 0.5).astype(int)
    fg = fg[:,:,:3] # drop alpha channel

    #TODO: Implement gradient-domain copy&paste. 
    # Formulate the problem as explained in the practical 1
    # descrption - solve only for the pixels that are pasted.
    

    # Naive copy-paste for comparision
    mask3 = np.reshape(mask,[mask.shape[0], mask.shape[1], 1]) 
    I_naive = fg*mask3 + bg*(1-mask3)
    
    plt.figure(figsize=(9, 9))

    plt.subplot(121)
    plt.title('Naive')
    plt.axis('off')
    io.imshow(I_naive)

    plt.subplot(122)
    plt.title('Poisson Blending')
    plt.axis('off')
    io.imshow(I_dest)

    plt.show()

    io.imsave(path.join('results','copy_paste.jpg'), I_dest)
