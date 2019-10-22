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
import time

from task4_grad_domain import img2grad_field, reconstruct_grad_field

if __name__ == "__main__":
    #TODO: Replace with your own image
    #im = io.imread(path.join('images','rubberduck.jpg'))
    im = io.imread(path.join('images','mountain4-1.jpg'))
    im = skimage.img_as_float(im)

    im_gray = color.rgb2gray(im)

    G = img2grad_field(im_gray)

    # Scale the gradient field by the interpolating function f
    f = interp1d([-1,-0.1,0, 0.1, 1], [-1,-0.15,0,0.15,1])
    interpolated = f(G)

    Gm = np.sqrt(np.sum(G*G, axis=2))
    #w = np.ones (img.shape)
    w = 1/(Gm + 0.0001)     # To avoid pinching artefacts

    start = time.time()
    imr = reconstruct_grad_field(interpolated,w,im_gray[0,0], im_gray).clip(0,1)

    # Reconstruct colour by scaling colour
    rows = imr.shape[0]
    cols = imr.shape[1]
    imr_color = np.zeros(im.shape)
    for x in range(rows):
        for y in range(cols):
            # Calculate scale for pixel, then scale pixel r, g and b by that value
            scale = imr[x][y] / im_gray[x][y]
            imr_color[x][y] = np.multiply(im[x][y], scale)
            imr_color[x][y] = np.clip(imr_color[x][y], 0, 1)
    end = time.time()
    print(end - start)

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
