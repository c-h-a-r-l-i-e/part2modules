#Task 4: Gradient domain reconstruction

import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sksparse.cholmod import cholesky
from scipy import signal
import skimage
import time

USE_CHOLESKY = True

def img2grad_field(img):
    """Return a gradient field for a greyscale image
    The function returns image [height,width,2], where the last dimension selects partial derivates along x or y
    """
    # img must be a greyscale image
    sz = img.shape
    G = np.zeros([sz[0], sz[1], 2])
    # Gradients along x-axis
    G[:,:,0] = signal.convolve2d( img, np.array([1, -1, 0]).reshape(1,3), 'same', boundary='symm' )
    # Gradients along y-axis
    G[:,:,1] = signal.convolve2d( img,  np.array([1, -1, 0]).reshape(3,1), 'same', boundary='symm' )
    return G

def reconstruct_grad_field( G, w, v_00, img ):
    """Reconstruct a (greyscale) image from a gradcient field
    G - gradient field, for example created with img2grad_field
    w - weight assigned to each gradient
    v_00 - the value of the first pixel 
    """
    sz = G.shape[:2] 
    N = sz[0]*sz[1]

    # Gradient operators as sparse matrices
    o1 =  np.ones((N,1))
    B = np.concatenate( (-o1, np.concatenate( (np.zeros((sz[0],1)), o1[:N-sz[0]]), 0 ) ), 1)
    B[N-sz[0]:N,0] = 0
    Ogx = sparse.spdiags(B.transpose(), [0 ,sz[0]], N, N ) # Forward difference operator along x

    B = np.concatenate( (-o1 ,np.concatenate((np.array([[0]]), o1[0:N-1]) ,0)), 1)
    B[sz[0]-1::sz[0], 0] = 0
    B[sz[0]::sz[0],1] = 0
    Ogy = sparse.spdiags( B.transpose(), [0, 1], N, N ) # Forward difference operator along y

    Ogx_prime = Ogx.transpose()
    Ogy_prime = Ogy.transpose()

    w_diag = sparse.spdiags([w.flatten('F')], [0], N, N)

    # Introduce constraint to ensure value of first pixel remains the same
    C = sparse.spdiags([[1]], [0], N, N)

    A = Ogx_prime @ w_diag @ Ogx + Ogy_prime @ w_diag @ Ogy +  C

    Gx = G[:,:,0].flatten('F')
    Gy = G[:,:,1].flatten('F')
    C2 = np.zeros((N,)) # Another constraint to ensure value of first pixel remains the same
    C2[0] = v_00

    b = Ogx_prime @ w_diag @ Gx + Ogy_prime @ w_diag @ Gy + C2

    if USE_CHOLESKY:
        factor = cholesky(A)
        I = factor(b)

    else:
        I = sparse.linalg.spsolve(A, b)


    new_img = I.reshape(img.shape, order='F')

    return new_img


if __name__ == "__main__":
    im = io.imread(path.join('images','house.jpg'), as_gray=True)
    im = skimage.img_as_float(im)

    G = img2grad_field(im)
    Gm = np.sqrt(np.sum(G*G, axis=2))
    
    #w = np.ones (img.shape)
    w = 1/(Gm + 0.0001)     # To avoid pinching artefacts

    start = time.time()
    imr = reconstruct_grad_field(G,w,im[0,0], im).clip(0,1)
    end = time.time()

    print("Time to reconstruct field {}".format(end-start))

    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(im,cmap='gray')

    plt.subplot(132)
    plt.title('Reconstructed')
    plt.axis('off')
    plt.imshow(imr,cmap='gray')

    plt.subplot(133)
    plt.title('Difference')
    plt.axis('off')
    plt.imshow(imr-im)
    plt.show()
