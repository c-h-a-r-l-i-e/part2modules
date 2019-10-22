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
from task4_grad_domain import img2grad_field

USE_CHOLESKY=True


def reconstruct_grad_domain( G, mask, bg ):
    """Reconstruct a (greyscale) image from a gradcient domain
    G - gradient field, for example created with img2grad_field
    w - weight assigned to each gradient
    mask - mask used to filter the image
    bg - background image
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

    # Calculate pixels which are in the boundary
    boundary = []
    h, w = mask.shape
    for x in range(h):
        for y in range(w):
            if mask[x,y] == 1 and (x==0 or x==h-1 or y==0 or y==w-1 or min(mask[x,y-1], mask[x-1,y], mask[x+1,y], mask[x,y+1])==0):
                boundary.append((x,y))

    K = len(boundary)
    # TODO: make this work just on place we need it!!!

    E = sparse.csc_matrix((K, N))
    T_E = np.zeros((K,))
    for i, (x,y) in enumerate(boundary):
        pos = y * sz[0] + x # Calulcate the pixel's position
        E[i, pos] = 1
        T_E[i] = bg[x,y]

    A = Ogx_prime @ Ogx + Ogy_prime @ Ogy +  E.transpose() @ E

    Gx = G[:,:,0].flatten('F')
    Gy = G[:,:,1].flatten('F')

    b = Ogx_prime @ Gx + Ogy_prime @ Gy + E.transpose() @ T_E

    if USE_CHOLESKY:
        factor = cholesky(A)
        I = factor(b)

    else:
        I = sparse.linalg.spsolve(A, b)


    new_img = I.reshape(bg.shape, order='F')

    return new_img


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

    new_fg = fg.copy()
    for i in range(3): #For R, G and B
        fg_i = fg[:,:,i]
        G = img2grad_field(fg_i)
        fgr = reconstruct_grad_domain(G, mask, bg[:,:,i])
        new_fg[:,:,i] = fgr

    mask3 = np.reshape(mask,[mask.shape[0], mask.shape[1], 1]) 
    I_dest = new_fg *mask3 + bg*(1-mask3)

    



 

    # Naive copy-paste for comparision
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
