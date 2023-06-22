import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift

img1 = cv2.imread('arara_filtro.png')
img2 = cv2.imread('arara.png')
img3 = cv2.imread('barra1.png')
img4 = cv2.imread('barra2.png')
img5 = cv2.imread('barra3.png')
img6 = cv2.imread('barra4.png')
img7 = cv2.imread('quadrado.png')
img8 = cv2.imread('teste.png')


sizes = 64
img1_s = img1
img2_s = img2[0:sizes, 0:sizes]
img3_s = img3[5:sizes+5, 5:sizes+5]

F1s = np.fft.fft2(img1_s)
F2s = np.fft.fft2(img2_s)
F3s = np.fft.fft2(img3_s)

plt.figure(figsize=(12,8)) 
plt.subplot(231)
plt.imshow(img1_s, cmap="gray"); plt.axis('off')
plt.subplot(232)
plt.imshow(img2_s, cmap="gray"); plt.axis('off')
plt.subplot(233)
plt.imshow(img3_s, cmap="gray"); plt.axis('off')

# the log of the magnitudes 
plt.subplot(234)
plt.imshow(np.log(1 + np.fft.fftshift(np.abs(F1s))), cmap="gray")
plt.axis('off')
plt.subplot(235)
plt.imshow(np.log(1 + np.fft.fftshift(np.abs(F2s))), cmap="gray")
plt.axis('off')
plt.subplot(236)
plt.imshow(np.log(1 + np.fft.fftshift(np.abs(F3s))), cmap="gray")
plt.axis('off')

n2 = F1s.shape[0]//2
m2 = F1s.shape[1]//2

F1p = np.fft.fft2.fftshift(F1s).copy()
F1p[n2-9:n2+9, m2-9:m2+9] = 0 # square high pass filter, removes first frequencies
F1p = np.fft.fft2.ifftshift(F1p)
    
F2p = np.fft.fft2.fftshift(F2s).copy()
F2p[:n2-9, :] = 0 # square low pass filter, removes higher frequencies
F2p[:, :m2-9] = 0 # square low pass filter, removes higher frequencies
F2p[n2+9:, :] = 0 # square low pass filter, removes higher frequencies
F2p[:, m2+9:] = 0 # square low pass filter, removes higher frequencies
F2p = np.fft.ifftshift(F2p)

F3p = F3s.copy()
F3p[5:-5,5:-5] = 0 # band stop filter
#F3p = np.fft.ifftshift(F3p)

i1p = np.fft.ifft2(F1p)
i2p = np.fft.ifft2(F2p)
i3p = np.fft.ifft2(F3p)

g = f + gaussian_noise(f.shape, mean=0, std=0.08)
g = np.clip(g.astype(int), 0, 255)

K = 7
w_mean = np.ones([K,K])/float(K*K)
def fft_imagefilter(g, w):
    ''' A function to filter an image g with the filter w
    '''
    # padding the filter so that it has the same size of the image
    pad1 = (g.shape[0]//2)-w.shape[0]//2
    wp = np.pad(w, (pad1,pad1-1), "constant",  constant_values=0)

    # computing the Fourier transforms
    W = fftn(wp)
    G = fftn(g)
    R = np.multiply(W,G)
    
    r = np.real(fftshift(ifftn(R)))
    return r

r_mean = fft_imagefilter(g, w_mean)


def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H


def gaussian_filter(k=5, sigma=1.0):
    ''' Gaussian filter
    :param k: defines the lateral size of the kernel/filter, default 5
    :param sigma: standard deviation (dispersion) of the Gaussian distribution
    :return matrix with a filter [k x k] to be used in convolution operations
    '''
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)

