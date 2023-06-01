import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

img1 = cv2.imread('img_aluno1.jpg')
img2 = cv2.imread('img_aluno2.jpg')

img_pb1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_pb2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#1 suavização média
blur = cv2.blur(img_pb1,(15,15))

plt.subplot(121), plt.imshow(cv2.cvtColor(img_pb1,cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.show()

blur1 = cv2.blur(img_pb2,(15,15))

plt.subplot(121), plt.imshow(cv2.cvtColor(img_pb2,cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(blur1,cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.show()


#filtro laplaciano
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(img1, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado = np.vstack([img1, lap]) 
plt.imshow(cv2.cvtColor(resultado,cv2.COLOR_BGR2RGB))
plt.show()

#bordas roberts
roberts_cross_v = np.array( [[1, 0 ],
                             [0,-1 ]] )
  
roberts_cross_h = np.array( [[ 0, 1 ],
                             [ -1, 0 ]] )

img = cv2.imread("img_aluno2.jpg",0).astype('float64')
img/=255.0
vertical = ndimage.convolve(img, roberts_cross_v )
horizontal = ndimage.convolve(img, roberts_cross_h )

plt.figure(figsize=(10,10))
plt.subplot(221)
edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img*=255
plt.imshow(edged_img)
plt.show()

#bordas de prewitt
#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img1, -1, kernelx)
img_prewitty = cv2.filter2D(img1, -1, kernely)


cv2.imshow("Original Image", img1)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)

plt.show()

#sobel
sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img1,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()