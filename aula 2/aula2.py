import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img1 = cv2.imread('lena.png')
img2 = cv2.imread('unequalized.jpg')
img3 = cv2.imread('img_aluno.jpg')

#1-Implemente o algoritmo de conversão para níveis de cinza e converta as imagens. 
img_pb1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_pb2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img_pb3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb1,cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb2,cv2.COLOR_BGR2RGB))
plt.subplot(223)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb3,cv2.COLOR_BGR2RGB))

plt.show()

#2-gerar a imagem negativa lena e img_aluno
img_n1=255-img1
img_n3=255-img3

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_n1,cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_n3,cv2.COLOR_BGR2RGB))

plt.show()

#3-normalizaçao da imagem lena e img_aluno
normalizedImg1 = img_pb1
normalizedImg3 = img_pb3
img_norm1 = cv2.normalize(img1,normalizedImg1,0,100,cv2.NORM_MINMAX)
img_norm3 = cv2.normalize(img3,normalizedImg3,0,100,cv2.NORM_MINMAX)

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(cv2.cvtColor(img_norm1,cv2.COLOR_BGR2RGB),cmap="gray")
plt.axis('off')
plt.subplot(222)
plt.imshow(cv2.cvtColor(img_norm3,cv2.COLOR_BGR2RGB),cmap="gray")
plt.axis('off')

plt.show()

#4-imagem operador logaritmico lena e img_aluno
z = np.arange(256)
c_scale = 255/(np.log2(1+255))

z_log2=c_scale*np.log2(z+1)

imgL1 = ((c_scale*np.log2(img1.astype(np.uint32)+1))).astype(np.uint8)
imgL3 = ((c_scale*np.log2(img3.astype(np.uint32)+1))).astype(np.uint8)
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(imgL1,cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(imgL3,cv2.COLOR_BGR2RGB))

plt.show()

#5-1Operador logístico lena e img_aluno
k=0.045
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
img_sig1 = (255/(1+np.exp(-k*(img1.astype(np.int32)-127)))).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_sig1,cv2.COLOR_BGR2RGB))

plt.subplot(222)
plt.axis("off")
img_sig3 = (255/(1+np.exp(-k*(img3.astype(np.int32)-127)))).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_sig3,cv2.COLOR_BGR2RGB))

plt.show()

plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))

plt.show()

#5-2-a-histograma
hist = cv2.calcHist([img_pb2],[0],None,[256],[0,256])
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb2, cv2.COLOR_GRAY2RGB))
plt.subplot(222)
# hist/= hist.sum() normalizado
plt.title("Histograma escala de cinza")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist)
plt.xlim([0, 256])

plt.show()

#5-2-b-histograma normalizado
hist = cv2.calcHist([img_pb2],[0],None,[256],[0,256])
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb2, cv2.COLOR_GRAY2RGB))
plt.subplot(222)
hist/= hist.sum()
plt.title("Histograma escala de cinza(normalizado)")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist)
plt.xlim([0, 256])

plt.show()

# #histograma no canal Azul
chans=cv2.split(img3)
colors =("b","w","w")

plt.figure(figsize=(16,16))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.title("Histograma escala de cinza img_aluno")
plt.xlabel("bins")
plt.ylabel("pixels")

plt.show()

plt.subplot(221)
plt.title("Histograma escala de B")
plt.xlabel("bins")
plt.ylabel("pixels")
for (chan, color) in zip(chans, colors):
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256])
plt.show()

# #histograma no canal Verde
colors =("w","g","w")
plt.subplot(221)
plt.title("Histograma escala de G")
plt.xlabel("bins")
plt.ylabel("pixels")
for (chan, color) in zip(chans, colors):
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256])
plt.show()
	
# #histograma no canal Vermelho
colors =("w","w","r")
plt.subplot(221)
plt.title("Histograma escala de R")
plt.xlabel("bins")
plt.ylabel("pixels")
for (chan, color) in zip(chans, colors):
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256])
plt.show()

#histograma em escala de cinza img_aluno
hist = cv2.calcHist([img_pb3],[0],None,[256],[0,256])
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb3, cv2.COLOR_GRAY2RGB))
plt.subplot(222)
# hist/= hist.sum() normalizado
plt.title("Histograma escala de cinza")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist)
plt.xlim([0, 256])

plt.show()

# #Histograma equalizado
plt.figure(figsize=(20,10))
equalized1 = cv2.equalizeHist(img_pb1)
equalized2 = cv2.equalizeHist(img_pb2)
equalized3 = cv2.equalizeHist(img_pb3)

plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb1, cv2.COLOR_GRAY2RGB))
plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(equalized1, cv2.COLOR_GRAY2RGB))
hist1 = cv2.calcHist([equalized1],[0],None,[256],[0,256])
plt.subplot(223)
plt.title("Histograma escala de cinza(equalizado)")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist1)

plt.show()

plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb2, cv2.COLOR_GRAY2RGB))
plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(equalized2, cv2.COLOR_GRAY2RGB))
hist2 = cv2.calcHist([equalized2],[0],None,[256],[0,256])
plt.subplot(223)
plt.title("Histograma escala de cinza(equalizado)")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist2)

plt.show()

plt.subplot(221)
plt.axis("off")
plt.imshow(cv2.cvtColor(img_pb3, cv2.COLOR_GRAY2RGB))
plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(equalized3, cv2.COLOR_GRAY2RGB))
hist3 = cv2.calcHist([equalized3],[0],None,[256],[0,256])
plt.subplot(223)
plt.title("Histograma escala de cinza(equalizado)")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(hist3)
plt.subplots_adjust(top=0.92, bottom=0.01, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.show()