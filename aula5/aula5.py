import numpy as np
import matplotlib.pyplot as plt
import cv2

def imageConvolution(img, kernel):
    h, w = img.shape
    kw, kh = kernel.shape
    resultado = np.zeros((h,w))
    valores = np.zeros((h + 2, w + 2))

    for i in range(1, valores.shape[0] - 1):
        for j in range(1, valores.shape[1] - 1):
            valores[i, j] = img[i-1, j-1]
    
    for y in range(h):
        for x in range(w):
            soma = np.zeros((kh,kw))
            for ky in range(kw):
                for kx in range(kh):
                    soma[ky,kx] = valores[y+ky,x+kx]
            resultado[y, x] = np.sum(soma*kernel)

    return resultado


img1 = cv2.imread('images/circuito.tif')
img2 = cv2.imread('images/pontos.png')
img3 = cv2.imread('images/linhas.png')
img4 = cv2.imread('images/igreja.png')
img5 = cv2.imread('images/root.jpg')
img6 = cv2.imread('images/harewood.jpg')
img7 = cv2.imread('images/nuts.jpg')
img8 = cv2.imread('images/snow.jpg')
img_aluno = cv2.imread('images/img_aluno.jpg')

img1_PB = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_PB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3_PB = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4_PB = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5_PB = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
img6_PB = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
img7_PB = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
img8_PB = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
img_alunoPB = cv2.cvtColor(img_aluno, cv2.COLOR_BGR2GRAY)

#1 ruido pimenta e sal

plt.figure(figsize=(8,8))
plt.subplot(111)
plt.imshow(img1_PB, cmap="gray")
plt.axis('off')
plt.show()

for i in range(3):
    circuitoFiltrado = cv2.medianBlur(img1_PB, 3)
    img1_PB = circuitoFiltrado
    plt.figure(figsize=(16,16))
    plt.subplot(111)
    plt.title('Imagem Filtrada -- ruído sal e pimenta')
    plt.imshow(img1_PB, cmap="gray")
    plt.axis('off')
    plt.show()

#2 aplicar filtro

filtro = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])


pontosFiltrados = cv2.filter2D(img2_PB, -1, filtro)
_, pontosLimiar = cv2.threshold(pontosFiltrados, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img2_PB, cmap="gray")
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Pontos Destacados')
plt.imshow(pontosLimiar, cmap="gray")
plt.axis('off')
plt.show()

#3 aplicar filtros

plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.title('img3')
plt.imshow(img3_PB, cmap="gray")
plt.axis('off')
plt.show()

filtroH= np.array([[-1, -1, -1],
                              [ 2,  2,  2],
                              [-1, -1, -1]], dtype=np.float32)

filtro_45 = np.array([[-1, -1,  2],
                            [-1,  2, -1],
                            [ 2, -1, -1]], dtype=np.float32)

filtroV = np.array([[-1, 2, -1],
                            [-1, 2, -1],
                            [-1, 2, -1]], dtype=np.float32)

filtro_menos_45 = np.array([[ 2, -1, -1],
                                  [-1,  2, -1],
                                  [-1, -1,  2]], dtype=np.float32)

resultadoH = imageConvolution(img3_PB,filtroH)
resultado_45 = imageConvolution(img3_PB,filtro_45)
resultadoV = imageConvolution(img3_PB,filtroV)
resultado_menos_45 = imageConvolution(img3_PB,filtro_menos_45)

limiar = 127
_, limiarizadaH = cv2.threshold(resultadoH, limiar, 255, cv2.THRESH_BINARY)
_, limiarizada_45 = cv2.threshold(resultado_45, limiar, 255, cv2.THRESH_BINARY)
_, limiarizadaV= cv2.threshold(resultadoV, limiar, 255, cv2.THRESH_BINARY)
_, limiarizada_menos_45 = cv2.threshold(resultado_menos_45, limiar, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.title('Linhas Horizontais')
plt.imshow(limiarizadaH, cmap="gray")
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Linhas 45 Graus')
plt.imshow(limiarizada_45, cmap="gray")
plt.axis('off')

plt.subplot(2,2,3)
plt.title('Linhas Verticais')
plt.imshow(limiarizadaV, cmap="gray")
plt.axis('off')

plt.subplot(2,2,4)
plt.title('Linhas -45 Graus')
plt.imshow(limiarizada_menos_45, cmap="gray")
plt.axis('off')
plt.show()

#4 bordas de Canny
limiar_min = 100
limiar_max = 200
igrejaBordasCanny = cv2.Canny(img4_PB, limiar_min, limiar_max)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img4_PB, cmap="gray")
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Bordas de Canny')
plt.imshow(igrejaBordasCanny, cmap="gray")
plt.axis('off')
plt.show()

#5 Crescimento de Região

img5_Blur = cv2.GaussianBlur(img5_PB,(5,5),0)
_,thresholdF = cv2.threshold(img5_Blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

mask = np.zeros((img5.shape[0] + 2, img5.shape[1] + 2), np.uint8)

seed_point = (422, 269)
color = (0,0,0)
minColor = color
maxColor = (128,128,128)

plt.figure(figsize=(16,16))

plt.subplot(221)
plt.imshow(img5_PB, cmap="gray")
plt.axis('off')

plt.subplot(222)
plt.imshow(thresholdF, cmap="gray")
plt.axis('off')

cv2.floodFill(thresholdF, mask, seed_point, color, minColor, maxColor)

for y in range(img5.shape[0]):
    for x in range(img5.shape[1]):
        if mask[y, x].all():
            img5_PB[y, x] = 0


plt.subplot(223)
plt.imshow(thresholdF, cmap="gray")
plt.axis('off')

plt.subplot(224)
plt.imshow(img5_PB, cmap="gray")
plt.axis('off')
plt.show()

#6 limiarização do Método de Otsu
_, img6_limiarizada = cv2.threshold(img6_PB, 0, 255, cv2.THRESH_OTSU)
_, img7_limiarizada = cv2.threshold(img7_PB, 0, 255, cv2.THRESH_OTSU)
_, img8_limiarizada = cv2.threshold(img8_PB, 0, 255, cv2.THRESH_OTSU)
_, img_aluno_limiarizada = cv2.threshold(img_alunoPB, 0, 255, cv2.THRESH_OTSU)

plt.figure(figsize=(8,8))
plt.subplot(4,4,1)
plt.title('img6')
plt.imshow(img6_PB, cmap="gray")
plt.axis('off')

plt.subplot(4,4,2)
plt.title('img7')
plt.imshow(img7_PB, cmap="gray")
plt.axis('off')

plt.subplot(4,4,3)
plt.title('img8')
plt.imshow(img8_PB, cmap="gray")
plt.axis('off')

plt.subplot(4,4,4)
plt.title('img_aluno')
plt.imshow(img_alunoPB, cmap="gray")
plt.axis('off')

plt.subplot(4,4,5)
plt.imshow(img6_limiarizada, cmap="gray")
plt.axis('off')

plt.subplot(4,4,6)
plt.imshow(img7, cmap="gray")
plt.axis('off')

plt.subplot(4,4,7)
plt.imshow(img8_limiarizada, cmap="gray")
plt.axis('off')

plt.subplot(4,4,8)
plt.imshow(img_aluno_limiarizada, cmap="gray")
plt.axis('off')
plt.show()