import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
#Fourier

img1 = cv2.imread('./images/arara.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/barra1.png', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('./images/barra2.png', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('./images/barra3.png', cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('./images/barra4.png', cv2.IMREAD_GRAYSCALE)
img6 = cv2.imread('./images/quadrado.png', cv2.IMREAD_GRAYSCALE)
img7 = cv2.imread('./images/teste.png', cv2.IMREAD_GRAYSCALE)       


# Aplica a transformada de Fourier
f1 = np.fft.fft2(img1)
f1_shift = np.fft.fftshift(f1)

f2 = np.fft.fft2(img2)
f2_shift = np.fft.fftshift(f2)

f3 = np.fft.fft2(img3)
f3_shift = np.fft.fftshift(f3)

f4 = np.fft.fft2(img4)
f4_shift = np.fft.fftshift(f4)

f5 = np.fft.fft2(img5)
f5_shift = np.fft.fftshift(f5)

f6 = np.fft.fft2(img6)
f6_shift = np.fft.fftshift(f6)

f7 = np.fft.fft2(img7)
f7_shift = np.fft.fftshift(f7)


# Calcula o espectro de magnitude
magnitude_spectrum1 = 20 * np.log(np.abs(f1_shift))

magnitude_spectrum2 = 20 * np.log(np.abs(f2_shift))

magnitude_spectrum3 = 20 * np.log(np.abs(f3_shift))

magnitude_spectrum4 = 20 * np.log(np.abs(f4_shift))

magnitude_spectrum5 = 20 * np.log(np.abs(f5_shift))

magnitude_spectrum6 = 20 * np.log(np.abs(f6_shift))

magnitude_spectrum7 = 20 * np.log(np.abs(f7_shift))

# Exibe a imagem original e o espectro de Fourier
plt.subplot(4,4,1)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4,4,3)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4,4,5)
plt.imshow(img3, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4,4,7)
plt.imshow(img4, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4,4,9)
plt.imshow(img5, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4,4,11)
plt.imshow(img6, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4,4,13)
plt.imshow(img7, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(4, 4, 2)
plt.imshow(magnitude_spectrum1, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 4)
plt.imshow(magnitude_spectrum2, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 6)
plt.imshow(magnitude_spectrum3, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 8)
plt.imshow(magnitude_spectrum4, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 10)
plt.imshow(magnitude_spectrum5, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 12)
plt.imshow(magnitude_spectrum6, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.subplot(4, 4, 14)
plt.imshow(magnitude_spectrum7, cmap='gray')
plt.axis('off')
plt.title('Espectro de Fourier')

plt.show()

#2 filtros passa-baixa, passa-alta, passa-banda e rejeita-banda
#passa-baixa

# Carrega a imagem
image = cv2.imread('./images/teste.png', cv2.IMREAD_GRAYSCALE)

# Configurações do filtro passa-baixa
cutoff_freq = 30

# Realiza a transformada de Fourier na imagem
f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

# Calcula as dimensões da imagem e o centro
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Cria uma máscara passa-baixa
mask = np.zeros((rows, cols), np.uint8)
mask[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] = 1

# Aplica a máscara na transformada de Fourier
f_shift_filtered = f_shift * mask

# Realiza a transformada inversa de Fourier
f_inverse = np.fft.ifftshift(f_shift_filtered)
image_filtered = np.fft.ifft2(f_inverse)
image_filtered = np.abs(image_filtered)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(image_filtered, cmap='gray')
plt.title('Filtrada (Passa-Baixa)')

plt.show()

#passa-alta
# Carrega a imagem
image = cv2.imread('./images/teste.png', cv2.IMREAD_GRAYSCALE)

# Configurações do filtro passa-alta
cutoff_freq = 30

# Realiza a transformada de Fourier na imagem
f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

# Calcula as dimensões da imagem e o centro
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Cria uma máscara passa-alta
mask = np.ones((rows, cols), np.uint8)
mask[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] = 0

# Aplica a máscara na transformada de Fourier
f_shift_filtered = f_shift * mask

# Realiza a transformada inversa de Fourier
f_inverse = np.fft.ifftshift(f_shift_filtered)
image_filtered = np.fft.ifft2(f_inverse)
image_filtered = np.abs(image_filtered)

# Normaliza a imagem filtrada para exibir corretamente
image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(image_filtered, cmap='gray')
plt.title('Filtrada (Passa-Alta)')

plt.show()

#passa-banda
# Carrega a imagem
image = cv2.imread('./images/teste.png', cv2.IMREAD_GRAYSCALE)

# Configurações do filtro passa-banda
cutoff_low = 30
cutoff_high = 70

# Realiza a transformada de Fourier na imagem
f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

# Calcula as dimensões da imagem e o centro
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Cria uma máscara passa-baixa
mask_low = np.zeros((rows, cols), np.uint8)
mask_low[crow - cutoff_low: crow + cutoff_low, ccol - cutoff_low: ccol + cutoff_low] = 1

# Cria uma máscara passa-alta
mask_high = np.ones((rows, cols), np.uint8)
mask_high[crow - cutoff_high: crow + cutoff_high, ccol - cutoff_high: ccol + cutoff_high] = 0

# Combina as máscaras para obter o filtro passa-banda
mask_bandpass = mask_low * mask_high

# Aplica a máscara na transformada de Fourier
f_shift_filtered = f_shift * mask_bandpass

# Realiza a transformada inversa de Fourier
f_inverse = np.fft.ifftshift(f_shift_filtered)
image_filtered = np.fft.ifft2(f_inverse)
image_filtered = np.abs(image_filtered)

# Normaliza a imagem filtrada para exibir corretamente
image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(image_filtered, cmap='gray')
plt.title('Filtrada (Passa-Banda)')

plt.show()

#rejeita-banda
# Configurações do filtro rejeita-banda
cutoff_low = 30
cutoff_high = 70

# Realiza a transformada de Fourier na imagem
f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

# Calcula as dimensões da imagem e o centro
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Cria uma máscara passa-baixa
mask_low = np.ones((rows, cols), np.uint8)
mask_low[crow - cutoff_low: crow + cutoff_low, ccol - cutoff_low: ccol + cutoff_low] = 0

# Cria uma máscara passa-alta
mask_high = np.zeros((rows, cols), np.uint8)
mask_high[crow - cutoff_high: crow + cutoff_high, ccol - cutoff_high: ccol + cutoff_high] = 1

# Combina as máscaras para obter o filtro rejeita-banda
mask_bandstop = mask_low * mask_high

# Aplica a máscara na transformada de Fourier
f_shift_filtered = f_shift * mask_bandstop

# Realiza a transformada inversa de Fourier
f_inverse = np.fft.ifftshift(f_shift_filtered)
image_filtered = np.fft.ifft2(f_inverse)
image_filtered = np.abs(image_filtered)

# Normaliza a imagem filtrada para exibir corretamente
image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(image_filtered, cmap='gray')
plt.title('Filtrada (Rejeita-Banda)')

plt.show()

#filtro gaussiano
# Carrega a imagem
image = cv2.imread('./images/teste.png', cv2.IMREAD_GRAYSCALE)

# Configurações do filtro de ruído gaussiano
mean = 0
stddev = 20

# Gera o ruído gaussiano
noise = np.random.normal(mean, stddev, size=image.shape)

# Aplica o ruído à imagem
image_noisy = image + noise

# Normaliza a imagem para o intervalo 0-255
image_noisy = cv2.normalize(image_noisy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem com ruído
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(image_noisy, cmap='gray')
plt.title('Imagem com Ruído Gaussiano')

plt.show()

#rejeita banda definido pela arara filtro png
# Carrega as imagens
imagem_original = cv2.imread('./images/arara.png')
filtro = cv2.imread('./images/arara_filtro.png', cv2.IMREAD_GRAYSCALE)

# Converte a imagem original para escala de cinza
imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

# Calcula a transformada de Fourier da imagem original e do filtro
fft_imagem = np.fft.fft2(imagem_original)
fft_filtro = np.fft.fft2(filtro, s=imagem_original.shape)

# Aplica a filtragem no domínio da frequência
fft_resultado = fft_imagem * (1 - fft_filtro)

# Calcula a transformada inversa de Fourier para obter a imagem filtrada
imagem_filtrada = np.fft.ifft2(fft_resultado).real

# Normaliza os valores da imagem filtrada para o intervalo de 0 a 255
imagem_filtrada = cv2.normalize(imagem_filtrada, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Exibe a imagem original e a imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(imagem_original, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(imagem_filtrada, cmap='gray')
plt.axis('off')
plt.title('Filtrada (Rejeita-Banda)')
plt.show()