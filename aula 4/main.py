import cv2
import numpy as np
from matplotlib import pyplot as plt

#1

def plot_fourier_spectrum(image):
    # Aplica a transformada de Fourier na imagem
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Plota a imagem original
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Imagem de entrada'), plt.xticks([]), plt.yticks([])

    # Plota o espectro de Fourier
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Espectro de Fourier'), plt.xticks([]), plt.yticks([])

    plt.show()

#2

# Carrega a imagem
image_path = 'caminho/para/a/imagem.jpg'
image = cv2.imread(image_path, 0)  # Carrega a imagem em escala de cinza

# Aplica a transformada de Fourier e plota o espectro
plot_fourier_spectrum(image)

def apply_filter(image, filter_type, cutoff_freq, width=10):
    # Realiza a transformada de Fourier na imagem
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Aplica o filtro correspondente
    if filter_type == 'passa_baixa':
        fshift[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] = 0
    elif filter_type == 'passa_alta':
        fshift[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] = fshift[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] * width
    elif filter_type == 'passa_banda':
        fshift[:crow - cutoff_freq, :] = 0
        fshift[crow + cutoff_freq:, :] = 0
        fshift[:, :ccol - cutoff_freq] = 0
        fshift[:, ccol + cutoff_freq:] = 0
    elif filter_type == 'rejeita_banda':
        fshift[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] = 0

    # Realiza a transformada inversa de Fourier
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

#3

# Carrega a imagem
image_path = 'caminho/para/a/imagem.jpg'
image = cv2.imread(image_path, 0)  # Carrega a imagem em escala de cinza

# Parâmetros do filtro
cutoff_freq = 30  # Freq. de corte para filtros passa-baixa, passa-alta, rejeita-banda
width = 10  # Largura para filtro passa-alta

# Aplica e salva os filtros
filtered_image = apply_filter(image, 'passa_baixa', cutoff_freq)
cv2.imwrite('passa_baixa.jpg', filtered_image)

filtered_image = apply_filter(image, 'passa_alta', cutoff_freq, width)
cv2.imwrite('passa_alta.jpg', filtered_image)

filtered_image = apply_filter(image, 'passa_banda', cutoff_freq)
cv2.imwrite('passa_banda.jpg', filtered_image)

filtered_image = apply_filter(image, 'rejeita_banda', cutoff_freq)
cv2.imwrite('rejeita_banda.jpg', filtered_image)

# Carrega a imagem
image_path = 'caminho/para/a/imagem.jpg'
image = cv2.imread(image_path)

# Aplica o filtro gaussiano
filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

# Salva a imagem do filtro gaussiano
cv2.imwrite('filtro_gaussiano.jpg', filtered_image)

# Salva a imagem filtrada
cv2.imwrite('imagem_filtrada.jpg', image)

#4

# Carrega a imagem original e a imagem do filtro
imagem_path = 'arara.png'
filtro_path = 'arara_filtro.png'

imagem = cv2.imread(imagem_path)
filtro = cv2.imread(filtro_path, 0)  # Carrega o filtro em escala de cinza

# Aplica o filtro rejeita-banda na imagem
resultado = cv2.filter2D(imagem, -1, filtro)

# Salva o resultado
cv2.imwrite('resultado_arara.png', resultado)