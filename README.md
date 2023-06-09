# Redu-o-de-Dimensionalidade-em-Imagens-para-Redes-Neurais-
import cv2
import numpy as np

# Carregar a imagem colorida
img = cv2.imread('ponte.jpg')

# Converter a imagem para tons de cinza
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar o filtro Gaussiano para suavizar a imagem
img_suavizada = cv2.GaussianBlur(img_cinza, (7, 7), 0)

# Binarizar a imagem usando um limiar de 160
limiar = 160
_, img_bin = cv2.threshold(img_suavizada, limiar, 255, cv2.THRESH_BINARY)

# Inverter a imagem binarizada
_, img_bin_inv = cv2.threshold(img_suavizada, limiar, 255, cv2.THRESH_BINARY_INV)

# Criar uma imagem final combinando as imagens suavizada, binarizada e a máscara
resultado = np.vstack([
    np.hstack([img_suavizada, img_bin]),
    np.hstack([img_bin_inv, cv2.bitwise_and(img_cinza, img_cinza, mask=img_bin)])
])

# Exibir a imagem binarizada
cv2.imshow("Binarização da imagem", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()

