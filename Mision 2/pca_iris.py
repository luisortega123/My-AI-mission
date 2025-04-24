import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 

datos_completos = load_iris()
X = datos_completos.data

X_shape = X.shape
medias = np.mean(X, axis=0)
X_centrado = X - medias
print(np.mean(X_centrado, axis=0))

matriz_cov = np.cov(X_centrado.T)

# Calcular valores y vectores propios
valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

# Imprimir para verificar (opcional)


indices_ascendetes = np.argsort(valores_propios)
indices_ordenados = indices_ascendetes[::-1]
valores_propios_ordenados = valores_propios[indices_ordenados]
vectores_propios_ordenados = vectores_propios[:, indices_ordenados]

matriz_w = vectores_propios_ordenados[:, 0:2]

# Es un array NumPy de 150x2
X_pca = X_centrado @ matriz_w
print(f"\n{X_pca}")

plt.scatter(X_pca[:,0],X_pca[:,1])

plt.show()