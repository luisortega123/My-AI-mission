import numpy as np

def sumar_matrices_np(matriz_a, matriz_b):
    # Verifica que ambas matrices tengan contenido
    if not matriz_a or not matriz_a[0]:
        raise ValueError("Error: La matriz A está vacía o mal formada.")
    if not matriz_b or not matriz_b[0]:
        raise ValueError("Error: La matriz B está vacía o mal formada.")
    
    try:
        # Convierte las listas a arrays de NumPy
        array_a = np.array(matriz_a)
        array_b = np.array(matriz_b)
        
        # Realiza la suma elemento por elemento
        suma_array = array_a + array_b

    except ValueError as e:
        raise ValueError(f"Error al sumar matrices: {e}") from e
            
    return suma_array


def multiplicar_matriz_por_escalar_np(matriz_lista, escalar):
    # Valida la matriz y el tipo del escalar
    if not matriz_lista or not matriz_lista[0]:
        raise ValueError("Error: La matriz está vacía o mal formada.")
    if not isinstance(escalar, (int, float)):
        raise TypeError("Error: El escalar debe ser un número.")

    try:
        # Convierte y multiplica por el escalar
        matriz_np = np.array(matriz_lista)
        resultado_np = matriz_np * escalar

    except ValueError as e:
        raise ValueError(f"Error en multiplicación: {e}") from e
        
    return resultado_np


def matriz_traspuesta_np(matriz_lista):
    # Verifica que la matriz sea válida
    if not matriz_lista or not matriz_lista[0]:
        raise ValueError("Error: La matriz está vacía o mal formada.")
    
    try:
        # Convierte y transpone la matriz
        matriz_np = np.array(matriz_lista)
        matriz_transpuesta_np = matriz_np.T

    except ValueError as e:
        raise ValueError(f"Error en transpuesta: {e}") from e

    return matriz_transpuesta_np


def multiplicar_matrices_np(matriz_a_lista, matriz_b_lista):
    # Valida ambas matrices
    if not matriz_a_lista or not matriz_a_lista[0]:
        raise ValueError("Error: La matriz A está vacía.")
    if not matriz_b_lista or not matriz_b_lista[0]:
        raise ValueError("Error: La matriz B está vacía.")
    
    try:
        # Conversión a arrays y multiplicación matricial
        array_a = np.array(matriz_a_lista)
        array_b = np.array(matriz_b_lista)
        resultado_np = array_a @ array_b

    except ValueError as e:
        raise ValueError(f"Error en multiplicación matricial: {e}") from e    
    
    return resultado_np

''' PCA '''

# Análisis PCA del dataset Iris
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 

# Carga el dataset completo
datos_completos = load_iris()
X = datos_completos.data  # Obtiene las características

# Centrado de datos: resta la media de cada característica
medias = np.mean(X, axis=0)
X_centrado = X - medias

# Cálculo de matriz de covarianza
matriz_cov = np.cov(X_centrado.T)

# Calcula valores y vectores propios
valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

# Ordena componentes principales por varianza descendente
indices_ordenados = np.argsort(valores_propios)[::-1]
vectores_propios_ordenados = vectores_propios[:, indices_ordenados]

# Selecciona los dos primeros componentes
matriz_w = vectores_propios_ordenados[:, 0:2]

# Proyección de datos al espacio reducido (150x2)
X_pca = X_centrado @ matriz_w

# Visualización de los datos proyectados
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()