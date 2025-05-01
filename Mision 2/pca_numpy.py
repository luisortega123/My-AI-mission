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
# Matriz original
X = datos_completos.data  # Obtiene las características (150 muestras x 4 variables)
# Definimos un valor a K
K = 2
# Paso 1: Centrar datos restando la media de cada columna (variable)
medias = np.mean(X, axis=0)
X_centrado = X - medias

# Verifica que las medias de las columnas centradas estén (casi) en cero
resultado_media = np.mean(X_centrado, axis=0)
assert np.allclose(resultado_media, 0), "Las medias no están cerca de cero"
print("  Verificación de centrado: OK")

# Paso 2: Calcular la matriz de covarianza (muestra cómo varían juntas las variables)
matriz_cov = np.cov(X_centrado.T)

# Paso 3: Obtener valores y vectores propios (eigen decomposicion de la matriz de covarianza)
# - Valores propios: varianza explicada por cada componente
# - Vectores propios: direcciones (componentes principales)
valores_propios, vectores_propios = np.linalg.eig(matriz_cov)

# Paso 4: Ordenar componentes principales por importancia (mayor varianza explicada)
indices_ordenados = np.argsort(valores_propios)[::-1]
vectores_propios_ordenados = vectores_propios[:, indices_ordenados]
# Verificar no negatividad de valores propios
assert np.all(valores_propios >= -1e-9), "Error: Se encontraron valores propios negativos"
print("  Verificación de No negatividad: OK")
#  Verificar usando NumPy que cada elemento es mayor o igual que el siguiente elemento en la lista
valores_propios_ordenados = valores_propios[indices_ordenados]
assert np.all(valores_propios_ordenados[: -1] >= valores_propios_ordenados[1:]), "Error: El elemento es mayor o igual que el siguiente elemento"
print("  Verificación de es mayor o igual: OK")
# Suma de Eigenvalues vs. Traza
suma_valores_propios = np.sum(valores_propios)
traza_matriz_cov = np.trace(matriz_cov)
assert np.isclose(suma_valores_propios, traza_matriz_cov), "Error: No son iguales"
print("  Verificación de suma vs traza: OK")
# Paso 5: Seleccionar los dos primeros componentes principales (2D para graficar)
matriz_w = vectores_propios_ordenados[:, 0:K]
# Asegurarnos que tengan norma unitaria (es decir, que su longitud o magnitud sea igual a 1).
assert np.isclose(np.linalg.norm(matriz_w[:, 0]), 1.0), "Error: No cumplen con la norma unitaria" 
assert np.isclose(np.linalg.norm(matriz_w[:, 1]), 1.0), "Error: El segundo componente principal no tiene norma unitaria"
# Calcule el producto escalar entre el primer y segundo vector componente 
producto_escalar = matriz_w[:, 0] @ matriz_w[:, 1]
casi_cero = np.isclose(producto_escalar, 0.0)
assert casi_cero, "Error: Los primeros dos componentes principales no son ortogonales (producto escalar no es cero)"
# tambien puedes hacer esto(manera mas simplificada): 
assert np.isclose(matriz_w[:, 0] @ matriz_w[:, 1], 0.0), "Error: Los primeros dos componentes principales no son ortogonales"
print("Verificación de ortogonalidad PC1 vs PC2: OK")
# Paso 6: Proyectar los datos centrados sobre los componentes seleccionados
# Resultado: nueva representación en espacio reducido (150x2)
X_pca = X_centrado @ matriz_w

# Verificar las dimensiones de la matriz proyectada X_pca
assert X_pca.shape == (X.shape[0], K), f"Error: La forma de X_pca no es la esperada. Esperada: {(X.shape[0], K)}, Obtenida: {X_pca.shape}"
print(f"Verificación de forma X_pca ({K} componentes): OK") # Mensaje OK solo si el assert no falló

# --- Alternativa usando SVD (Descomposición en Valores Singulares) ---

# Paso extra: Usar SVD en lugar de matriz de covarianza para obtener componentes principales
U, s, Vh = np.linalg.svd(X_centrado, full_matrices=False)

# Los vectores fila de Vh son los componentes principales (similares a vectores propios)
W_svd = Vh.T
matriz_w_final = W_svd[:, :K]  # Selecciona los primeros 2 componentes
# Asegurarnos que tengan norma unitaria en SVD (es decir, que su longitud o magnitud sea igual a 1).
assert np.isclose(np.linalg.norm(matriz_w_final[:, 0]), 1.0), "Error: No cumplen con la norma unitaria usando la alternativa SVD" 
assert np.isclose(np.linalg.norm(matriz_w_final[:, 1]), 1.0), "Error: El segundo componente principal en SVD no tiene norma unitaria"
# Calcule el producto escalar entre el primer y segundo vector componente alternativa (SVD)
assert np.isclose(matriz_w_final[:, 0] @ matriz_w_final[:, 1], 0.0), "Error: Los primeros dos componentes principales no son ortogonales"
print("Verificación de ortogonalidad usando (SVD) PC1 vs PC2: OK")
# Proyección de datos usando SVD
X_pca_final = X_centrado @ matriz_w_final

# Verificar las dimensiones de la matriz proyectada X_pca_final (SVD)
assert X_pca_final.shape == (X.shape[0], K), f"Error: La forma de X_pca no es la esperada. Esperada: {(X.shape[0], K)}, Obtenida: {X_pca_final.shape}"
print(f"Verificación de forma X_pca_final ({K} componentes): OK") # Mensaje OK solo si el assert no falló


# Paso final: Visualización de los datos proyectados (en 2D)
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title('PCA del Dataset Iris')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
#plt.show()


# Paso final alternativo: Visualización de los datos proyectados (en 2D)
plt.scatter(X_pca_final[:,0], X_pca_final[:,1])
plt.title('PCA del Dataset Iris Alternativo')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show() 
