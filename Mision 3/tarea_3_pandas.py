# -*- coding: utf-8 -*-
"""
Implementación de Naive Bayes Gaussiano desde cero para el dataset Iris
Incluye comparación con scikit-learn y explicaciones matemáticas detalladas
"""

# =============================================================================
# 1. Importación de librerías y configuración inicial
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# =============================================================================
# 2. Carga y exploración de datos
# =============================================================================
# Cargamos el dataset desde un archivo CSV
df = pd.read_csv("Iris.csv")

# Inspección inicial de la estructura de los datos
print("--- Primeras 5 filas del DataFrame ---")
print(df.head())

print("\n--- Últimas 5 filas del DataFrame ---")
print(df.tail())

# Análisis de tipos de datos y valores faltantes
print("\n--- Información general del DataFrame (Tipos de datos, Nulos) ---")
df.info()

# Estadísticos descriptivos básicos
print("\n--- Estadísticas Descriptivas (columnas numéricas) ---")
print(df.describe().round(3))

# Distribución de clases (probabilidad a priori)
print("\n--- Conteo de muestras por Especie (Forma 1) ---")
print(df.value_counts("Species"))

# =============================================================================
# 3. Manipulación de datos y selección de características
# =============================================================================
# Acceso a elementos usando posición (index-based)
print("\n--- Acceso por posición (iloc): Última fila, columnas 1 y 3 ---")
print(df.iloc[-1, [1, 3]])

# Acceso a elementos usando etiquetas (label-based)
print("\n--- Acceso por etiqueta (loc): Fila 149, columnas SepalWidthCm y PetalWidthCm ---")
print(df.loc[149, ["SepalWidthCm", "PetalWidthCm"]])

# Filtrado de datos usando condiciones booleanas
print("\n--- Filtro: Especie 'Iris-setosa' con SepalLengthCm < 5 ---")
print(df[(df["Species"] == "Iris-setosa") & (df["SepalLengthCm"] < 5)])

# =============================================================================
# 4. Carga de datos desde scikit-learn y preparación
# =============================================================================
# Cargamos el dataset directamente desde scikit-learn para validación
datos = load_iris()

# X: Matriz de características (150 muestras × 4 características)
# Y: Vector de etiquetas (0: setosa, 1: versicolor, 2: virginica)
X = datos.data
Y = datos.target

print(f"\n--- Datos cargados desde scikit-learn ---")
print(f"Forma de la matriz de características (X): {X.shape}")
print(f"Forma del vector de etiquetas (Y): {Y.shape}")
print(f"Clases únicas en Y: {np.unique(Y)}")

# =============================================================================
# 5. Cálculo de parámetros del modelo (Entrenamiento Naive Bayes)
# =============================================================================
"""
FUNDAMENTO MATEMÁTICO:
Naive Bayes asume que las características siguen una distribución normal por clase.
Parámetros a calcular para cada clase:
- μ (media): np.mean()
- σ (desviación estándar): np.std()
- Prior (probabilidad a priori): Conteo de clase / Total de muestras
"""

# Separamos los datos por clase
X_clase0 = X[Y == 0]  # Iris-setosa
X_clase1 = X[Y == 1]  # Iris-versicolor
X_clase2 = X[Y == 2]  # Iris-virginica

# Calculamos estadísticos para cada clase
media_clase0 = np.mean(X_clase0, axis=0)
std_clase0 = np.std(X_clase0, axis=0)
prior_clase0 = X_clase0.shape[0] / X.shape[0]

# Repetimos para las otras clases (en producción usaríamos un bucle)
media_clase1 = np.mean(X_clase1, axis=0)
std_clase1 = np.std(X_clase1, axis=0)
prior_clase1 = X_clase1.shape[0] / X.shape[0]

media_clase2 = np.mean(X_clase2, axis=0)
std_clase2 = np.std(X_clase2, axis=0)
prior_clase2 = X_clase2.shape[0] / X.shape[0]

# =============================================================================
# 6. Implementación de la función de densidad de probabilidad Gaussiana
# =============================================================================
"""
FUNCIÓN DE DENSIDAD DE PROBABILIDAD (PDF) GAUSSIANA:
            1                  (x - μ)^2
P(x) = ----------- * exp(- -------------)
        σ√(2π)               2σ^2

Donde:
- μ: media de la característica para la clase
- σ: desviación estándar de la característica para la clase
- x: valor de la característica a evaluar
"""
def gaussian_pdf(x, mu, std):
    epsilon = 1e-9  # Evitar división por cero
    varianza = std**2 + epsilon
    denominador = np.sqrt(2 * np.pi * varianza)
    exponente = -((x - mu) ** 2) / (2 * varianza)
    pdf_valor = (1 / denominador) * np.exp(exponente)
    return pdf_valor

# =============================================================================
# 7. Función de predicción manual de Naive Bayes
# =============================================================================
"""
FUNDAMENTO MATEMÁTICO DE LA PREDICCIÓN:
Clase predicha = argmax [ log(P(clase)) + Σ log(P(x_i|clase)) ]

Usamos logaritmos para:
1. Evitar underflow numérico al multiplicar probabilidades pequeñas
2. Convertir multiplicaciones en sumas (computacionalmente más estable)
"""
def predecir_clases_nb(flor_nueva, lista_media, lista_stds, lista_priors):
    log_posteriors = []
    for clase_idx in range(len(lista_media)):
        media_actual = lista_media[clase_idx]
        std_actual = lista_stds[clase_idx]
        prior_actual = lista_priors[clase_idx]

        
        # Calcular log(prior)
        epsilon_prior = 1e-9  # Suavizado para evitar log(0)
        log_prior_actual = np.log(prior_actual + epsilon_prior)
        
        # Calcular log-likelihood sumando todas las características
        probabilidades_pdf = gaussian_pdf(flor_nueva, media_actual, std_actual)
        log_probabilidades_pdf = np.log(probabilidades_pdf + epsilon_prior)
        log_likelihood = np.sum(log_probabilidades_pdf)
        
        # Log posterior = log(prior) + log(likelihood)
        log_posterior_actual = log_likelihood + log_prior_actual
        log_posteriors.append(log_posterior_actual)
    
    # Seleccionar la clase con mayor probabilidad posterior
    return np.argmax(log_posteriors)

# Definir las listas de medias, desviaciones estándar y priors
lista_medias = [media_clase0, media_clase1, media_clase2]
lista_stds = [std_clase0, std_clase1, std_clase2]
lista_priors = [prior_clase0, prior_clase1, prior_clase2]

# =============================================================================
# 8. Evaluación del modelo implementado manualmente
# =============================================================================
predicciones = []
for i in range(X.shape[0]):
    flor_actual = X[i]
    prediccion = predecir_clases_nb(flor_actual, lista_medias, lista_stds, lista_priors)
    predicciones.append(prediccion)

predicciones_np = np.array(predicciones)

# Cálculo de precisión
aciertos = np.sum(predicciones_np == Y)
precision_propia = aciertos / X.shape[0]

# =============================================================================
# 9. Validación con scikit-learn
# =============================================================================
modelo_sk = GaussianNB()
modelo_sk.fit(X, Y)
predicciones_sk = modelo_sk.predict(X)

# Comparación de resultados
print(f"\n--- Comparación de implementaciones ---")
print(f"Precisión Manual: {precision_propia * 100:.2f}%")
print(f"Precisión Scikit-learn: {(np.sum(predicciones_sk == Y)/X.shape[0])*100:.2f}%")

"""
NOTA FINAL:
La pequeña diferencia en precisión se debe a:
1. Implementación de suavizado (epsilon) diferente
2. Estrategias de manejo de precisión numérica
3. Posibles diferencias en cálculo de la varianza
"""