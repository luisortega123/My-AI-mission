from sklearn.datasets import fetch_california_housing
import numpy as np  

# Importamos los DataFrame
datos_california = fetch_california_housing()
X = datos_california.data   # Las características (features)
Y = datos_california.target # # El objetivo (target), es decir, lo que quieres predecir
# Ajustar la Forma de Y
Y = np.reshape(Y, (-1,1)) # el np.reshape cambia la forma (shape) de un array sin cambiar sus datos.

# Calcular medias y desviaciones estándar DESDE X ORIGINAL
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0) 

# Calcular X_scaled USANDO X, mu y sigma
#    (¡Cuidado con la división por cero si sigma es 0! Podemos añadir un valor pequeño)
epsilon = 1e-8 # Un valor pequeño para evitar división por cero
X_scaled = (X - mu) / (sigma + epsilon)

# Añadir la Columna de Unos (Bias) a X
unos = np.ones((X_scaled.shape[0], 1))

# Crear X_bias usando la columna de unos y X_scaled
X_bias = np.hstack((unos,X_scaled)) # np.hstack apila los arrays


# Implementar la Función de Hipótesis
def calcular_hipotesis(X_bias, theta):
        
        h_theta = X_bias @ theta
        return h_theta  
# ¡La lógica central de la hipótesis está lista! 


# Siguiente Paso: Implementar la Función de Coste (MSE)
def calcular_coste(X_bias, y, theta):
    m = X_bias.shape[0]
    predicciones = calcular_hipotesis(X_bias, theta) # Llama a tu función de hipótesis
    errores = predicciones - y # Calcula la diferencia predicciones - y
    errores_cuadraticos = errores ** 2 # Eleva los errores al cuadrado
    suma_errores_cuadraticos = np.sum(errores_cuadraticos) # Suma todos los errores cuadráticos
    coste =  1 / (2 * m) * suma_errores_cuadraticos # Aplica el factor 1 / (2 * m)
    return coste


# Implementar el Descenso de Gradiente
def descenso_gradiente(X_bias, y, theta_inicial, alpha, num_iteraciones):
    m = X_bias.shape[0] # Número de muestras
    theta = theta_inicial.copy() # Empezamos con theta_inicial (copiamos para no modificar el original)
    historial_coste = [] # Lista para guardar el coste en cada iteración (opcional)

    for i in range(num_iteraciones):
        # 1. Calcular Predicciones (h_theta) - ¿Cómo llamas a tu función?
        predicciones = calcular_hipotesis(X_bias, theta)

        # 2. Calcular Errores (predicciones - y)
        errores  = predicciones - y

        # 3. Calcular el Gradiente Vectorizado: (1/m) * X^T * errores
        #    (Necesitas la transpuesta de X_bias: X_bias.T)
        gradiente = (1/m) * X_bias.T @ errores

        # 4. Regla de Actualizar Theta: theta = theta - alpha * gradiente
        theta = theta - alpha * gradiente

        # 5. Calcular y guardar el coste (opcional)
        coste_actual = calcular_coste(X_bias, y, theta)
        historial_coste.append(coste_actual)

    # Devolver el theta final y el historial de coste
    return theta, historial_coste


# Entrenar al modelo = encontrar al theta optimo!!
# --- Entrenamiento del Modelo ---

# 1. Inicializar theta (vector de ceros con forma (n+1, 1))
n_caracteristicas = X.shape[1] # Número de características originales
theta_inicial = np.zeros((n_caracteristicas + 1, 1)) # Usa np.zeros con la forma correcta

# 2. Definir hiperparámetros
alpha = 0.01 # Elige una tasa de aprendizaje
num_iteraciones = 1000 # Elige un número de iteraciones

# 3. Ejecutar descenso de gradiente
theta_final, historial_coste = descenso_gradiente(X_bias, Y, theta_inicial, alpha, num_iteraciones) # Pasa los argumentos correctos

# (Opcional) Imprimir theta_final para ver los parámetros aprendidos
print("Theta final:", theta_final)
print(f"Último valor del historial de coste: {historial_coste}")


# (Aquí sería útil GRAFICAR el historial_coste para ver la convergencia)
import matplotlib.pyplot as plt
plt.plot(historial_coste)
plt.xlabel("Iteraciones")
plt.ylabel("Coste J(θ)")
plt.title("Convergencia del Descenso de Gradiente (Datos Escalados)")
plt.show()



# Función de Predicción
def predecir(X_nuevos, theta_final, mu, sigma):
    epsilon = 1e-8
    X_nuevos_escalados = (X_nuevos - mu) / (sigma + epsilon)
    unos = np.ones((X_nuevos_escalados.shape[0],1))
    X_nuevos_Bias = np.hstack((unos, X_nuevos_escalados))
    predicciones = X_nuevos_Bias @ theta_final
    return predicciones