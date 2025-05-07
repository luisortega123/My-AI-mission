import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Funcion sigmoide nos dara un valor entre 0 y 1, este es util para probabilidades
def sigmoid(z):
    exp = np.exp(-z)
    resultado_sigmoide = 1 / (1 + exp)
    return resultado_sigmoide


# Pruebas de valor para sigmoide
# Diccionario para almacenar los resultados
sigmoid_list = {}
# Lista para valores z
z_values = [
    0,
    10,
    100,
    0.1,
    0.5,
    -100,
    -10,
    1000,
    -500,
]
#
for z in z_values:
    result = sigmoid(z)
    sigmoid_list[z] = result

# Imprimir resutaldos
for z, g_z in sigmoid_list.items():
    print(f"sigmoid({z}) = {g_z}")


# función de hipótesis hθ(x)
# Multiplicamos las matrices
def calcular_hipotesis(X, theta):
    Z_vector = X @ theta
    Z_vector_prob = sigmoid(Z_vector)
    return Z_vector_prob


# Ejemplos
# X_ejemplo: 2 observaciones, 1 característica + columna de unos
X_ejemplo = np.array([[1.0, 5.0], [1.0, -2.0]])  # Observación 1  # Observación 2

# theta_ejemplo: para intercepto y 1 característica
theta_ejemplo = np.array([[0.5], [-0.1]])  # theta_0  # theta_1


probabilidades_ejemplo = calcular_hipotesis(X_ejemplo, theta_ejemplo)
print(f"Probabilidades ejemplo:\n", probabilidades_ejemplo)

# Función de coste (Entropía cruzada binaria):
# J(θ) = -(1/m) * ∑[ y(i) * log(hθ(x(i))) + (1 - y(i)) * log(1 - hθ(x(i))) ]
def calcular_coste(X, y, theta):
    epsilon = 1e-8  # Un valor muy pequeño para estabilidad numérica
    m = X.shape[0]  # Número de observaciones

    # 1. Obtener las predicciones (probabilidades)
    h_theta_X = calcular_hipotesis(X, theta)

    # 2. Recortar las predicciones para evitar log(0) o log(1) exactos
    h_theta_X_clipped = np.clip(h_theta_X, epsilon, 1 - epsilon)

    # 3. Calcular los dos términos de la función de coste de entropía cruzada
    # Termino cuando y=1: y * log(h)
    product1 = y * np.log(h_theta_X_clipped)
    # Termino cuando y=0: (1-y) * log(1-h)
    product2 = (1 - y) * np.log(1 - h_theta_X_clipped)

    # 4. Sumar los términos para todas las observaciones
    sum_products = product1 + product2
    sum_total = np.sum(sum_products)

    # 5. Aplicar el factor -1/m
    coste = - (1 / m) * sum_total
    
    return coste


y_ejemplo = np.array([[1],  # Etiqueta verdadera para la observación 1
                    [0]]) # Etiqueta verdadera para la observación 2

coste_ejemplo = calcular_coste(X_ejemplo, y_ejemplo, theta_ejemplo)
print(f"Calcular el coste:\n", coste_ejemplo)

# Funcion GD
def descenso_gradiente(X, y, theta, alpha, num_iteraciones):
    historial_coste = []  # Lista para guardar el coste en cada iteración (opcional)
    m = X.shape[0]
    for i in range(num_iteraciones):
        
        # 1. Calcular Predicciones (h_theta) 
        predicciones = calcular_hipotesis(X, theta)

        # 2. Calcular Errores (predicciones - y)
        errores = predicciones - y

        # 3. Calcular el Gradiente Vectorizado: (1/m) * X^T * errores
        #    (Necesitas la transpuesta de X_bias: X_bias.T)
        gradiente = (1 / m) * X.T @ errores

        # 4. Regla de Actualizar Theta: theta = theta - alpha * gradiente
        theta = theta - alpha * gradiente

        # 5. Calcular y guardar el coste (opcional)
        coste_actual = calcular_coste(X, y, theta)
        historial_coste.append(coste_actual)

    # Devolver el theta final y el historial de coste
    return theta, historial_coste
    

# EMPIEZA EL ENTRENAMIENTO
datos = load_breast_cancer()
X_procesada = datos.data
y = datos.target
y = y.reshape(-1, 1)

# Escalado de caracteristicas No olvidar!
epsilon = 1e-8
mu = np.mean(X_procesada, axis=0)
sigma = np.std(X_procesada, axis=0)
X_scaled = (X_procesada - mu) / (sigma + epsilon)

# Agregar una columna de unos
unos = np.ones((X_scaled.shape[0], 1))

# Combinarlos para crear tu matriz final que le pasarás al Descenso de Gradiente
X_bias_scaled = np.hstack((unos, X_scaled))
num_parametros = X_bias_scaled.shape[1] 
theta_inicial = np.zeros((num_parametros, 1))


alpha = 0.1 
num_iteraciones = 5000

# Pruebas con varios alpha para poder determinar cual seria el mejor 
resultados_historial = {}
alphas_prueba = [
    0.3,
    0.1,
    0.03,
    0.01,
    0.001,
]

for alpha in alphas_prueba:
    theta_resultado, historial_coste = descenso_gradiente(X_bias_scaled, y,theta_inicial, alpha, num_iteraciones)
    resultados_historial[alpha] = historial_coste

for alpha, historial_coste in resultados_historial.items():
    plt.plot(historial_coste, label=f"alpha = {alpha}")


theta_final_logistica, cost_history_logistica = descenso_gradiente(X_bias_scaled, y, theta_inicial, alpha, num_iteraciones)

print("Theta final:", theta_final_logistica)
print("Historial de coste:", cost_history_logistica)

plt.plot(cost_history_logistica)
plt.title("Convergencia del Descenso de Gradiente para Regresión Logística")
plt.xlabel("Número de Iteraciones")
plt.ylabel("Coste J(θ)")
plt.show()


# Funcion predecir
def predecir(X, theta):
    # Calcula probabilidades usando la función de hipótesis ya existente
    Z_predecir_prob = calcular_hipotesis(X, theta) 
    # Aplica el umbral estándar: >= 0.5 predice clase 1, < 0.5 predice clase 0
    predicciones = (Z_predecir_prob >= 0.5).astype(int) 
    return predicciones

# Accueracy
y_predicciones = predecir(X_bias_scaled, theta_final_logistica)
aciertos = y == y_predicciones
total_aciertos = np.sum(aciertos)
print(f"Aciertos: {total_aciertos} de {len(y)}")


porcentaje = (total_aciertos/len(y)) * 100
print(f"Porcentaje de aciertos: {porcentaje:,.2f}%")

# Alternativa
# --- Calcular Accuracy con el MEJOR theta ---
print("Calculando Accuracy...")
y_predicciones = predecir(X_bias_scaled, theta_final_logistica)
accuracy = np.mean(y == y_predicciones) # Forma concisa
print(f"Accuracy del modelo: {accuracy * 100:.2f}%")