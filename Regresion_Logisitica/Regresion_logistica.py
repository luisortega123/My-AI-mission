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
def calcular_coste(X, y, theta, lmbda_reg):
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
    coste_original = - (1 / m) * sum_total
    
    # Parte 2: Calculándolo directamente aquí la funcion de Regularizacion
    theta_para_regular = theta[1:]
    suma_cuadrados_theta = np.sum(theta_para_regular**2)
    termino_regularizacion = (lmbda_reg/(2*m)) * suma_cuadrados_theta

    coste_total = coste_original + termino_regularizacion

    return coste_total


y_ejemplo = np.array([[1],  # Etiqueta verdadera para la observación 1
                    [0]]) # Etiqueta verdadera para la observación 2

coste_ejemplo = calcular_coste(X_ejemplo, y_ejemplo, theta_ejemplo, lmbda_reg=0)
print(f"Calcular el coste:\n", coste_ejemplo)

# Funcion GD
def descenso_gradiente(X, y, theta, alpha, num_iteraciones, lmbda_reg):
    y = y.reshape(-1,1)
    historial_coste = []  # Lista para guardar el coste en cada iteración (opcional)
    m = X.shape[0]
    for i in range(num_iteraciones):
        
        # 1. Calcular Predicciones (h_theta) 
        predicciones = calcular_hipotesis(X, theta)

        # 2. Calcular Errores (predicciones - y)
        errores = predicciones - y

        # 3. Calcular el Gradiente Vectorizado: (1/m) * X^T * errores
        #    (Necesitas la transpuesta de X_bias: X_bias.T)
        gradiente_original = (1 / m) * X.T @ errores
        # Agregamos termino de penalizacion
        theta_penalizacion = theta.copy()
        theta_penalizacion[0] = 0
        penalizacion_gradiente = theta_penalizacion * (lmbda_reg/m)
        gradiente_regularizado = gradiente_original + penalizacion_gradiente

        # 4. Regla de Actualizar Theta: theta = theta - alpha * gradiente
        theta = theta - alpha * gradiente_regularizado

        # 5. Calcular y guardar el coste (opcional)
        coste_actual = calcular_coste(X, y, theta, lmbda_reg)
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
    theta_resultado, historial_coste = descenso_gradiente(X_bias_scaled, y,theta_inicial, alpha, num_iteraciones, lmbda_reg=0)
    resultados_historial[alpha] = historial_coste

for alpha, historial_coste in resultados_historial.items():
    plt.plot(historial_coste, label=f"alpha = {alpha}")


theta_final_logistica, cost_history_logistica = descenso_gradiente(X_bias_scaled, y, theta_inicial, alpha, num_iteraciones, lmbda_reg=0)

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


num_iteraciones_prueba = 100
alpha_prueba = 0.1

lmbda_lista = {}
lmbda_valores = [
    0,
    0.1,
    0.01,
    0.001,
    1,
    10,
    100,
    1000,
]


for lmbda_reg in lmbda_valores:
    print(f"--- Procesando lambda = {lmbda_reg} ---")
    theta_calculado,  historial_coste = descenso_gradiente(X_bias_scaled, y, theta_inicial.copy(), alpha_prueba, num_iteraciones_prueba, lmbda_reg)
    lmbda_lista[lmbda_reg] = (theta_calculado, historial_coste)
    
for lmbda_reg, resultados_tupla in lmbda_lista.items():
    theta_calculado = resultados_tupla[0]
    print(f"lambda({lmbda_reg}) = {resultados_tupla}")
    print(theta_calculado.T)

    plt.plot(resultados_tupla[1], label=f"Coste para lambda = {lmbda_reg}")

plt.xlabel("Iteraciones")
plt.ylabel("Coste J(θ)")
plt.title("Curvas de Coste por Lambda")
plt.legend()
plt.show()


valores_theta1 = []
valores_theta2 = []
valores_theta3 = []
valores_theta4 = []
valores_theta5 = []
valores_theta6 = []
valores_theta7 = []
valores_theta8 = []

for un_lambda_especifico in lmbda_valores:
    theta_calculado, historial_coste = lmbda_lista[un_lambda_especifico]
    valor_theta1 = theta_calculado[1,0]
    
    valores_theta1.append(theta_calculado[1, 0])
    valores_theta2.append(theta_calculado[2, 0])
    valores_theta3.append(theta_calculado[3, 0])
    valores_theta4.append(theta_calculado[4, 0])
    valores_theta5.append(theta_calculado[5, 0])
    valores_theta6.append(theta_calculado[6, 0])
    valores_theta7.append(theta_calculado[7, 0])
    valores_theta8.append(theta_calculado[8, 0])

plt.plot(lmbda_valores[1:], valores_theta1[1:], label=r'$\theta_1$')
plt.plot(lmbda_valores[1:], valores_theta2[1:], label=r'$\theta_2$')
plt.plot(lmbda_valores[1:], valores_theta3[1:], label=r'$\theta_3$')
plt.plot(lmbda_valores[1:], valores_theta4[1:], label=r'$\theta_4$')
plt.plot(lmbda_valores[1:], valores_theta5[1:], label=r'$\theta_5$')
plt.plot(lmbda_valores[1:], valores_theta6[1:], label=r'$\theta_6$')
plt.plot(lmbda_valores[1:], valores_theta7[1:], label=r'$\theta_7$')
plt.plot(lmbda_valores[1:], valores_theta8[1:], label=r'$\theta_8$')

plt.xlabel(r'$\lambda$', fontsize=12)
plt.ylabel(r'Valor de los coeficientes $\theta_j$', fontsize=12)
plt.title(r'Coeficientes $\theta_1$ a $\theta_8$ en función de $\lambda$', fontsize=14)

plt.legend()      # Muestra la leyenda con las etiquetas
plt.grid(True)    # Cuadrícula para mejor visualización
plt.tight_layout()
plt.xscale('log')
plt.show()
