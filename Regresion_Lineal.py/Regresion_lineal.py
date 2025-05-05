from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

# Importamos los DataFrame
datos_california = fetch_california_housing()
X = datos_california.data  # Las características (features)
y = datos_california.target  # # El objetivo (target), es decir, lo que quieres predecir
# Ajustar la Forma de Y
y = np.reshape(
    y, (-1, 1)
)  # el np.reshape cambia la forma (shape) de un array sin cambiar sus datos.

# Calcular medias y desviaciones estándar DESDE X ORIGINAL
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

# Calcular X_scaled USANDO X, mu y sigma
#    (¡Cuidado con la división por cero si sigma es 0! Podemos añadir un valor pequeño)
epsilon = 1e-8  # Un valor pequeño para evitar división por cero
X_scaled = (X - mu) / (sigma + epsilon)

# Añadir la Columna de Unos (Bias) a X
unos = np.ones((X_scaled.shape[0], 1))

# Crear X_bias usando la columna de unos y X_scaled
X_bias_scaled = np.hstack((unos, X_scaled))  # np.hstack apila los arrays


# Implementar la Función de Hipótesis
def calcular_hipotesis(X_bias, theta):

    h_theta = X_bias @ theta
    return h_theta


# ¡La lógica central de la hipótesis está lista!


# Siguiente Paso: Implementar la Función de Coste (MSE)
def calcular_coste(X_bias, y, theta):
    m = X_bias.shape[0]
    predicciones = calcular_hipotesis(X_bias, theta)  # Llama a tu función de hipótesis
    errores = predicciones - y  # Calcula la diferencia predicciones - y
    errores_cuadraticos = errores**2  # Eleva los errores al cuadrado
    suma_errores_cuadraticos = np.sum(
        errores_cuadraticos
    )  # Suma todos los errores cuadráticos
    coste = 1 / (2 * m) * suma_errores_cuadraticos  # Aplica el factor 1 / (2 * m)
    return coste


# Implementar el Descenso de Gradiente
def descenso_gradiente(X_bias, y, theta_inicial, alpha, num_iteraciones):
    m = X_bias.shape[0]  # Número de muestras
    theta = (
        theta_inicial.copy()
    )  # Empezamos con theta_inicial (copiamos para no modificar el original)
    historial_coste = []  # Lista para guardar el coste en cada iteración (opcional)

    for i in range(num_iteraciones):
        # 1. Calcular Predicciones (h_theta) - ¿Cómo llamas a tu función?
        predicciones = calcular_hipotesis(X_bias, theta)

        # 2. Calcular Errores (predicciones - y)
        errores = predicciones - y

        # 3. Calcular el Gradiente Vectorizado: (1/m) * X^T * errores
        #    (Necesitas la transpuesta de X_bias: X_bias.T)
        gradiente = (1 / m) * X_bias.T @ errores

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
n_caracteristicas = X.shape[1]  # Número de características originales
theta_inicial = np.zeros(
    (n_caracteristicas + 1, 1)
)  # Usa np.zeros con la forma correcta

# 2. Definir hiperparámetros
alpha = 0.01  # Elige una tasa de aprendizaje
num_iteraciones = 4000  # Elige un número de iteraciones


resultados_historial = {}
alphas_prueba = [
    0.3,
    0.1,
    0.03,
    0.01,
    0.001,
]

for alpha in alphas_prueba:
    theta_resultado, historial_coste = descenso_gradiente(
        X_bias_scaled, y, theta_inicial, alpha, num_iteraciones
    )
    resultados_historial[alpha] = historial_coste

for alpha, historial_coste in resultados_historial.items():
    plt.plot(historial_coste, label=f"Alpha = {alpha}")

plt.legend()
plt.xlabel("Iteraciones")
plt.ylabel("Coste J(θ)")
plt.title("Comparación de la Convergencia según la Tasa de Aprendizaje")
plt.show()

alpha_optimo = 0.1
unos_unscaled = np.ones((X.shape[0], 1))
X_bias_unscaled = np.hstack((unos_unscaled, X))

# Definir hiperparámetros (asegúrate de que sean los mismos para ambas)
num_iteraciones = 4000  # O el valor que uses
alpha_optimo = 0.1
n_features_original = X.shape[1]
theta_inicial_base = np.zeros((n_features_original + 1, 1))

# Ejecución 1: CON Escalado
print("Ejecutando GD con datos ESCALADOS...")
_, historial_coste_scaled = descenso_gradiente(
    X_bias_scaled, y, theta_inicial_base.copy(), alpha_optimo, num_iteraciones
)
print("Terminado.")

# Ejecución 2: SIN Escalado
print("Ejecutando GD con datos SIN ESCALAR...")
_, historial_coste_unscaled = descenso_gradiente(
    X_bias_unscaled, y, theta_inicial_base.copy(), alpha_optimo, num_iteraciones
)
print("Terminado.")

# Concepto del Ploteo:
plt.plot(historial_coste_scaled, label="Con Escalado")
plt.plot(historial_coste_unscaled, label="Sin Escalado ")  # Esta podría verse muy rara!
plt.xlabel("Iteraciones")
plt.ylabel("Coste J(θ)")
plt.title("Convergencia con vs. Sin Escalado de Características")
plt.legend()

# ---> ¡CONSEJO IMPORTANTE! <---
# Es MUY probable que los valores en historial_coste_unscaled sean GIGANTES.
# Esto hará que la curva de historial_coste_scaled parezca una línea plana en cero.
# Para poder ver bien la curva BUENA (la escalada), quizás necesites
# limitar el eje Y. Puedes probar añadiendo:
# plt.ylim(top=np.min(historial_coste_scaled)*10) # Ajusta el 'top' según veas
# O un valor fijo si conoces el rango del coste escalado: plt.ylim(0, valor_razonable)
#
plt.grid(True)
plt.show()

# 3. Ejecutar descenso de gradiente
theta_final, historial_coste = descenso_gradiente(
    X_bias_scaled, y, theta_inicial, alpha, num_iteraciones
)  # Pasa los argumentos correctos

# (Opcional) Imprimir theta_final para ver los parámetros aprendidos
print("Theta final:", theta_final)
print(f"Último valor del historial de coste: {historial_coste}")


# (Aquí sería útil GRAFICAR el historial_coste para ver la convergencia)

plt.plot(historial_coste)
plt.xlabel("Iteraciones")
plt.ylabel("Coste J(θ)")
plt.title("Convergencia del Descenso de Gradiente (Datos Escalados)")
plt.show()


# Función de Predicción
def predecir(X_nuevos, theta_final, mu, sigma):
    epsilon = 1e-8
    X_nuevos_escalados = (X_nuevos - mu) / (sigma + epsilon)
    unos = np.ones((X_nuevos_escalados.shape[0], 1))
    X_nuevos_Bias = np.hstack((unos, X_nuevos_escalados))
    predicciones = X_nuevos_Bias @ theta_final
    return predicciones


# FUNCION NORMAL = β = (Xᵀ . X)⁻¹. Xᵀ . y
# Matriz X (2 observaciones, 1 característica + columna de unos)
# Fila 1: Observación 1 (intercepto=1, característica 1=2.0)
# Fila 2: Observación 2 (intercepto=1, característica 1=3.0)
X_ejemplo = np.array([[1.0, 2.0],
                      [1.0, 3.0]])

# Vector y (valores reales para las 2 observaciones)
# Fila 1: Valor real para Observación 1 = 5.0
# Fila 2: Valor real para Observación 2 = 7.0
y_ejemplo = np.array([[5.0],
                      [7.0]])

# Pasos para Fórmula normal: β = (Xᵀ . X)⁻¹. Xᵀ . y
# Traspuesta de X
Xt_ejemplo = X_ejemplo.T
print(Xt_ejemplo)
# Multiplicar matricez
XtX_ejemplo = Xt_ejemplo @ X_ejemplo
print(f"Multiplicacion de matricez: \n{XtX_ejemplo}")
# Calculamos la inversa. 
inv_XtX_ejemplo = np.linalg.inv(XtX_ejemplo)
print(f"La matriz inversa es: \n{inv_XtX_ejemplo}")
# Hasta aqui tenemos : # (Xᵀ X)⁻¹  
# Calcular Xᵀ y → transpuesta de X multiplicada por y
Xty_ejemplo = Xt_ejemplo @ y_ejemplo
print(f"Calculamos Xᵀ.y, transpuesta de X multiplicada por y: \n{Xty_ejemplo}")
# Calcular θ = (Xᵀ X)⁻¹ (Xᵀ y)
theta_ejemplo = inv_XtX_ejemplo @ Xty_ejemplo
print(f"Nuestro tetha ejemplo es : \n{theta_ejemplo}")

def calcular_ecuacion_normal(X,y):
    Xt = X.T
    XtX = Xt @ X
    inv_XtX = np.linalg.inv(XtX)
    Xty = Xt @ y
    theta = inv_XtX @ Xty

    return theta

# Cargamos datos
datos_california = fetch_california_housing()
X = datos_california.data  # Las características (features)
y = datos_california.target  # El objetivo (target)
# X_Bias sin escalar


# Calculamos theta usando la Ecuación Normal con la X 
theta_calculado_normal = calcular_ecuacion_normal(X_bias_unscaled, y) 

print(f"Theta calculado por Ecuación Normal:\n{theta_calculado_normal}")

# X_Bias sin escalar
X_bias_unscaled = np.hstack((unos_unscaled, X)) # Correcto, usa X original
# Calculamos theta usando la Ecuación Normal con la X SIN ESCALAR
theta_calculado_normal = calcular_ecuacion_normal(X_bias_unscaled, y)
print(f"Theta calculado por Ecuación Normal (sin escalar):\n{theta_calculado_normal}")
print("Theta GD:", theta_final.T) 
print("Theta EN:", theta_calculado_normal.T) # Este es de EN con datos SIN ESCALAR

# Comparas GD escalado con EN sin escalar
diferencia = np.linalg.norm(theta_final - theta_calculado_normal) # Usas theta_calculado_normal (o _unscaled)
print("Diferencia (GD escalado vs EN sin escalar):", diferencia)

# X_Bias con escalar
# Preparamos X con el término de sesgo (columna de unos al principio)
# Calculamos theta usando la Ecuación Normal con la X 
theta_calculado_normal_scaled = calcular_ecuacion_normal(X_bias_scaled, y) 

print("\nTheta GD escalado:", theta_final.T) # .T para verlo como fila si es largo
print("Theta EN escalado:", theta_calculado_normal_scaled.T)
diferencia = np.linalg.norm(theta_final - theta_calculado_normal_scaled)
# Comparación: GD con escalado vs. EN sin escalado
diferencia_correcta = np.linalg.norm(theta_final - theta_calculado_normal_scaled)
print("Diferencia (GD escalado vs EN ESCALADO):", diferencia_correcta)



