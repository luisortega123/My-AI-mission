# Diagnóstico y Control del Modelo: Overfitting y Regularización

## ¿Qué es el Overfitting (Sobreajuste)?

El **overfitting** ocurre cuando un modelo aprende *demasiado bien* los datos con los que fue entrenado. No solo aprende los **patrones generales**, sino también las **particularidades, errores o ruido** de esos datos. Como consecuencia, **pierde capacidad para generalizar** a nuevos datos: **memoriza** en lugar de *entender*.

> 📌 **Definición simple**: El modelo rinde bien en los datos de entrenamiento, pero falla con datos nuevos porque ha memorizado en lugar de aprender.

---

## Causas Comunes del Overfitting

| Causa                            | Explicación                                                                                      | Ejemplo                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **Modelo demasiado complejo**    | Tiene demasiados parámetros o flexibilidad en relación con la cantidad/simplicidad de los datos. | Ajustar un polinomio de grado 10 a 15 puntos que siguen una línea recta.                         |
| **Pocos datos de entrenamiento** | No hay suficiente información para aprender patrones generalizables. El modelo ajusta el ruido.  | Con solo 5 ejemplos, el modelo puede "pasar por todos los puntos", pero fallar con datos nuevos. |
| **Ruido en los datos**           | El modelo aprende errores o anomalías como si fueran patrones reales.                            | Datos mal etiquetados o con errores que el modelo intenta memorizar.                             |
| **Entrenamiento excesivo**       | Aun si el modelo es adecuado, entrenarlo demasiado tiempo hace que memorice.                     | Después de muchas épocas, el modelo deja de aprender y empieza a copiar el entrenamiento.        |

---

## Sobre la cantidad de datos

* Si **tienes pocos datos** y un **modelo muy complejo**, este podría *ajustarse perfectamente* a esos pocos puntos.
* Pero eso no implica que **haya aprendido bien**.
* Al llegar nuevos datos, ese ajuste perfecto puede resultar **muy pobre**.

> 🎯 **Conclusión**: Con pocos datos, un modelo complejo **no tiene suficiente evidencia** para distinguir entre **señal** (patrón general) y **ruido** (casualidades del conjunto de entrenamiento).

---

## 🧠 Resumen de las causas del Overfitting

* Modelo demasiado complejo para los datos o la tarea.
* Conjunto de datos de entrenamiento muy pequeño.
* Presencia excesiva de ruido en los datos.
* Entrenamiento durante demasiadas iteraciones (épocas).

---

## 🧩 Underfitting (Subajuste)

El **underfitting** ocurre cuando un modelo es **demasiado simple** para captar la complejidad real de los datos de entrenamiento. Como resultado:

* **No aprende bien** los patrones presentes.
* **Comete muchos errores**, incluso con los datos con los que fue entrenado.
* Falla en generalizar a nuevos datos porque **ni siquiera ha logrado aprender los datos originales**.

> 📌 **Definición simple**: El modelo no está aprendiendo ni siquiera los patrones de entrenamiento, y por eso comete errores altos *en todo*.

---

## ¿Cómo se ve el underfitting?

| Tipo de error                  | Resultado | ¿Por qué ocurre?                                                                     |
| ------------------------------ | --------- | ------------------------------------------------------------------------------------ |
| **Error en entrenamiento**     | **Alto**  | El modelo no logra ajustarse a los patrones presentes en los datos.                  |
| **Error en prueba/validación** | **Alto**  | Si no entendió los datos de entrenamiento, difícilmente podrá entender datos nuevos. |

> ❗ El rendimiento es pobre de forma consistente, tanto en entrenamiento como en validación.

---

## Causas Comunes del Underfitting

| Causa                                  | Explicación                                                                                       | Ejemplo                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Modelo demasiado simple**            | Tiene pocos parámetros o una estructura rígida que no puede capturar la complejidad de los datos. | Usar una línea recta para datos que tienen una forma curva.       |
| **Datos de entrada poco informativos** | Las variables (features) no contienen suficiente información relevante.                           | Predecir precios de casas solo con el número de ventanas.         |
| **Entrenamiento insuficiente**         | El modelo no tuvo suficiente tiempo o ciclos de entrenamiento para aprender los patrones.         | Cortar el entrenamiento antes de que el error baje lo suficiente. |

---

## 🧠 Resumen

* El underfitting es lo **opuesto** al overfitting.
* El modelo **no aprende bien** ni siquiera los datos de entrenamiento.
* Puede deberse a una arquitectura demasiado simple, mala calidad de datos o entrenamiento insuficiente.
* Los **errores serán altos en todas las fases**: tanto en entrenamiento como en prueba.


# 📚 Bias-Variance Tradeoff (Compromiso Sesgo-Varianza)



## 🎯 ¿Qué es el Bias-Variance Tradeoff?

Es el equilibrio que buscamos entre dos fuentes de error en los modelos de Machine Learning:

* **Sesgo (Bias)**: Error por suposiciones demasiado simplistas.
* **Varianza (Variance)**: Error por sensibilidad excesiva a los datos de entrenamiento.
```
Nuestro objetivo es **minimizar el error total** que un modelo comete en datos que nunca ha visto antes.
```


## Manifestación Práctica del Tradeoff Sesgo-Varianza

## 🔍 1. Comparación de Errores: ¿Sesgo o Varianza?

Cuando entrenas un modelo, puedes calcular:

* **Error de entrenamiento**: el error que comete el modelo sobre los datos con los que fue entrenado.
* **Error de validación/prueba**: el error que comete sobre datos que nunca ha visto.

Comparar estos errores te permite diagnosticar lo siguiente:

### 🔴 A. Alto Sesgo (Underfitting)

* **Error de entrenamiento: Alto**
* **Error de validación: Similarmente alto**
* **Diferencia entre ambos: Pequeña**

👉 Esto indica que el modelo es **demasiado simple** para capturar los patrones del problema. No aprende bien ni siquiera los datos de entrenamiento. Necesitas un modelo más complejo.

### 🔵 B. Alta Varianza (Overfitting)

* **Error de entrenamiento: Bajo**
* **Error de validación: Alto**
* **Diferencia entre ambos: Grande**

👉 Esto indica que el modelo se ha **ajustado demasiado** a los datos de entrenamiento (incluso al ruido). No generaliza bien. Puede requerir regularización, simplificación, o más datos.

### ✅ C. Buen Equilibrio (Sweet Spot)

* **Error de entrenamiento: Bajo**
* **Error de validación: También bajo (ligeramente más alto)**
* **Diferencia: Pequeña**

👉 El modelo ha aprendido lo suficiente **sin sobreajustarse**. Está generalizando bien.

---

## 📈 2. Curvas de Aprendizaje Típicas

Las curvas de aprendizaje muestran cómo cambian los errores a medida que:

* Aumentas el **tamaño del conjunto de entrenamiento**, o
* Aumentas la **complejidad del modelo**.

Aquí cómo se ven típicamente en cada caso:

### 🔴 A. Alto Sesgo (Underfitting)

**Curvas vs tamaño del entrenamiento:**

* Ambas curvas (entrenamiento y validación) están **altas** y cercanas entre sí.
* A medida que se agregan más datos, **no hay gran mejora**.
* **El modelo no mejora con más datos porque es demasiado simple.**

📊 Ejemplo gráfico (mental):

```
Error
  |
  |       --------------------   ← entrenamiento
  |       --------------------   ← validación
  |
  +----------------------------> tamaño de los datos
```

---

### 🔵 B. Alta Varianza (Overfitting)

**Curvas vs tamaño del entrenamiento:**

* **Error de entrenamiento muy bajo**
* **Error de validación mucho más alto**
* Hay una **gran brecha (gap)** entre ambos.
* A medida que se añaden más datos, la brecha puede **disminuir**.

📊 Ejemplo gráfico:

```
Error
  |
  |  \                         ← entrenamiento (muy bajo)
  |    \     ________          ← validación (alto, pero bajando con más datos)
  |
  +---------------------------> tamaño de los datos
```

---

### ✅ C. Buen Equilibrio (Sweet Spot)

**Curvas vs tamaño del entrenamiento:**

* El error de entrenamiento es bajo, pero no perfecto.
* El error de validación es un poco más alto, pero **ambos convergen**.
* Es señal de que el modelo está generalizando bien.

📊 Ejemplo gráfico:

```
Error
  |
  |   \         
  |    \______                     ← entrenamiento
  |           \_______            ← validación
  |
  +-----------------------------> tamaño de los datos
```

---



# Estrategias Generales para Combatir el Underfitting y el Overfitting

En el entrenamiento de modelos de machine learning, uno de los principales desafíos es encontrar el equilibrio entre **subajuste (underfitting)** y **sobreajuste (overfitting)**. A continuación, se presentan estrategias prácticas y razonadas para abordar cada caso.

---

## ¿Cómo combatir el UNDERFITTING?

El underfitting ocurre cuando un modelo es demasiado simple para capturar los patrones subyacentes de los datos. Algunas estrategias efectivas incluyen:

### 1. Aumentar la complejidad del modelo

* **Elegir un modelo más expresivo**:

  * Si usas regresión lineal, prueba con regresión polinómica (añadiendo términos como x², x³, etc.).
    👉 *¿Qué hiperparámetro controlarías aquí?* El grado del polinomio.
  * Si usas árboles de decisión, permite que crezcan más profundos.
  * Considera modelos más complejos como redes neuronales o SVM con kernel no lineal.

### 2. Ingeniería de características (Feature Engineering)

* Agrega nuevas características relevantes.
* Introduce combinaciones de variables (interacciones).
* Asegúrate de incluir representaciones adecuadas del dominio del problema.

### 3. Asegurar entrenamiento suficiente

* Aumenta el número de épocas o iteraciones.
* Verifica que el algoritmo haya tenido oportunidad de converger.

### 4. Ajustar la regularización

* Si estás aplicando regularización (por ejemplo, con parámetro λ), revisa que **no sea excesiva**.
  Un λ demasiado alto puede hacer que el modelo sea demasiado simple.
  👉 *Reducir λ puede permitirle aprender más patrones reales.*

---

## ¿Cómo combatir el OVERFITTING?

El overfitting ocurre cuando el modelo aprende demasiado bien los datos de entrenamiento, incluyendo el ruido o las particularidades del conjunto, y falla al generalizar. Estas estrategias ayudan a evitarlo:

### 1. Regularización

* Penaliza los valores grandes de los parámetros del modelo (θ) para evitar que se ajusten demasiado a los datos.

  * **L1 (Lasso)**: puede llevar a modelos más escuetos (sparse).
  * **L2 (Ridge)**: reduce gradualmente todos los pesos.

  👉 *¿Cómo ayuda esto?* Reduce la complejidad efectiva del modelo sin cambiar su estructura base.

### 2. Selección de características o reducción de dimensionalidad

* Elimina variables irrelevantes o ruidosas.
* Aplica técnicas como **PCA (Análisis de Componentes Principales)** o métodos de selección automatizada para reducir la dimensionalidad.

### 3. Early Stopping (Detención Temprana)

* Monitorea el error en el conjunto de validación durante el entrenamiento.
* Si el error de validación comienza a aumentar mientras el error de entrenamiento sigue bajando, detén el entrenamiento.
  👉 *Esto previene que el modelo se "memorice" los datos.*

### 4. Más datos de entrenamiento

* Cuantos más ejemplos diversos tengas, más robusto será el modelo.
* Ayuda a reducir el sesgo inducido por un conjunto pequeño o no representativo.

### 5. Filtrar o limpiar datos (con cuidado)

* Identifica y elimina outliers si están claramente afectando el modelo.
  ⚠️ *Hazlo solo si puedes justificarlo bien*, ya que podrías introducir sesgo si te excedes.

### 6. Métodos de ensamblaje (Ensemble Methods)

* Combina múltiples modelos para obtener predicciones más estables y precisas:

  * **Bagging** (como Random Forests) reduce la varianza.
  * **Boosting** (como XGBoost) puede mejorar el sesgo y la varianza a la vez.

---


### Comparación de Estrategias contra Underfitting y Overfitting

| Categoría                                | Combatir Underfitting                                                               | Combatir Overfitting                                                |
| ---------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Complejidad del Modelo**               | Aumentar complejidad (p. ej., redes más grandes, polinomios, árboles más profundos) | Reducir complejidad (modelos más simples, limitar profundidad)      |
| **Características**                      | Añadir o transformar características relevantes                                     | Eliminar características irrelevantes o ruidosas                    |
| **Entrenamiento**                        | Aumentar número de iteraciones/épocas                                               | Early stopping (detener cuando se sobreajusta al set de validación) |
| **Regularización**                       | Reducir regularización (bajar λ)                                                    | Añadir o aumentar regularización L1 / L2                            |
| **Datos**                                | No suele ser la primera opción, pero ayuda                                          | Añadir más datos de entrenamiento                                   |
| **Dimensionalidad**                      | No aplica directamente                                                              | Reducción de dimensionalidad (PCA, selección de características)    |
| **Técnicas avanzadas**                   | —                                                                                   | Técnicas de ensamblaje (Bagging, Boosting)                          |
| **Aumento de Datos (Data Augmentation)** | —                                                                                   | Útil para generalizar mejor (imágenes, texto, audio)                |



## 🧠 Tipos de Bias en Machine Learning

| Concepto                 | ¿Qué es?                                  | ¿Dónde aparece?        |
| ------------------------ | ----------------------------------------- | ---------------------- |
| **Bias como intercepto** | Columna de unos → parámetro β₀            | Modelos lineales, RN   |
| **Bias como sesgo**      | Suposiciones erróneas → error sistemático | Bias-Variance Tradeoff |

---

## 🔍 1. **Bias como Parámetro (Intercepto)**

* Se refiere al término independiente en modelos lineales:

  Y = β₀ + β₁X₁ + β₂X₂ + …

* Es un **parámetro aprendido** por el modelo.

* Se introduce agregando una **columna de unos** a la matriz de entrada X.

---

## 🔍 2. **Bias como Error Sistemático**

* Error causado por suposiciones rígidas (por ejemplo, que todo es lineal).
* Se define como la **diferencia entre la predicción promedio del modelo y la realidad**.
* Es una **medida de error teórico**, no un parámetro.

---

## 📈 ¿Qué ocurre con Sesgo y Varianza?

| Tipo de Modelo           | Sesgo | Varianza | Resultado             |
| ------------------------ | ----- | -------- | --------------------- |
| Muy simple               | Alto  | Bajo     | Underfitting          |
| Muy complejo             | Bajo  | Alto     | Overfitting           |
| Equilibrado (sweet spot) | Medio | Medio    | Generalización óptima |

---

## 📉 Error Total

El error total en un modelo puede expresarse como:

**Error total = Sesgo² + Varianza + Error irreducible**

* **Sesgo²**: Error por suposiciones erróneas (underfitting).
* **Varianza**: Error por sobreajuste al conjunto de entrenamiento (overfitting).
* **Error irreducible**: Ruido inherente al problema. No se puede eliminar.

---

## ⚖️ El Compromiso

* Reducir **sesgo** suele **aumentar varianza**.
* Reducir **varianza** suele **aumentar sesgo**.
* El punto óptimo (💡 *sweet spot*) es donde el error total es mínimo y el modelo **generaliza bien**.

---

## 🛠️ ¿Cómo controlar la complejidad?

Tú eliges la complejidad del modelo con las siguientes "perillas":

* **Tipo de modelo**: lineal vs red neuronal, árbol de decisión, etc.
* **Hiperparámetros**:

  * Grado del polinomio
  * Profundidad del árbol
  * Capas y neuronas en redes
* **Regularización**: penaliza la complejidad (controla el overfitting).

---

## 📊 ¿Cómo encontrar el sweet spot?

1. **División de Datos**:

   * Entrenamiento: aprende los parámetros.
   * Validación: elige el mejor modelo/hiperparámetro.
   * Prueba: evalúa el modelo final.

2. **Curvas de Aprendizaje**:

   * Gráfica de error de entrenamiento y validación al aumentar la complejidad.
   * El sweet spot suele estar donde el error de validación es mínimo.

3. **Validación Cruzada (Cross-Validation)**:

   * Evalúa el rendimiento de forma más robusta.
   * Recomendado para seleccionar hiperparámetros con mayor confianza.

---

## 🧩 Conclusión

* El **bias-variance tradeoff** es uno de los conceptos más fundamentales para entender por qué un modelo no está funcionando bien.
* **No hay una fórmula mágica** para saber cuánta complejidad es ideal: lo descubrimos **experimentando** y validando.
* Tu tarea como modelador es ajustar esa complejidad para que el modelo **aprenda lo suficiente pero no memorice**.


---


## Regularización: Previniendo el Sobreajuste sin Perder Capacidad de Aprendizaje

### Formula

$$\text{Término de Regularización L2} = \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

### ¿Qué es la Regularización?

La **regularización** es una técnica fundamental que modifica la función de coste de un modelo para reducir su complejidad y evitar el sobreajuste (*overfitting*). Lo hace penalizando los **parámetros grandes** del modelo (θ), lo que tiende a producir modelos más simples y generalizables.

---

### ¿Por qué se necesita?

* Queremos minimizar el **error de predicción** en los datos de entrenamiento.
* Pero si el modelo es demasiado complejo (por ejemplo, tiene parámetros θ muy grandes), puede **memorizar los datos** en lugar de aprender patrones generales.
* Esto causa **sobreajuste**, es decir, bajo error en entrenamiento pero alto error en datos nuevos.

La regularización combate esto añadiendo una **penalización por complejidad** directamente en la función de coste.

---

### ¿Cómo se modifica la función de coste?

Tomemos como ejemplo la **Regresión Lineal Regularizada con L2** (también conocida como *Ridge Regression*).

La nueva función de coste se define como:

$$J(θ) = (1 / 2m) ∑*{i=1}^{m} (h\_θ(x^{(i)}) - y^{(i)})² + (λ / 2m) ∑*{j=1}^{n} θ\_j²$$

Donde:

* m es el número de ejemplos.
* h\_θ(x) es la predicción del modelo.
* y^{(i)} es el valor real para el ejemplo i.
* λ ≥ 0 es el **coeficiente de regularización**.
* La suma en el segundo término excluye generalmente θ₀ (el sesgo/intercepto), ya que no suele penalizarse.

> Esto no es mas que la suma de nuestra funcion de coste mas la funcion de regularizacion
---

### ¿Cómo afecta al Descenso de Gradiente?

La actualización de los parámetros también se modifica para incluir la penalización. Si antes actualizábamos así:


$$
\theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}
$$

Con regularización L2, el gradiente se ajusta así:


$$
\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right]
$$


Esto significa que cada θ\_j es "empujado" ligeramente hacia cero en cada paso, evitando que crezca demasiado.

> Nota: La penalización **no aplica a θ₀**, así que su actualización se mantiene igual que antes.

---

### Beneficios de la Regularización

* **Reduce el riesgo de overfitting**, haciendo que el modelo generalice mejor.
* **Controla la complejidad** del modelo sin cambiar su arquitectura.
* **Fácil de implementar**, ya que solo requiere ajustar la función de coste y el gradiente.

### Que pasa si lmbda es 0?
Si `lmbda_reg = 0`, entonces el **término de regularización** se anula completamente:

$$
\text{término\_de\_regularización} = \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2 = 0
$$

Por tanto:

$$
\text{coste\_total} = \text{coste\_original} + 0 = \text{coste\_original}
$$

### ¿Qué significa esto?
Si el parámetro de regularización λ = 0, la función de coste regularizada se convierte exactamente en la función de coste original, sin regularización. Esto se debe a que el término de penalización (por ejemplo, en regularización L2, la suma de los cuadrados de los parámetros) se multiplica por λ:

---
# Pasos de Modificacion del algoritmo


## ⚙️ Modificación del Modelo de Regresión Lineal con Regularización L2

### 📌 Paso 1: Modificar la Función de Coste

Comenzamos con la **función de coste estándar** utilizada en regresión lineal, conocida como **Error Cuadrático Medio (MSE)**. Esta función mide qué tan lejos están las predicciones del modelo respecto a los valores reales:

### 📉 Función de Coste Original (MSE)

$$
J(θ) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_θ(x^{(i)}) - y^{(i)} \right)^2
$$

Aquí:

* $m$ es el número de ejemplos de entrenamiento.
* $h_θ(x^{(i)})$ es la predicción del modelo para el ejemplo $i$.
* $y^{(i)}$ es el valor real para ese ejemplo.
* $θ$ representa los parámetros del modelo (también llamados coeficientes o pesos).

---

### 🛡️ Agregando Regularización L2 (Ridge)

Para evitar que los coeficientes del modelo se vuelvan demasiado grandes (lo que podría llevar a **sobreajuste**), agregamos un término de **penalización** que castiga los valores grandes de $θ$, excepto el término de sesgo $θ_0$.

### 🧮 Función de Coste con Regularización L2

$$
J(θ) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_θ(x^{(i)}) - y^{(i)} \right)^2 + \frac{λ}{2m} \sum_{j=1}^{n} θ_j^2
$$

#### Donde:

* $λ$ (lambda) es el **parámetro de regularización**. Controla cuánto penalizamos los coeficientes grandes.
* El segundo término **no incluye** a $θ_0$, ya que este representa el sesgo base del modelo y **no debería penalizarse**.
* La suma de la regularización va desde $j = 1$ hasta $n$, dejando fuera $j = 0$.

---

🔍 **¿Por qué no penalizamos $θ_0$?**

Porque $θ_0$ actúa como el punto de partida del modelo (es el valor que predice cuando todas las variables son cero). Penalizarlo no contribuye a controlar la complejidad del modelo y podría llevar a un ajuste subóptimo.


---

## 🎯 ¿Por qué no se penaliza θ₀ en la regularización L2?

Cuando aplicamos **regularización L2 (Ridge)** en un modelo de regresión lineal, es importante entender **por qué el término de intercepción $θ_0$** —también conocido como el **sesgo o bias**— **no se incluye en la penalización**.

### 🧠 ¿Qué es θ₀?

* $θ_0$ es el valor que predice el modelo **cuando todas las variables de entrada son cero**.
* Es como el punto de partida o el "ajuste base" del modelo.

---

### 🧪 ¿Cómo se implementa esto?

En la **función de coste**, solo se penalizan los coeficientes que están asociados a las **características de entrada**, es decir, desde $θ_1$ hasta $θ_n$:

$$
J(θ) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_θ(x^{(i)}) - y^{(i)}\right)^2 + \frac{λ}{2m} \sum_{j=1}^{n} θ_j^2
$$

👉 Observa que la **suma del segundo término empieza en $j = 1$**, **no en 0**. Por eso, $θ_0$ queda **fuera de la penalización**.

---

### 🔄 ¿Y en la actualización del gradiente?

Al actualizar los parámetros durante el entrenamiento (por ejemplo, con gradiente descendente), el **término de penalización**:

$$
\frac{λ}{m}θ_j
$$

...**también se aplica solamente a $θ_1, θ_2, ..., θ_n$**.
Esto se refleja en el código, donde usualmente se escribe algo como `theta[1:] += ...`.

---

### ⚖️ ¿Por qué no penalizamos $θ_0$?

* ✅ **Porque no representa una variable de entrada**, sino un desplazamiento general en todas las predicciones.
* ✅ Penalizarlo **no ayuda a prevenir el sobreajuste** causado por variables complejas o irrelevantes.
* ❌ De hecho, podría llevar a un **modelo subóptimo**, limitando la capacidad del modelo para ajustarse correctamente a los datos.

---

### 🧵 En resumen:

| Elemento             | ¿Se penaliza en L2? | ¿Por qué?                                                             |
| -------------------- | ------------------- | --------------------------------------------------------------------- |
| $θ_0$ (intercepción) | ❌ No                | No es una característica; penalizarlo puede dañar el ajuste base.     |
| $θ_j$, $j ≥ 1$       | ✅ Sí                | Representan características del modelo que pueden causar sobreajuste. |

---

✅ **El objetivo principal de la regularización L2 es controlar la complejidad del modelo penalizando solo los coeficientes asociados a las variables de entrada**, no al sesgo general.

---

## **Paso 2: Modificar la Función de Descenso de Gradiente (con Regularización L2)**

### **Paso A: Preparar el vector para la penalización**

La regularización L2 añade una **penalización a los valores grandes de los parámetros** para evitar que el modelo sobreajuste.
Sin embargo, **no debemos penalizar el parámetro θ₀** (el término independiente o sesgo), ya que no está asociado a ninguna característica y su penalización podría afectar negativamente el entrenamiento.

Por eso, vamos a crear una **copia del vector `θ` (theta)**, pero con el primer valor igual a cero.

En código, esto sería:

```python
theta_para_penalizacion = theta.copy()
theta_para_penalizacion[0] = 0
```

Esto da como resultado un nuevo vector:

$$
\theta_{\text{penalización}} = 
\begin{bmatrix}
0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_n
\end{bmatrix}
$$

### **Paso B: Calcular la penalización para el gradiente**

Ahora, vamos a calcular el **vector de penalización** que sumaremos al gradiente original.
Este vector se obtiene multiplicando cada elemento de `θ_para_penalizacion` por un escalar que incluye el **parámetro de regularización** λ y el número de ejemplos m:

$$
\text{penalización\_gradiente} = \frac{\lambda}{m} \cdot \theta_{\text{penalización}}
$$

Este vector tiene el mismo tamaño que `θ` y **solo penaliza los parámetros distintos de θ₀**.

---

## 🐛 Proceso de Depuración Completo: Problemas al Ejecutar el Modelo con Distintos Lambdas

### Contexto

Estábamos probando nuestro modelo de regresión lineal regularizada con distintos valores de `lambda`, con el siguiente bloque:

```python
for lmbda_reg in lmbda_valores:
    theta_calculado, historial_coste = descenso_gradiente(
        X_bias_scaled, y, theta_inicial.copy(), alpha, num_iteraciones, lmbda_reg
    )
```

El objetivo era observar cómo cambiaban los parámetros `theta` y la función de coste con distintos grados de regularización. Pero **empezaron a ocurrir problemas graves**:

* La máquina se volvía extremadamente lenta.
* El script nunca terminaba de ejecutarse.
* No había errores explícitos visibles al principio.

---

### 🧩 Etapa 1: TypeError en `num_iteraciones`

Error observado:

```
TypeError: 'float' object cannot be interpreted as an integer
```

Esto ocurrió en la función `descenso_gradiente`, en esta línea:

```python
for i in range(num_iteraciones):
```

**Hipótesis inicial**: `num_iteraciones` no estaba llegando como entero, sino como `float`.

---

#### Paso 1.1: Verificar `num_iteraciones` ANTES del bucle

Agregamos este print en el script principal:

```python
print(f"DEBUG SCRIPT: Antes del bucle de lambdas, num_iteraciones = {num_iteraciones}, tipo = {type(num_iteraciones)}")
```

**Salida esperada** (si estuviera bien):

```
DEBUG SCRIPT: Antes del bucle de lambdas, num_iteraciones = 200, tipo = <class 'int'>
```

Pero eso estaba bien, así que fuimos más adentro.

---

#### Paso 1.2: Verificar `num_iteraciones` DENTRO de la función

Agregamos este print al inicio de `descenso_gradiente`:

```python
print(f"DEBUG GD: Al inicio de la función, num_iteraciones = {num_iteraciones}, tipo = {type(num_iteraciones)}")
```

**Salida obtenida**:

```
DEBUG GD: Al inicio de la función, num_iteraciones = 0.001, tipo = <class 'float'>
```

**Descubrimiento**: Estábamos pasando mal los argumentos en la llamada. Lo que estaba llegando como `num_iteraciones` en realidad era `alpha`.

---

#### ✔️ Solución 1: Corregir el orden de los argumentos

La llamada correcta debía ser:

```python
theta_calculado, historial_coste = descenso_gradiente(
    X_bias_scaled, y, theta_inicial.copy(), alpha_real, num_iteraciones_real, lmbda_reg
)
```

Con esto, el error de tipo desapareció.

---

### 🧩 Etapa 2: Explosión de Formas y Lentitud

Aunque el script ya no tiraba error, ahora tenía síntomas distintos:

* El modelo se volvía extremadamente lento.
* `theta` tenía formas gigantes.
* `errores` explotaba en tamaño.

---

#### Paso 2.1: Verificar formas en cada iteración

Agregamos estos prints dentro de `descenso_gradiente`, después de calcular `predicciones` y `errores`:

```python
print(f"Iter {i+1} DEBUG GD: theta.shape = {theta.shape}")
print(f"Iter {i+1} DEBUG GD: predicciones.shape = {predicciones.shape}")
print(f"Iter {i+1} DEBUG GD: errores.shape = {errores.shape}")
```

**Salida obtenida (ejemplo con lambda=0):**

```
Iter 1 DEBUG GD: theta.shape = (9, 20640)
Iter 1 DEBUG GD: predicciones.shape = (20640, 1)
Iter 1 DEBUG GD: errores.shape = (20640, 20640)
```

**Algo estaba muy mal.**

---

#### Paso 2.2: Diagnóstico más fino del error de dimensiones

Agregamos más prints para analizar justo antes y después de calcular `errores`:

```python
predicciones = calcular_hipotesis(X_bias, theta)
print(f"Iter {i+1} DEBUG GD ANTES DE ERRORES: theta.shape={theta.shape}, predicciones.shape={predicciones.shape}, y.shape={y.shape}")
errores = predicciones - y
print(f"Iter {i+1} DEBUG GD DESPUÉS DE ERRORES: errores.shape={errores.shape}")
```

**Salida clave**:

```
Iter 1 DEBUG GD ANTES DE ERRORES: theta.shape=(9,1), predicciones.shape=(20640,1), y.shape=(20640,)
Iter 1 DEBUG GD DESPUÉS DE ERRORES: errores.shape=(20640,20640)
```

**Descubrimiento**: `y` tenía forma `(20640,)` (vector de 1 dimensión), mientras que `predicciones` era `(20640,1)`. Python hizo *broadcasting* para hacer compatible la resta, creando una matriz de tamaño `(20640,20640)`.

---

#### ✔️ Solución 2: Forzar forma correcta de `y`

Al inicio de `descenso_gradiente`, agregamos:

```python
y = y.reshape(-1, 1)
```

**Resultado** tras el fix:

```
Iter 1 DEBUG GD ANTES DE ERRORES: theta.shape=(9,1), predicciones.shape=(20640,1), y.shape=(20640,1)
Iter 1 DEBUG GD DESPUÉS DE ERRORES: errores.shape=(20640,1)
```

✅ Ahora todas las formas se mantenían correctas. La lentitud extrema desapareció y el entrenamiento se comportó como se esperaba.

---

#### Paso 2.3 (Opcional): Verificar explosiones numéricas

Aún con formas correctas, se puede tener lentitud por problemas numéricos. Verificamos esto dentro de `calcular_coste`:

```python
print(f"CALC_COSTE: errores_cuadraticos.shape = {errores_cuadraticos.shape}")
print(f"CALC_COSTE: Primeros 3 errores_cuadraticos: {errores_cuadraticos[:3].T}")
print(f"CALC_COSTE: Hay NaNs? {np.isnan(errores_cuadraticos).any()}, Hay Infs? {np.isinf(errores_cuadraticos).any()}")
```

**Posible salida si hubiera overflows**:

```
CALC_COSTE: errores_cuadraticos.shape = (20640, 1)
CALC_COSTE: Primeros 3 errores_cuadraticos: [[inf inf inf]]
CALC_COSTE: Hay NaNs? False, Hay Infs? True
```

Pero en nuestro caso, tras corregir la forma de `y`, **no hubo problemas numéricos**.

---

### ✅ Conclusión

Gracias a un proceso metódico de **debugging con prints**, descubrimos dos errores graves:

1. Parámetros mal pasados (`alpha` y `num_iteraciones` estaban invertidos).
2. Forma de `y` incorrecta, causando explosiones de matrices y lentitud.

Estas correcciones fueron **críticas para el funcionamiento correcto del modelo**, y para que pudiéramos hacer las pruebas con múltiples valores de `lambda`.


## ✅ CheckList de Buenas Prácticas para Debugging y Modelado

### 📌 Variables y Parámetros

* [ ] Confirmar que `alpha`, `num_iteraciones`, `lambda` están en el **orden correcto** al llamar funciones.
* [ ] Asegurarse de que `num_iteraciones` sea un `int`, no un `float`.

### 📏 Formas de las Matrices

* [ ] Convertir `y` a forma `(m, 1)` antes de operaciones vectorizadas:

  ```python
  y = y.reshape(-1, 1)
  ```
* [ ] Verificar que `theta` tenga forma `(n, 1)` si `X` es `(m, n)`.

### 🔍 Verificaciones Intermedias (Debugging)

* [ ] Agregar prints de forma en puntos clave:

  ```python
  print(f"theta.shape = {theta.shape}")
  print(f"predicciones.shape = {predicciones.shape}")
  print(f"errores.shape = {errores.shape}")
  ```
* [ ] Usar prints con `.any()` para detectar `NaN` o `Inf`:

  ```python
  print(np.isnan(matriz).any(), np.isinf(matriz).any())
  ```

### 🧪 Testeo Controlado

* [ ] Probar primero con un número pequeño de iteraciones (`num_iteraciones = 5 o 10`) y `lambda = 0` para validar la lógica antes de entrenar completamente.

### ⚠️ Señales de Error Común

| Síntoma                         | Posible Causa                             |
| ------------------------------- | ----------------------------------------- |
| TypeError con `range()`         | `num_iteraciones` es `float`              |
| Errores gigantes `(m, m)`       | `y` tiene forma `(m,)` → usar `.reshape`  |
| `theta` con forma rara `(n, m)` | Broadcasting incorrecto o errores previos |
| Script extremadamente lento     | Matrices gigantes por formas incorrectas  |
| `coste` devuelve `inf` o `nan`  | Overflow → revisar `alpha` o escalado     |




## 📊 **Resultados de la Experimentación con Regularización L2 (λ)**

### ¿Cómo se comportó θ 0(el intercepto) a medida que λ cambiaba?

Realizamos experimentos utilizando un conjunto de datos de precios de casas en California para analizar cómo afecta la regularización L2 en un modelo de regresión lineal.

### Regresion Lineal:
![alt text](<Regresion_Lineal.py/Coeficientes theta.png>)
### Regresion Logistica:
![alt text](<Regresion_Logisitica/Coeficientes ttheta en funcion del lmbda.png>) 




### 🧪 **¿Qué me dice esto?**

En este análisis, **no aplicamos penalización a θ₀** (el término que corresponde al valor inicial). Esto se debe a que cuando **λ = 0**, no se agrega ninguna restricción, lo que significa que no se penaliza este término.

Observamos en el gráfico cómo los coeficientes (los valores multiplicados por las variables) cambian cuando **λ** varía. A medida que **λ** aumenta, **los coeficientes tienden a hacerse más pequeños**, lo que significa que estamos "empujando" los coeficientes hacia **0**.

### 🔎 **¿Por qué sucede esto?**

Esto ocurre por la **regularización L2**. Cuando aumentamos **λ**:

* El modelo se hace más simple, ya que reduce los valores de los coeficientes.
* Si **λ** es grande, el modelo no confiará tanto en cada variable, evitando que alguna variable sea demasiado importante. Esto ayuda a **evitar el sobreajuste** (cuando el modelo "se ajusta demasiado" a los datos de entrenamiento).

### 📏 **Función de Coste con Regularización L2**

La función de coste de la regresión lineal con regularización L2, también conocida como **Ridge Regression**, es la siguiente:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

Donde:

* **J(θ)** es la función de coste, que mide qué tan bien se ajusta el modelo.
* **hₜₕₐ(xᵢ)** es la predicción del modelo para los datos de entrada.
* **yᵢ** es el valor real que queremos predecir.
* **λ** es el parámetro de regularización, que controla cuánto penalizamos a los coeficientes.
* **θⱼ** son los coeficientes de las características.
* **m** es el número de ejemplos de entrenamiento.

### ⚖️ **Resumen:**

* **Cuando λ = 0:** No hay penalización, lo que puede hacer que el modelo se ajuste demasiado a los datos, llevando a un **sobreajuste**.

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2
$$

* **Cuando λ es alto:** Los coeficientes se hacen más pequeños, favoreciendo un modelo más simple, lo que ayuda a evitar el **sobreajuste**(overfitting) y mejora la generalización del modelo.

---


### Regresion Lineal:
![alt text](<Regresion_Lineal.py/Curvas de coste lmbda.png>)
### Regresion Logistica:
![alt text](<Regresion_Logisitica/Curvas coste de lmbda.png>)

---

## 📉 Análisis del Gráfico: *Curvas de Coste por Lambda*

Este gráfico nos muestra cómo el parámetro \$\lambda\$ (lambda, de regularización) afecta el **aprendizaje** de nuestro modelo de regresión lineal.

### 1. **Verificación de la Convergencia**

Cada línea en el gráfico representa cómo cambia el coste durante el entrenamiento, para un valor distinto de \$\lambda\$. Lo que buscamos es que el coste:

* Disminuya progresivamente.
* Se estabilice (indica que el modelo ha "convergido").

> ⚠️ **Nota**: Si el coste *no* baja o se comporta de forma rara para cierto \$\lambda\$, podría indicar que esa configuración no está funcionando bien. Tal vez \$\lambda\$ es demasiado alto o interactúa mal con el valor de `alpha`.

---

### 2. **Comparación del Error Final con Diferentes Valores de \$\lambda\$**

* **Cuando \$\lambda = 0\$ (sin regularización):**
  El modelo tiene total libertad para ajustarse a los datos de entrenamiento. Por eso, el coste final es usualmente **más bajo**: el modelo "memoriza" los datos.

* **Cuando \$\lambda > 0\$ (con regularización):**
  A medida que aumentamos \$\lambda\$, el modelo debe **equilibrar dos objetivos**:

  1. Minimizar el error de predicción.
  2. Mantener los valores de los parámetros \$\theta\_j\$ **pequeños** (evitar que crezcan mucho).

  Esto suele llevar a un **coste de entrenamiento más alto**, pero también reduce el riesgo de *overfitting* (sobreajuste).

---

### ✅ **Resumen del gráfico:**

* Verifica si el modelo está entrenando correctamente (convergencia).
* Muestra cómo el modelo reacciona ante distintos niveles de regularización.
* Ayuda a detectar si un \$\lambda\$ **demasiado grande** está haciendo que el modelo sea **demasiado simple** (lo que llamamos *underfitting*).

---

## 🎯 ¿Qué pasa con los coeficientes \$\theta\$ cuando usamos regularización?

### ¿Algunos coeficientes se reducen a cero o cerca de cero más rápido que otros?

✅ **Sí**, cuando aumentamos \$\lambda\$, algunos coeficientes \$\theta\_j\$ se acercan a cero más rápido que otros.

Esto pasa porque con **regularización fuerte**, el modelo trata de **penalizar más** a ciertos coeficientes. Si ve que una variable no está aportando mucho, la "castiga" y empuja su \$\theta\$ hacia cero.

```
❗ Si un coeficiente se hace pequeño o casi cero con una lambda alta, el modelo cree que esa variable no es tan importante para hacer predicciones.
```

---

### 🔍 ¿Qué implica esto?

* El modelo está buscando **simplicidad**: usar solo las variables que realmente ayudan.
* Si un \$\theta\$ baja rápido, es porque el modelo **confía menos** en esa variable.
* Es como una **selección automática de características**: las menos útiles se "apagan" solas.

---

### 🔬 Conexión con la teoría

* En la regularización L2 (*Ridge*), el coste total incluye un término adicional que penaliza los \$\theta\_j\$ grandes:

  $$
  J(\theta) = \text{Error cuadrático} + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
  $$

* Como ves, se penaliza tener coeficientes grandes. Por eso, el modelo prefiere hacerlos pequeños, **a menos que realmente sean necesarios para predecir bien**.

## 📊 Resultados de la Experimentación con Regularización L2 (λ)
A continuación se presentan los resultados obtenidos al aplicar regularización L2 a dos modelos: Regresión Lineal y Regresión Logística. Se analizaron los cambios en los coeficientes $\theta_j$ al variar el parámetro de regularización $\lambda$, observando cómo esto afecta tanto al entrenamiento como a la simplicidad del modelo.

📌 A. Modelo de Regresión Lineal
1. Estabilidad del Intercepto θ₀
En este experimento, aunque el gráfico "Coeficientes theta.png" muestra únicamente los coeficientes $\theta_1$ a $\theta_8$, al inspeccionar los arrays completos de theta_calculado, se observó que el valor de $\theta_0$ (el intercepto) se mantuvo relativamente estable.



Esto es coherente con la teoría de la regularización L2, ya que $\theta_0$ no es penalizado en la función de coste. Por lo tanto, su valor no se ve afectado significativamente por el aumento de $\lambda$.

1. Encogimiento de los Coeficientes $\theta_1$ a $\theta_8$
Del gráfico "Coeficientes theta.png", se observa que todos los coeficientes disminuyen en magnitud a medida que aumenta $\lambda$:

Por ejemplo, $\theta_1$ (línea azul) comienza en aproximadamente 0.81 cuando $\lambda$ es bajo, y se reduce hasta casi 0.05 cuando $\lambda = 1000$.

Otros coeficientes, como $\theta_5$ o $\theta_7$, bajan incluso más rápido y tienden más rápidamente a cero.

Este comportamiento refleja el efecto clásico del "shrinkage" (encogimiento): la regularización L2 penaliza los coeficientes grandes, empujándolos hacia cero.
Esto sugiere que el modelo considera que algunas variables son menos importantes para la predicción, y por tanto sus $\theta_j$ son reducidos con más fuerza. En otras palabras, el modelo automáticamente "selecciona" cuáles características conservar y cuáles descartar, aunque en Ridge nunca llegan exactamente a cero.

📌 B. Modelo de Regresión Logística
1. Estabilidad del Intercepto θ₀
De forma similar al modelo lineal, se observó que el valor de $\theta_0$ en la regresión logística se mantiene estable a pesar del aumento en $\lambda$:

Para $\lambda = 0$, $\theta_0 \approx 1.35$

Para $\lambda = 1000$, $\theta_0 \approx 1.31$

Esto nuevamente es esperado, ya que $\theta_0$ no es penalizado en la regularización L2.

2. Encogimiento de los Coeficientes $\theta_1$ a $\theta_8$
El gráfico "Coeficientes theta en función del lambda.png" muestra cómo los primeros 8 coeficientes disminuyen al aumentar $\lambda$.
Se decidió graficar solo estos primeros 8 de los 30 coeficientes disponibles en el dataset del cáncer de mama para facilitar la visualización.

Tendencia observada:

Cuando $\lambda$ es bajo, algunos $\theta_j$ comienzan con valores relativamente altos (entre 0.3 y 0.7).

A medida que $\lambda$ aumenta, todos estos coeficientes tienden hacia cero, aunque no todos con la misma rapidez.

A diferencia del modelo lineal, en este gráfico el encogimiento parece más abrupto para algunos coeficientes específicos, lo que puede deberse a que el modelo logístico es más sensible a la regularización por la naturaleza de su función de coste (log loss).

### 🔄 Comparación entre los dos modelos
Aunque ambos modelos usan el mismo principio de regularización L2, sus gráficos de coeficientes lucen diferentes debido a varios factores:

Tipo de modelo: Lineal vs Logístico.

Cantidad y tipo de variables: El modelo lineal usó un dataset más pequeño con 8 características, mientras que el logístico trabajó con 30.

Magnitud de los coeficientes: En la regresión logística, los coeficientes son más pequeños desde el inicio, lo que hace que el efecto visual del encogimiento sea más notorio o abrupto.

Ambos modelos muestran el mismo comportamiento esencial:

A mayor $\lambda$, mayor penalización, lo que lleva a coeficientes más pequeños y, por ende, a modelos más simples.

 ### A. Modelo de Regresión Lineal — "Coeficientes theta.png"
🔹 1. Estabilidad del Intercepto θ₀
Aunque el gráfico "Coeficientes theta.png" muestra únicamente los coeficientes θ₁ a θ₈, al observar los arrays de theta_calculado, se vio que el valor del intercepto θ₀ se mantuvo estable en todos los valores de λ.
Por ejemplo, para λ = 0, θ₀ ≈ 2.07, y para λ = 1000, θ₀ ≈ 2.02.
Esto es coherente con lo esperado, ya que el intercepto no es penalizado por la regularización L2, por lo tanto no se ve afectado por el aumento de λ.
🔹 2. Encogimiento de los coeficientes θ₁ a θ₈
En el gráfico se observa cómo los coeficientes disminuyen (efecto "shrinkage") al aumentar λ.
Por ejemplo, el coeficiente θ₁ (línea azul) comienza en aproximadamente 0.81 cuando λ es pequeño, y disminuye hasta alrededor de 0.25 cuando λ = 1000.

En contraste, coeficientes como θ₅ (línea morada) comienzan ya cerca de 0.05 y se mantienen prácticamente planos, lo que indica que esa característica tiene una importancia muy baja y el modelo tiende a descartarla rápidamente.

θ₇ (línea rosa) presenta una reducción más pronunciada, bajando de aproximadamente 0.5 a casi 0.1, lo cual muestra que es una característica moderadamente importante, pero que pierde peso a medida que el modelo se simplifica.

En general, los coeficientes más relevantes resisten más el encogimiento, mientras que los menos informativos tienden rápidamente hacia cero. Esto ilustra cómo la regularización actúa como un filtro automático de características.

### 📌 B. Modelo de Regresión Logística — "Coeficientes ttheta en funcion del lmbda.png"
🔹 1. Estabilidad del Intercepto θ₀
Para el modelo de regresión logística, también se observó que el intercepto θ₀ se mantuvo estable.
Por ejemplo, para λ = 0, θ₀ ≈ 1.35, y para λ = 1000, θ₀ ≈ 1.31.
Al igual que en la regresión lineal, esto es esperable porque el intercepto no es penalizado por la regularización L2.

🔹 2. Encogimiento de los coeficientes θ₁ a θ₈
En este gráfico se muestran solo los primeros 8 coeficientes (de un total de 30 del dataset de cáncer de mama), por claridad visual.

Cuando λ es pequeño (a la izquierda del gráfico, valores de log(λ) cercanos a -3), los coeficientes θ₁ a θ₈ toman valores entre aproximadamente -0.4 y -0.1, es decir, la mayoría empiezan en valores negativos moderados.

A medida que λ aumenta, todos los coeficientes disminuyen su magnitud y tienden hacia cero, mostrando el clásico efecto de "shrinkage".
Algunos, como **θ₃ o θ



# Resumen: Diagnóstico y Control del Modelo – Overfitting, Underfitting y Regularización

El objetivo principal al entrenar un modelo de Machine Learning es que **generalice bien** a datos nuevos, no vistos durante el entrenamiento. Dos problemas comunes que impiden esto son el overfitting y el underfitting, ambos relacionados con el **compromiso sesgo-varianza**. La **regularización** es una técnica clave para controlar estos problemas, especialmente el overfitting.

## 1. Overfitting (Sobreajuste)

### ¿Qué es?
Ocurre cuando un modelo aprende *demasiado bien* los datos de entrenamiento, capturando no solo los patrones generales sino también el **ruido, errores o particularidades** de ese conjunto específico. Como resultado, **memoriza** en lugar de *entender*, llevando a un excelente rendimiento en los datos de entrenamiento pero un **bajo rendimiento en datos nuevos**.

### Características:
- Error de entrenamiento: Muy bajo.
- Error de validación/prueba: Alto.
- Implica **Alta Varianza**: el modelo es muy sensible a pequeñas fluctuaciones en los datos de entrenamiento.

### Causas Comunes:
- **Modelo demasiado complejo** en relación a la cantidad/simplicidad de los datos (e.g., un polinomio de alto grado para datos lineales).
- **Pocos datos de entrenamiento**: insuficientes para aprender patrones generalizables.
- **Ruido en los datos**: el modelo intenta ajustarse a estos errores.
- **Entrenamiento excesivo**: el modelo empieza a memorizar tras muchas iteraciones.

## 2. Underfitting (Subajuste)

### ¿Qué es?
Ocurre cuando un modelo es **demasiado simple** para captar la complejidad real de los datos. No aprende bien los patrones ni siquiera de los datos de entrenamiento.

### Características:
- Error de entrenamiento: Alto.
- Error de validación/prueba: Alto.
- Implica **Alto Sesgo**: el modelo hace suposiciones demasiado simplistas sobre los datos.

### Causas Comunes:
- **Modelo demasiado simple** (e.g., una línea recta para datos con forma curva).
- **Datos de entrada (features) poco informativos**.
- **Entrenamiento insuficiente**: el modelo no tuvo tiempo de aprender.

## 3. El Compromiso Sesgo-Varianza (Bias-Variance Tradeoff)

Este es un concepto fundamental que describe el equilibrio necesario entre dos fuentes de error:

-   **Sesgo (Bias):** Error debido a suposiciones incorrectas o demasiado simplistas en el modelo. Un alto sesgo causa underfitting.
-   **Varianza (Variance):** Error debido a la sensibilidad del modelo a las pequeñas fluctuaciones en los datos de entrenamiento. Una alta varianza causa overfitting.

El **error total** de un modelo se puede descomponer (teóricamente) como:
$$Error_{total} = Sesgo^2 + Varianza + Error\ Irreducible$$
Donde el **Error Irreducible** es el ruido inherente al problema que ningún modelo puede eliminar.

-   Reducir el sesgo (aumentando la complejidad del modelo) tiende a aumentar la varianza.
-   Reducir la varianza (simplificando el modelo o con más datos) tiende a aumentar el sesgo (si se simplifica demasiado).
-   **Objetivo:** Encontrar el "sweet spot" o punto óptimo donde el error total en datos no vistos es mínimo, logrando un buen equilibrio.

### Diagnóstico:
Se compara el error del modelo en el conjunto de entrenamiento y en el de validación/prueba:
-   **Alto Sesgo (Underfitting):** Error de entrenamiento alto, error de validación alto (similares).
-   **Alta Varianza (Overfitting):** Error de entrenamiento bajo, error de validación alto (gran diferencia).
-   **Buen Equilibrio:** Error de entrenamiento bajo, error de validación bajo (y cercano al de entrenamiento).

Las **curvas de aprendizaje** (error vs. tamaño del conjunto de entrenamiento o vs. complejidad del modelo) también ayudan a visualizar estos problemas.

## 4. Estrategias para Combatir Underfitting y Overfitting

| Problema                        | Estrategias para Combatir                                                                                                                                                                                                                                                           |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Underfitting** (Alto Sesgo)   | 1. **Aumentar la complejidad del modelo** (polinomios de mayor grado, redes neuronales más profundas/anchas).<br>2. **Ingeniería de características** (añadir features relevantes, crear interacciones).<br>3. **Entrenar más tiempo** (más épocas/iteraciones).<br>4. **Reducir la regularización** (si se está aplicando de forma excesiva). |
| **Overfitting** (Alta Varianza) | 1. **Regularización** (L1, L2 - ver abajo).<br>2. **Conseguir más datos de entrenamiento**.<br>3. **Reducir la complejidad del modelo** (modelos más simples, menos features).<br>4. **Selección de características / Reducción de dimensionalidad** (PCA).<br>5. **Early Stopping** (detener el entrenamiento cuando el error de validación empieza a subir).<br>6. **Métodos de Ensamblaje** (Bagging como Random Forests, Boosting como XGBoost).<br>7. **Aumento de Datos (Data Augmentation)**. |

## 5. Regularización L2 (Ridge Regression)

La regularización es una técnica para prevenir el overfitting al añadir un término de penalización a la función de coste del modelo. Esta penalización desalienta que los parámetros (coeficientes $\theta_j$) del modelo tomen valores demasiado grandes.

### Función de Coste con Regularización L2:
Se modifica la función de coste original (e.g., Error Cuadrático Medio para regresión lineal) añadiendo el término de regularización:
$$J(\theta) = \underbrace{\frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2}_{\text{Coste Original (MSE)}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2}_{\text{Término de Regularización L2}}$$
Donde:
-   $m$: número de ejemplos de entrenamiento.
-   $h_\theta(x^{(i)})$: predicción del modelo para el ejemplo $i$.
-   $y^{(i)}$: valor real para el ejemplo $i$.
-   $\theta_j$: parámetros (coeficientes) del modelo.
-   $\lambda$ (lambda): **parámetro de regularización**. Controla la magnitud de la penalización. Un $\lambda$ más alto implica una penalización mayor.
-   La suma del término de regularización $\sum_{j=1}^{n} \theta_j^2$ va desde $j=1$ hasta $n$. **Importante: $\theta_0$ (el término de intercepción o sesgo del modelo) no se penaliza.**

### ¿Por qué no se penaliza $\theta_0$?
$\theta_0$ representa el desplazamiento base del modelo y no está asociado a la complejidad de cómo se ajusta a las características de entrada. Penalizarlo no ayuda a controlar el overfitting y podría llevar a un ajuste subóptimo.

### Actualización del Gradiente con Regularización L2 (para Descenso de Gradiente):
La actualización para cada $\theta_j$ (donde $j \ge 1$) se modifica:
$$\theta_j := \theta_j - \alpha \left[ \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m} \theta_j \right]$$
Para $\theta_0$ (si $j=0$):
$$\theta_0 := \theta_0 - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_0^{(i)} \right]$$
(Asumiendo $x_0^{(i)} = 1$. El término $\frac{\lambda}{m} \theta_j$ no se aplica a $\theta_0$).
El efecto es que, en cada paso, $\theta_j$ (para $j \ge 1$) se reduce proporcionalmente a su valor actual, además del ajuste por el error ("shrinkage" o encogimiento).

### Efecto de $\lambda$:
-   **Si $\lambda = 0$:** No hay regularización. La función de coste es la original. Alto riesgo de overfitting si el modelo es complejo.
-   **Si $\lambda > 0$:** Los coeficientes $\theta_j$ (para $j \ge 1$) tienden a disminuir. Cuanto mayor es $\lambda$, más pequeños se vuelven los coeficientes, llevando a un modelo más simple.
-   **Si $\lambda$ es muy grande:** Puede llevar a underfitting, ya que los coeficientes pueden volverse demasiado pequeños, perdiendo información útil.
-   El valor de $\theta_0$ se mantiene relativamente estable ya que no es afectado por la penalización de $\lambda$.

## 6. Interpretación de Resultados Experimentales con Regularización

### Gráficos de Coeficientes ($\theta_j$) vs. $\lambda$:
-   Muestran que a medida que $\lambda$ aumenta, la magnitud de los coeficientes $\theta_j$ (para $j \ge 1$) disminuye, tendiendo hacia cero.
-   $\theta_0$ (intercepto) permanece relativamente constante.
-   Coeficientes asociados a características menos relevantes tienden a disminuir más rápidamente.

### Gráficos de Coste vs. Iteraciones (para diferentes $\lambda$):
-   Permiten verificar la convergencia del entrenamiento para cada valor de $\lambda$.
-   El coste de entrenamiento final con $\lambda > 0$ suele ser un poco más alto que con $\lambda = 0$, ya que el modelo no se ajusta tan agresivamente al ruido del entrenamiento.
-   Un $\lambda$ óptimo es aquel que minimiza el error en un conjunto de validación, no necesariamente el que da el menor coste de entrenamiento.

---

## 📌 ¿Qué hace `train_test_split` en regresión lineal?

Divide tus datos en:

* **Conjunto de entrenamiento (`X_train`, `y_train`)** → se usa para ajustar los coeficientes del modelo (las "θ").
* **Conjunto de prueba (`X_test`, `y_test`)** → se usa para evaluar cómo de bien generaliza el modelo a datos nuevos.

Esto evita que evalúes el modelo sobre los mismos datos con los que fue entrenado, lo que podría ocultar problemas como el **overfitting**.

---

## 📊 Métricas comunes en regresión lineal

En vez de precisión o matriz de confusión (que son para clasificación), usamos métricas que miden **error numérico**:

### 1. **Mean Squared Error (MSE)**

Promedio del cuadrado de los errores:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Más sensible a errores grandes.

---

### 2. **Mean Absolute Error (MAE)**

Promedio del valor absoluto de los errores:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Menos sensible a valores extremos.

---

### 3. **R² Score (Coeficiente de determinación)**

Indica qué tan bien se explica la variación de los datos con el modelo (0 a 1, donde 1 es perfecto).

$$
R^2 = 1 - \frac{SSE}{SST}
$$

---

## 💬 Entonces, en resumen:

> “En regresión lineal, `train_test_split` permite evaluar el modelo en un conjunto independiente usando métricas como MSE, MAE, RMSE y R², lo cual es esencial para validar su capacidad de generalización.”

---

## Implementación de `train_test_split` en regresión logística

Gracias a la función `train_test_split`, podemos evaluar el rendimiento real del modelo con métricas de clasificación confiables, al medirlo sobre un conjunto de prueba independiente.

Cuando entrenas un modelo, necesitas evaluar su desempeño en datos nuevos, no solo en los que ha visto (entrenado). Si evaluaras el modelo en los mismos datos de entrenamiento:

* Obtendrías una visión irrealmente optimista del rendimiento del modelo.
* Podrías estar midiendo **overfitting** (el modelo aprendió los datos de memoria).

`train_test_split` separa el conjunto original en:

* `X_train`, `y_train`: para entrenar el modelo.
* `X_test`, `y_test`: para evaluar el modelo.

🧠 **¿Qué permite esta división?**
Gracias a ella, puedes:

* Medir cómo generaliza el modelo a datos que no ha visto.
* Calcular métricas como:
  ✅ **Accuracy**
  📦 **Matriz de Confusión**
  🎯 **Precision**
  🔍 **Recall**
  ⚖️ **F1 Score**

---

## 🧮 1. Matriz de Confusión

Es una tabla que resume cómo de bien ha hecho las predicciones un modelo de clasificación, basándose en las etiquetas verdaderas y las predichas.

### 📦 Estructura (para clasificación binaria):

|                        | Predicho: 0 | Predicho: 1 |
| ---------------------- | ----------- | ----------- |
| **Real: 0** (Negativo) | TN (✔)      | FP (❌)      |
| **Real: 1** (Positivo) | FN (❌)      | TP (✔)      |

* **TP** (True Positives): casos positivos predichos correctamente.
* **TN** (True Negatives): casos negativos predichos correctamente.
* **FP** (False Positives): negativos clasificados erróneamente como positivos.
* **FN** (False Negatives): positivos clasificados erróneamente como negativos.

---

## 📊 2. Accuracy (Exactitud)

Es el porcentaje de predicciones correctas (positivas y negativas).

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

🟢 Útil cuando las clases están **balanceadas** (mismo número de positivos y negativos).

---

## 🎯 3. Precision (Precisión)

Mide cuántos de los casos que predijiste como positivos **realmente lo son**.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

✅ Útil cuando los **falsos positivos son costosos** (por ejemplo, diagnosticar erróneamente una enfermedad grave).

---

## 🔍 4. Recall (Sensibilidad o Tasa de Verdaderos Positivos)

Mide cuántos de los casos **positivos reales** has detectado correctamente.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

✅ Muy importante cuando los **falsos negativos son peligrosos** (por ejemplo, no detectar un cáncer cuando sí lo hay).

---

## ⚖️ 5. F1 Score

Es el **promedio armónico** de precisión y recall. Equilibra ambos.

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

✅ Útil cuando quieres un equilibrio entre precisión y recall, especialmente importante con clases desbalanceadas.

---

## 🧠 Ejemplo sencillo:

Supón que:

* Tienes 100 pacientes.
* 10 tienen cáncer (positivos), 90 no (negativos).
* Tu modelo detecta 8 de esos 10 correctamente, pero se equivoca en 5 casos sin cáncer (falsos positivos).

Entonces:

* TP = 8
* FN = 2
* FP = 5
* TN = 85

Ahora puedes calcular:

* Accuracy = (8 + 85) / 100 = 93%
* Precision = 8 / (8 + 5) = 61.5%
* Recall = 8 / (8 + 2) = 80%
* F1 Score ≈ 69.6%

---

### ✅ Tabla comparativa de modelos de regresión

| Modelo                                     | Dataset            | MSE (Test) | MAE (Test) | R² (Test)  | Hiperparámetros / Notas                   |
| ------------------------------------------ | ------------------ | ---------- | ---------- | ---------- | ----------------------------------------- |
| Regresión Lineal (Descenso Gradiente)      | California Housing | 0.5559     | 0.5332     | 0.5758     | α=0.1, Iter=2500, λ=0.01, Datos escalados |
| Regresión Lineal (Ecuación Normal)         | California Housing | 0.5559     | 0.5332     | 0.5758     | Datos escalados                           |
| SVR (scikit-learn, kernel='linear', C=0.1) | California Housing | 0.5795     | 0.5123     | 0.5578     | C=0.1, Datos escalados                    |
| SVR (scikit-learn, kernel='rbf', C=3, γ=1) | California Housing | **0.3127** | **0.3711** | **0.7614** | Mejor resultado. Datos escalados          |

---

### 🏆 Mejor modelo: **SVR con kernel RBF (C=3, γ=1)**

* Supera ampliamente a la regresión lineal en MSE, MAE y R².
* Presenta un muy buen ajuste sin indicios evidentes de overfitting.
* Requiere mayor tiempo de cómputo, justificado por la mejora en desempeño.

---

### Interpretación de Resultados - Modelos de Regresión

En este análisis sobre el dataset **California Housing**, comparamos cuatro enfoques para regresión:

* **Regresión Lineal (Descenso Gradiente y Ecuación Normal):**
  Ambos métodos ofrecen resultados prácticamente idénticos, con MSE ≈ 0.556, MAE ≈ 0.533 y un coeficiente de determinación \$R^2\$ ≈ 0.58. Esto refleja un rendimiento moderado, adecuado para una predicción lineal básica pero con espacio para mejoras.

* **SVR con Kernel Lineal:**
  Este modelo muestra un desempeño comparable, ligeramente inferior en \$R^2\$ (0.56) y errores similares (MSE ≈ 0.58, MAE ≈ 0.51). Indica que la inclusión del margen suave del SVM con kernel lineal no aporta una ventaja significativa frente a la regresión lineal tradicional.

* **SVR con Kernel RBF (Mejor Modelo):**
  El modelo con kernel RBF logra un desempeño notablemente superior: reduce el MSE a 0.313, el MAE a 0.371 y aumenta el \$R^2\$ a 0.76. Esto evidencia que captura relaciones no lineales presentes en los datos, mejorando considerablemente la precisión sin sobreajustar.

**Conclusión:**
El SVR con kernel RBF es el modelo más efectivo para este problema, demostrando la importancia de considerar modelos no lineales cuando la relación entre variables no es simple. El mayor costo computacional se justifica por la mejora significativa en la capacidad predictiva.

---



| Modelo                | Dataset       | Accuracy (Test) | Precision (Test) | Recall (Test) | F1-Score (Test) | Parámetros Clave                  |
| --------------------- | ------------- | --------------- | ---------------- | ------------- | --------------- | --------------------------------- |
| Regresión Logística   | Breast Cancer | 0.9649          | 0.9718           | 0.9718        | 0.9718          | alpha=0.1, iter=2500, lambda=0.01 |
| SVC (kernel='linear') | Breast Cancer | 0.9649          | 0.9589           | 0.9859        | 0.9722          | C=0.1                             |


---

## Interpretación de Resultados: Modelos de Clasificación en Breast Cancer

| Métrica   | Regresión Logística (Test) | SVC (Test) |
| --------- | -------------------------- | ---------- |
| Accuracy  | 96.49%                     | 96.49%     |
| Precision | 97.18%                     | 95.89%     |
| Recall    | 97.18%                     | 98.59%     |
| F1-Score  | 97.18%                     | 97.22%     |

---

### 1. Exactitud (Accuracy)

Ambos modelos obtienen una exactitud idéntica (96.49%), lo que significa que los dos clasifican correctamente la mayoría de las muestras del conjunto de prueba.

### 2. Precisión (Precision)

* **Regresión Logística** presenta una precisión ligeramente mayor (97.18%) que SVC (95.89%).
* Esto indica que cuando la regresión logística predice un caso positivo (por ejemplo, tumor maligno), lo hace con mayor confianza y menos falsos positivos.

### 3. Sensibilidad / Recall

* **SVC** tiene un recall mayor (98.59% vs 97.18%), lo que indica que detecta un porcentaje más alto de casos positivos reales.
* Esto es especialmente importante en diagnósticos médicos, donde es preferible minimizar falsos negativos.

### 4. F1-Score

* Ambos modelos tienen un F1-score muy alto y casi igual, indicando un excelente balance entre precisión y recall.
* SVC tiene una ligera ventaja con 97.22% frente al 97.18% de regresión logística, aunque la diferencia es mínima.

---
Claro, aquí tienes una interpretación detallada para la tabla de regresión, siguiendo el estilo de la interpretación que hice para clasificación:

---




### Conclusión general:

* **Ambos modelos funcionan excepcionalmente bien para el problema de clasificación del cáncer de mama.**
* Si priorizas detectar todos los casos positivos (menos falsos negativos), **SVC es la mejor opción** gracias a su mayor recall.
* Si te importa más evitar falsos positivos (mayor precisión), la **regresión logística es ligeramente mejor**.
* La decisión final puede depender del contexto clínico y del costo asociado a los errores de clasificación.

---



## Resumen de  Métricas de Evaluación en Clasificación Binaria: Precision, Recall y F1-Score

| Métrica   | Se enfoca en...                           | Qué quiere evitar                 |
| --------- | ----------------------------------------- | --------------------------------- |
| Precision | Predicciones positivas correctas          | Falsos positivos                  |
| Recall    | Positivos reales correctamente detectados | Falsos negativos                  |
| F1        | Balance entre Precision y Recall          | Cuando uno de los dos falla mucho |
