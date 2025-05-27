# Regresion Logisitica

con esta tarea vamos a comprender e implementar la Regresión Logística desde cero para la clasificación binaria, entendiendo sus componentes matemáticos (función sigmoide, hipótesis, función de coste de entropía cruzada), cómo optimizarla con Descenso de Gradiente, y ser capaz de aplicarla y analizarla en un dataset.

# 📘 Regresión Logística – Conceptos Clave

## 🔧 Funciones a Implementar desde Cero
1. Función Sigmoide g(z)
2. Función de Hipótesis h(X, θ) (utiliza la sigmoide)
3. Función de Coste J(X, y, θ) (Entropía Cruzada Binaria)
4. Descenso de Gradiente (adaptado para clasificación)
5. Función de Predicción (aplica umbral 0.5 para clasificar en 0 o 1)

## 🔁 Función Sigmoide

```math
g(z) = \frac{1}{1 + e^{-z}}
```

* Convierte cualquier número (positivo o negativo) en un valor entre **0 y 1**.
* Tiene forma de **S**, y sus salidas son utiles por que pueden interpretarse como **probabilidades**.
* Por ejemplo, `g(0) = 0.5`, y si `z` es muy grande, `g(z)` se acerca a 1; si es muy pequeño, se acerca a 0. 


---

## 🧠 Función de Activación Sigmoide en Regresión Logística

En la regresión logística, utilizamos la función sigmoide como función de activación para modelar probabilidades. Este proceso se puede describir en los siguientes pasos:

1. **Calcular la Entrada `z`**

   Se calcula como el producto escalar entre los parámetros y las características:

   $$
   z = \theta^T x
   $$

   > Este valor puede ser cualquier número real: positivo, negativo o cero.

2. **Aplicar la Función Sigmoide**

   La función sigmoide toma `z` como entrada y devuelve un valor entre 0 y 1:

   $$
   g(z) = \frac{1}{1 + e^{-z}}
   $$

3. **Interpretar la Salida como Probabilidad**

   La salida de la función sigmoide se interpreta como la **probabilidad estimada** de que la observación pertenezca a la clase positiva (clase 1):

   $$
   h_\theta(x) = g(\theta^T x) \approx P(y = 1 \mid x; \theta)
   $$

---


## 🧠 Hipótesis del Modelo

```math
h_\theta(x) = g(\theta^T x)
```

* Esta fórmula se encarga de **hacer predicciones**.
* Multiplicamos los datos de entrada por los parámetros (`θ`) y aplicamos la función sigmoide.
* El resultado es una **probabilidad** de que la salida sea `1`.
  Ejemplo: si `hθ(x) = 0.8`, el modelo predice un **80% de probabilidad** de que `y = 1`.

---

## 💰 Función de Coste (Binary Cross-Entropy)

```math
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
```

* Nos dice **qué tan mal está funcionando el modelo**.
* Penaliza más fuerte cuando el modelo está seguro y se equivoca.
* Evitamos usar el **Error Cuadrático Medio (MSE)**, porque no se adapta bien a clasificación.

---

## 📉 Descenso de Gradiente

* Es el método que usamos para **encontrar los mejores parámetros** (`θ`).
* Calcula **qué tan lejos estamos** del mínimo de la función de coste.
* Da pasos pequeños en la dirección correcta para **mejorar el modelo**.
* Aunque usamos la sigmoide, la fórmula del gradiente se mantiene **muy parecida** a la de regresión lineal, lo cual simplifica la implementación.

---

## 🧭 Límite de Decisión

* Es la **frontera que separa las dos clases** (por ejemplo, spam vs no spam).
* Si `hθ(x) ≥ 0.5`, clasificamos como **1**; si es menor, como **0**.
* En un espacio 2D, es una **línea recta**; en espacios con más dimensiones, es un **hiperplano**.

---

## ⚙️ Consideraciones Prácticas

* 🔧 **Umbral ajustable**: El valor de 0.5 puede cambiarse según el problema (por ejemplo, para priorizar sensibilidad en medicina).
* 🧯 **Regularización**: Podemos añadir términos (L1 o L2) a la función de coste para **evitar el sobreajuste** (*overfitting*).
* 🎯 **Clasificación multiclase**: Se puede extender usando **Softmax** o estrategias **One-vs-Rest**.





### Comparación con Otros Métodos

| Característica              | Regresión Logística          | LDA / QDA                                                         |
| --------------------------- | ---------------------------- | ----------------------------------------------------------------- |
| Supuestos sobre los datos   | No hace suposiciones fuertes | Asume que los datos tienen forma de campana (distribución normal) |
| Frontera de decisión        | Recta (lineal)               | Recta o curva (cuadrática)                                        |
| Cómo calcula probabilidades | Directamente con la sigmoide | Basado en fórmulas estadísticas más complejas                     |
 
## Pasos a seguir en la interacion de GD: 



### 🔄 Ciclo del Descenso de Gradiente

En cada iteración del algoritmo de optimización se repiten los siguientes pasos:

1. **Calcular la Hipótesis**
   Se calcula $z = X\theta$ (o $\theta^T X$ si $X$ es una sola muestra), y luego se aplica la función sigmoide:

   $$
   h_\theta(X) = g(z)
   $$

   Esto nos da las probabilidades estimadas para cada muestra.

2. **Calcular el Error**
   Se obtiene la diferencia entre las predicciones y los valores reales:

   $$
   \text{errores} = h_\theta(X) - y
   $$

3. **Calcular el Gradiente**
   Se calcula usando la fórmula vectorizada:

   $$
   \nabla J(\theta) = \frac{1}{m} X^T \cdot \text{errores}
   $$

4. **Actualizar los Parámetros $\theta$**
   Se ajustan los parámetros para minimizar la función de coste:

   $$
   \theta := \theta - \alpha \cdot \nabla J(\theta)
   $$



# 📘 Pasos del Algoritmo de Regresión Logística (`load_breast_cancer`)


## 🔢 Función sigmoide

Para empezar, definimos la **función sigmoide**, que convierte cualquier número en un valor entre 0 y 1. Esto es muy útil para interpretar resultados como **probabilidades**.

Hice una lista de valores $z$ y apliqué la sigmoide para ver los resultados. Algunos puntos clave que me tengo que acordar:

* Si $z = 0$, la sigmoide da $0.5$.
* Si $z$ es muy grande, se acerca a $1.0$.
* Si $z$ es muy negativo, se acerca a $0.0$.

Fórmula:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

---

## 📈 Función de hipótesis $h_\theta(x)$

Ya habíamos visto esta función antes, pero ahora la usamos junto con la sigmoide para obtener una **matriz de probabilidades**.

La fórmula general es:

$$
h_\theta(x) = g(\theta^T x)
$$

---

## 💰 Función de coste (entropía cruzada binaria)

Para medir qué tan bien está aprendiendo el modelo, usamos la **entropía cruzada**, que castiga más cuando el modelo se equivoca con confianza.

$$
J(\theta) = -\frac{1}{m} \sum \left[ y \log(h_\theta(x)) + (1 - y) \log(1 - h_\theta(x)) \right]
$$

Agregamos un pequeño valor $\varepsilon$ para evitar errores como dividir entre cero o calcular $\log(0)$. Ese valor es tan pequeño que no afecta el resultado final, pero ayuda a evitar problemas numéricos.

---

## 📉 Descenso de Gradiente (GD)

Esta función sirve para ajustar los parámetros $\theta$ y minimizar el error.

Primero calculamos el **gradiente**:

$$
\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)
$$

Y luego actualizamos los parámetros con:

$$
\theta := \theta - \alpha \cdot \nabla J(\theta)
$$

Probé con varios valores de $\alpha$ (la tasa de aprendizaje) y vi cuál hacía que la curva de pérdida bajara más rápido y luego se estabilizara. Ese fue el mejor.

---

## 🚀 Empieza el entrenamiento

Cargué los datos desde `sklearn.datasets.load_breast_cancer` y seguí estos pasos:

* Escalé todas las características para que el modelo aprenda mejor.
* Agregué una columna de unos al dataset para que el modelo aprenda también el **intercepto** $\theta_0$, lo que le da más libertad para ajustar la curva.
* Usé un valor de $\alpha$ que funcionara bien y un número razonable de iteraciones (basado en cómo se ve la curva de pérdida).

Todo esto me permitió entrenar el modelo y practicar la función `predict`.

### Visualización del entrenamiento

Comparé la evolución del error y el efecto de distintos valores de $\alpha$:

![Curva de pérdida vs iteraciones](Regresion_Logisitica/Figure_2.png)

![Comparación de tasas de aprendizaje](Regresion_Logisitica/Figure_1.png)

---

## ✅ Función predecir

Con la hipótesis $h_\theta(x)$, calculamos probabilidades y luego usamos un **umbral** de 0.5 para convertir eso en una decisión:

* Si $h_\theta(x) \geq 0.5$ → predice clase **1**.
* Si $h_\theta(x) < 0.5$ → predice clase **0**.

Esto nos da una predicción binaria clara.

---

## 🎯 Accuracy del modelo

Para saber qué tan bien aprendió el modelo, calculé el **accuracy**, que es el porcentaje de predicciones correctas.

En este caso, obtuve:

$$
\text{Accuracy} = 97.01\%
$$

También probé una forma alternativa de calcularlo con menos pasos, solo para recordar que se puede hacer lo mismo de distintas maneras.

---




## 🤔 ¿Por qué usamos la entropía cruzada binaria? (BCE vs MSE)

Usamos la **entropía cruzada binaria** (BCE) en regresión logística porque se ajusta muy bien al funcionamiento de la **función sigmoide**, que nos da una probabilidad entre 0 y 1. En problemas de clasificación binaria, como este, donde solo existen dos posibles clases (0 o 1), la BCE se adapta perfectamente, ya que estamos modelando **probabilidades**.

La BCE tiene la ventaja de penalizar más fuertemente cuando el modelo se equivoca, especialmente cuando está muy seguro de su predicción y se equivoca. Esto ayuda a que el modelo aprenda más rápido y mejor. En cambio, el **error cuadrático medio** (MSE) no penaliza de la misma manera y no se comporta tan bien cuando estamos trabajando con **probabilidades**, ya que no mide la calidad de las predicciones de manera tan eficiente como la BCE.

En resumen, la BCE es más adecuada para este tipo de problemas, porque no solo mide la diferencia entre las predicciones y las clases reales, sino que también penaliza más fuertemente los errores cuando el modelo está muy confiado y equivocado.

## Cuadro comparativo entre BCE y MSE

| **Característica**          | **Entropía Cruzada Binaria (BCE)**                                                              | **Error Cuadrático Medio (MSE)**                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Uso principal**           | Problemas de clasificación binaria (0 o 1)                                                      | Problemas de regresión (predicciones continuas)                                                 |
| **Salida del modelo**       | Probabilidades (0 a 1)                                                                          | Cualquier valor real (números continuos)                                                        |
| **Fórmula**                 | $-y \log(h) - (1 - y) \log(1 - h)$                                                              | $\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$                                                  |
| **Qué mide**                | Cuánta "sorpresa" hay entre la predicción y el valor real                                       | La diferencia entre la predicción y el valor real                                               |
| **Penalización de errores** | Penaliza fuertemente los errores de alta certeza (predicciones incorrectas con mucha confianza) | Penaliza más los errores grandes, pero no lo suficiente para problemas de clasificación binaria |
| **Ventajas**                | Se ajusta a problemas binarios, es estadísticamente coherente, y ayuda al aprendizaje eficiente | Es simple y fácil de calcular, pero no es adecuado para probabilidades                          |
| **Desventajas**             | No es adecuado para regresión, y puede ser sensible a valores muy extremos                      | No es ideal para clasificación binaria, ya que no maneja bien las probabilidades                |

---

## **Resumen fácil**:

* **BCE** es la mejor opción cuando estás trabajando con **probabilidades y clasificación binaria** (0 o 1).
* **MSE** es mejor para **predicciones continuas** (por ejemplo, en regresión), pero no se adapta bien a los problemas de probabilidad.
---

## 🤔 ¿Qué significa que una función de coste sea "no convexa"?

Si usamos **MSE** (Error Cuadrático Medio) en lugar de **BCE** (Entropía Cruzada Binaria), la función de coste puede volverse **no convexa**. Esto sucede porque el **MSE** no se ajusta tan bien a la función sigmoide, y puede generar una función de coste con **múltiples mínimos locales**. Esto dificulta encontrar el mejor valor para los parámetros del modelo.

El **descenso de gradiente** es un algoritmo que busca minimizar la función de coste, es decir, encuentra el mínimo de la función para que el modelo sea lo más preciso posible.

### ¿Qué significa que una función de coste sea "convexa"?

Cuando una función es **convexa**, tiene una forma de **cuenco** o "U". En este caso, la función solo tiene un **mínimo global** (el fondo del cuenco), y no hay otros **picos** o "colinas" que distraigan el proceso de búsqueda del mínimo.

Cuando la función es convexa, **el descenso de gradiente** siempre llevará al **mínimo global**. No importa desde qué punto empieces, siempre irás hacia el punto más bajo de la función.

### ¿Qué pasa si la función de coste no es convexa?

Si la función **no es convexa** (como sucede con el **MSE** en regresión logística), entonces la función de coste puede tener **múltiples mínimos locales** (como montañas y valles). El **descenso de gradiente** podría quedarse atrapado en un **mínimo local** y no encontrar el mejor valor (mínimo global).

---

### 🏞️ Ejemplo Visual

**Función Convexa (como BCE):**

Imagina que estás en un campo con una sola gran colina que desciende en todas direcciones (función convexa). No importa en qué punto empieces, siempre **descenderás** hacia el punto más bajo, que es el **mínimo global**.

**Función No Convexa (como MSE):**

Ahora imagina un campo con varias montañas y valles (función no convexa). Si te encuentras en un valle pequeño (mínimo local), podrías pensar que has encontrado el mejor lugar. Sin embargo, hay un valle más profundo en otro lugar, el **mínimo global**. Si el descenso de gradiente se queda atrapado en el primer valle, no podrá encontrar el mínimo global.

---

### 📝 Resumen en palabras sencillas:

El **descenso de gradiente** busca el punto más bajo (mínimo) de una **función de coste** ajustando los parámetros del modelo.

* Si la función es **convexa** (como la BCE), el descenso de gradiente siempre encontrará el **mínimo global**.
* Si la función es **no convexa** (como con MSE en regresión logística), el descenso de gradiente podría quedarse atrapado en **mínimos locales** y no encontrar el mejor mínimo global.

---




## 📌 ¿Por la que la Entropía Cruzada Binaria (BCE) es "la elegida" para modelos como la Regresión Logística.  (Conexión con MLE)

Una de las razones más importantes para usar la **Entropía Cruzada Binaria (BCE)** en regresión logística es que **está directamente relacionada con un principio estadístico muy fuerte llamado *Estimación de Máxima Verosimilitud (MLE)*.**



## 📚 Fundamento Teórico de la Entropía Cruzada Binaria en Regresión Logística

### ❓ ¿Por qué usamos Entropía Cruzada Binaria (Binary Cross-Entropy, BCE) como función de coste?

Además de ser útil en la práctica, la Entropía Cruzada Binaria tiene una **base teórica muy sólida** que la hace especialmente adecuada para la regresión logística.

---

### 🎯 Objetivo del modelo

La regresión logística busca predecir la **probabilidad** de que algo ocurra (por ejemplo, que un correo sea spam o no). Por eso usamos la **función sigmoide**, que transforma cualquier número en un valor entre 0 y 1, justo lo que necesitamos para representar probabilidades:

$$
h_θ(x) = \frac{1}{1 + e^{-θ^T x}}
$$

---

### 🧪 ¿Qué hace la Entropía Cruzada?

La función de coste BCE mide **qué tan buena es la probabilidad** que el modelo asigna a los resultados reales. Penaliza más fuertemente cuando el modelo está muy seguro pero se equivoca (por ejemplo, predice 0.99 pero el resultado real es 0).

La fórmula general de la **Entropía Cruzada Binaria** es:

$$
J(θ) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_θ(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_θ(x^{(i)})) \right]
$$

---

### 📐 Base teórica: Estimación por Máxima Verosimilitud (MLE)

Matemáticamente, **minimizar la BCE es lo mismo que buscar los valores de $θ$ que hacen más probable que los datos reales ocurran**. Esto se llama:

> **Estimación por Máxima Verosimilitud (MLE)**

En otras palabras:

* El modelo no solo trata de acertar.
* Intenta asignar **altas probabilidades a los eventos correctos**.
* Esto lo alinea naturalmente con la función sigmoide, que **ya devuelve probabilidades**.

---

### 🧩 En resumen:

* ✅ La BCE es coherente con el objetivo probabilístico de la regresión logística.
* 🧠 Tiene un fuerte respaldo matemático (MLE).
* 📉 Castiga con más fuerza cuando el modelo se equivoca con seguridad.
* 🤝 Hace que entrenar el modelo sea una cuestión de **ajustar las probabilidades, no solo de acertar o fallar**.



---

### 🧠 ¿Qué busca la MLE?

Queremos encontrar los parámetros del modelo, representados como **θ**, que hagan que los **datos de entrenamiento que ya observamos** (las verdaderas etiquetas `y`) sean **lo más probables posible** según el modelo. Es decir, que nuestro modelo diga:

> "¡Sí, con estos parámetros, es muy probable que haya visto exactamente estos datos!"

---


### 📊 ¿Cómo se calcula esa probabilidad?

Para una sola observación $(x^{(i)}, y^{(i)})$, la probabilidad según el modelo es:

* Si $y^{(i)} = 1$, entonces la probabilidad es $h_{\theta}(x^{(i)})$
* Si $y^{(i)} = 0$, entonces la probabilidad es $1 - h_{\theta}(x^{(i)})$

Todo esto se puede escribir así:

$$
P(y^{(i)}|x^{(i)};\theta) = (h_{\theta}(x^{(i)}))^{y^{(i)}} (1 - h_{\theta}(x^{(i)}))^{1 - y^{(i)}}
$$

> *Compruébalo tú mismo: si y = 1, queda solo hθ(x); si y = 0, queda 1 − hθ(x)*.

---

### 📦 Verosimilitud total (Likelihood)

Ya que asumimos que las observaciones son independientes, multiplicamos todas las probabilidades:

$$
L(\theta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)};\theta)
$$

---

### 📈 Log-Verosimilitud

Trabajar con productos es incómodo, así que tomamos el logaritmo (para convertir productos en sumas):

$$
\log L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right]
$$

---

### 💡 ¡Sorpresa! ¡Esta fórmula ya la conoces!

La función de coste de **Entropía Cruzada Binaria (BCE)** es exactamente la **negación** del promedio de esa log-verosimilitud:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right]
$$

---

### 🧠 En resumen:

* Maximizar la log-verosimilitud (objetivo de MLE) es **equivalente a minimizar la función BCE**.
* El signo negativo y el factor $\frac{1}{m}$ solo convierten el problema de maximizar en uno de **minimización promedio**, que es justo lo que usa el **descenso de gradiente**.
* Esto le da a la BCE una base teórica muy sólida, **además de que es convexa** (lo cual es genial para evitar mínimos locales).

---

### 📌 Relación entre la teoría y la implementación

#### 1. **Función de hipótesis `hθ(x)`**

```python
def calcular_hipotesis(X, theta):
    Z_vector = X @ theta
    Z_vector_prob = sigmoid(Z_vector)
    return Z_vector_prob
```

Esta función calcula **hθ(x)**, que representa la **probabilidad** de que una muestra pertenezca a la clase 1. Esto es precisamente lo que necesita el MLE: una función que dé **probabilidades condicionales P(y|x;θ)**.

---

#### 2. **Función de coste `calcular_coste`**

```python
def calcular_coste(X, y, theta):
    ...
    coste = - (1 / m) * sum_total
    return coste
```

Esta es **exactamente** la fórmula de la **Entropía Cruzada Binaria (BCE)**, que como dijimos en la teoría, es la **forma negativa y promedio de la log-verosimilitud**:

* **MLE:** maximiza la log-verosimilitud.
* **BCE:** minimiza el coste (−log-verosimilitud promedio).

Por eso, esta función de coste **implementa MLE en forma negativa**, adaptada para optimización vía descenso de gradiente.

---

#### 3. **Descenso de Gradiente**

```python
theta = theta - alpha * gradiente
```

### 🧠 En resumen 

### Justificación estadística de la función de coste

Una razón fundamental para utilizar la **Entropía Cruzada Binaria** en regresión logística es su sólida base teórica en la **Estimación de Máxima Verosimilitud (MLE)**. En este modelo, queremos encontrar los parámetros θ que **maximicen la probabilidad de haber observado las etiquetas reales del entrenamiento**, dado nuestro modelo. Esto se logra **maximizando la log-verosimilitud**, la cual, al tomar su forma negativa y promedio, **se convierte en la función de coste que usamos: la BCE**.

Por tanto, el proceso de entrenamiento (con `calcular_coste` y `descenso_gradiente`) **no solo busca minimizar un error arbitrario, sino que está directamente fundamentado en probabilidad y estadística**: está **maximizando la verosimilitud de los datos observados**.

---

# ADELANTO INVESTIGACION PARA SIGUIENTE TAREA:

---

## 🧠 ¿Es la exactitud siempre la mejor métrica?

No. La **exactitud (accuracy)** solo mide el porcentaje de predicciones correctas. Pero en casos de **clases desbalanceadas**, puede dar una **falsa sensación de buen rendimiento**.

### 📌 Ejemplo clásico:

Supón que estamos diseñando un test para una **enfermedad rara** que afecta al 1% de la población.
De 1,000 personas, solo 10 la tienen.

Un modelo que **siempre predice "no tiene la enfermedad"** acertará en 990 casos.

* Exactitud = (990 aciertos) / 1000 = **99%**

¡Parece genial! Pero…

* No detectó **ni un solo caso verdadero**.
* **Recall = 0%**

Esto lo vuelve **inútil** para el propósito real: **detectar la enfermedad**.

---

## 🧩 Matriz de Confusión: ¿Qué significa cada caso?

Cuando entrenas un modelo para clasificar entre dos opciones (por ejemplo, **"enfermo"** o **"no enfermo"**), hay cuatro formas posibles en las que tu predicción puede coincidir (o no) con la realidad:

| Nombre 📌                     | Realidad 🧠 | Predicción 🤖       | ¿Qué pasó?                                                                                        |
| ----------------------------- | ----------- | ------------------- | ------------------------------------------------------------------------------------------------- |
| ✅ **Verdadero Positivo (TP)** | 1 (Enfermo) | 1 (Predijo enfermo) | El paciente **tenía la enfermedad** y el modelo **lo detectó correctamente**. Perfecto.           |
| ✅ **Verdadero Negativo (TN)** | 0 (Sano)    | 0 (Predijo sano)    | El paciente **no tenía la enfermedad** y el modelo **también dijo que no**. Muy bien.             |
| ⚠️ **Falso Positivo (FP)**    | 0 (Sano)    | 1 (Predijo enfermo) | El paciente **estaba sano**, pero el modelo **dijo que estaba enfermo**. Una **falsa alarma**.    |
| ❌ **Falso Negativo (FN)**     | 1 (Enfermo) | 0 (Predijo sano)    | El paciente **sí tenía la enfermedad**, pero el modelo **no la detectó**. El error **más grave**. |

---

### 🧠 ¿Por qué son importantes?

* **TP y TN** son los **aciertos** del modelo.
* **FP y FN** son los **errores**.
* A partir de ellos, se calculan métricas como **precisión**, **recall** y **F1-score**, que permiten entender mejor cómo se comporta el modelo en **situaciones críticas**.

---

¿Quieres que agregue una visualización estilo matriz con estos valores colocados en una tabla tipo cuadrícula (como un diagrama)?


## 📌 Métricas clave

### 🎯 Precisión (Precision)

> ¿De los que dije que eran positivos, cuántos lo eran realmente?

**Fórmula:**
**Precisión = TP / (TP + FP)**

**Importante cuando:** El coste de un **falso positivo** es alto.
**Ejemplos:**

* Clasificación de spam
* Recomendaciones de productos
* Sistema judicial (condenar a un inocente)

---

### 🔍 Recall (Sensibilidad, Exhaustividad)

> ¿De todos los que realmente eran positivos, cuántos detecté?

**Fórmula:**
**Recall = TP / (TP + FN)**

**Importante cuando:** El coste de un **falso negativo** es alto.
**Ejemplos:**

* Detección de enfermedades graves
* Fraude bancario
* Alerta temprana de incendios o catástrofes

---

### ⚖️ F1-Score (Balance entre precisión y recall)

> ¿Cómo consigo un equilibrio justo entre precisión y recall?

**Fórmula:**
**F1 = 2 \* (Precision \* Recall) / (Precision + Recall)**

* Es la **media armónica**: si una métrica es baja, el F1 también será bajo.
* Útil con **clases desbalanceadas**, o cuando es importante tener un **buen balance**.

---


Siguiendo con el ejemplo de la **enfermedad rara** (donde el 1% tiene la enfermedad y el 99% no):

Imagina que tenemos un modelo que **siempre predice "no tiene la enfermedad"**:

| **Resultado**               | **Realidad** | **Predicción** | **Cantidad** |
| --------------------------- | ------------ | -------------- | ------------ |
| **Verdadero Positivo (TP)** | 1            | 1              | 0            |
| **Falso Positivo (FP)**     | 0            | 1              | 0            |
| **Falso Negativo (FN)**     | 1            | 0              | 10           |
| **Verdadero Negativo (TN)** | 0            | 0              | 990          |

### **Accuracy**:

La **Accuracy** se calcula como:

**Accuracy** = (TP + TN) / Total = (0 + 990) / 1000 = **99%**
¡Una **Accuracy** del 99%, que parece excelente!

---

Sin embargo, si nos fijamos en **Recall** para la clase **"tiene la enfermedad"**, vemos lo siguiente:

### **Recall (Sensibilidad)**:

**Recall** = TP / (TP + FN) = 0 / (0 + 10) = **0%**
Esto significa que el modelo **no detecta ninguna persona enferma**, lo cual hace que **no sea útil para el diagnóstico** de la enfermedad.

---
## ✅ Conclusión

* Usa **Accuracy** solo si las clases están balanceadas.
* Usa **Precisión** si **falsos positivos** son costosos.
* Usa **Recall** si **falsos negativos** son peligrosos.
* Usa **F1-Score** cuando **ambos errores son críticos** o cuando hay **desequilibrio de clases**.
---


### ✅ **¿Cómo resumir la utilidad de Precisión, Recall y F1-Score?**

* **Precisión** te dice:

  > “¿Cuántos de los que el modelo **dijo que eran positivos**, **realmente lo eran**?”
  > Es útil cuando **no quieres dar falsas alarmas** (falsos positivos).
  > Ejemplo: Un filtro de spam — mejor no meter correos importantes en la carpeta de spam.

* **Recall** te dice:

  > “¿Cuántos de los que **realmente eran positivos**, **logramos detectar**?”
  > Es útil cuando **no quieres dejar pasar casos importantes** (falsos negativos).
  > Ejemplo: Diagnóstico de una enfermedad — mejor detectar todos los casos posibles, aunque te equivoques con algunos sanos.

* **F1-Score**:

  > Es una media entre precisión y recall.
  > Es útil cuando hay **desbalance de clases** o cuando **necesitas un equilibrio** entre no dar falsas alarmas y no dejar pasar casos.
  > Ejemplo: Detección de fraude — necesitas capturar la mayoría de fraudes (recall), pero también evitar acusar a gente inocente (precisión).

---

### 🧠 **¿Por qué el F1-Score intenta balancearlas?**

Porque en muchos problemas **no basta con solo precisión o solo recall**. Si una es muy alta y la otra muy baja, el modelo puede estar fallando en algo importante.
**F1 te obliga a que ambas sean razonablemente buenas.**

---

## ✂️ Implementación de `train_test_split` en Regresión Logística

Al usar `train_test_split` de `sklearn.model_selection`, dividimos el conjunto de datos original en dos subconjuntos:

* **X\_train, y\_train**: usados para entrenar el modelo.
* **X\_test, y\_test**: usados exclusivamente para evaluarlo.

Esta división permite medir el rendimiento real del modelo sobre datos **no vistos**, lo cual es crucial para evitar una evaluación sesgada. Evaluar el modelo sobre los mismos datos de entrenamiento puede resultar en:

* Una estimación **excesivamente optimista** de su rendimiento.
* Riesgo de **overfitting**, donde el modelo memoriza los datos en lugar de aprender patrones generalizables.

---

## 🎯 ¿Qué nos permite esta división?

Gracias a `train_test_split`, podemos:

✅ Evaluar **cómo generaliza** el modelo a nuevos datos.
📊 Calcular métricas de clasificación clave, como:

* **Accuracy** (exactitud)
* **Matriz de Confusión**
* **Precision**
* **Recall**
* **F1 Score**

