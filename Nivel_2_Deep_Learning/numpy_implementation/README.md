# 🧠 Red Neuronal con NumPy: 

> “Entender una red neuronal es descomponer la magia en matemáticas y funciones simples.”

---
  ### Resumen Clave 📝
  Una red neuronal artificial transforma entradas mediante combinaciones lineales y funciones no lineales llamadas activaciones para aprender patrones complejos. La propagación hacia adelante calcula salidas, mientras que la retropropagación ajusta los parámetros usando derivadas y la regla de la cadena.

  ### Analogía para Entenderlo Mejor 💡
  Imagina una red neuronal como una fábrica donde cada máquina (neurona) recibe materiales (entradas), los procesa con una receta (función afin), luego aplica un toque especial (función de activación) para crear productos intermedios, que luego se ensamblan en la siguiente línea (capa). La retropropagación es como una inspección de calidad que corrige el proceso paso a paso para mejorar el producto final.


---

## 📖 Explicación Completa y Sencilla

### 🧩 Neurona Artificial

Una neurona artificial recibe varias entradas, realiza un cálculo y produce un valor de activación. Este cálculo tiene dos pasos fundamentales:

1. **Transformación afín:**  
   Se multiplica la entrada $x$ por una matriz de pesos $W$, y se suma un vector de sesgos $b$ (o $c$). Matemáticamente:  
   $$
   z = W^\top x + b
    $$
> 🔧 ¿Qué es un peso en una red neuronal?
Un peso es un número que determina la importancia de una conexión entre dos neuronas. En una red neuronal, cada conexión entre una neurona de una capa y una neurona de la siguiente tiene un número asociado: ese número es el peso. 

2. **Función de activación no lineal:**  
   Sobre $z$ se aplica una función no lineal $g(z)$ llamada **función de activación**, dando la salida:  
   $$
   h = g(z) = g(W^\top x + b)
   $$

> ⚠️ **Importante:** La no linealidad es fundamental. Si la función de activación fuera lineal, toda la red sería equivalente a una función lineal, limitando mucho su capacidad para aprender patrones complejos.

---

### 🔑 Funciones de Activación Comunes

- **Sigmoide Logística:**  
  $$
  g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}
  $$  
  Se satura cuando $z$ es muy positivo o negativo, dificultando el aprendizaje por gradientes. No es recomendada para capas ocultas.

- **Tangente Hiperbólica (tanh):**  
  $$
  g(z) = \tanh(z)
  $$  
  Similar a la sigmoide pero centrada en 0 ($\tanh(0) = 0$), lo que facilita el entrenamiento.

- **Unidad Lineal Rectificada (ReLU):**  
  $$
  g(z) = \max(0, z)
  $$  
  Muy popular por su simplicidad y eficiencia. La derivada es constante (1) cuando la unidad está activa, ayudando a un aprendizaje más efectivo.

---

### 🏗 Arquitectura de un MLP (Perceptrón Multicapa)

- **Capas:**  
  Las redes se organizan en grupos llamados capas, que se conectan secuencialmente.  
  - Capa de entrada: recibe la entrada $x$.  
  - Capas ocultas: procesan la información internamente, sin salidas explícitas de entrenamiento.  
  - Capa de salida: produce el resultado final.

- **Dimensiones de pesos y sesgos:**  
  Para una capa con $n$ entradas y $p$ salidas, la matriz de pesos $W$ tiene dimensiones $p \times n$ y el vector de sesgos $b$ tiene dimensión $p$.

---

### ▶️ Forward Propagation (Propagación hacia adelante)

El proceso que calcula la salida a partir de una entrada $x$:

1. Inicializamos la activación de la capa 0 con la entrada:  
   $$
   h^{(0)} = x
   $$

2. Para cada capa $k = 1, \ldots, L$, calculamos:  
   - Activación lineal:  
     $$
     a^{(k)} = b^{(k)} + W^{(k)} h^{(k-1)}
     $$  
   - Salida con función de activación:  
     $$
     h^{(k)} = f(a^{(k)})
     $$

Finalmente, $h^{(L)}$ es la salida predicha $\hat{y}$.

---

### 🎯 Función de Coste

- **Entropía cruzada:**  
  Utilizada para clasificación, mide la diferencia entre la distribución real y la predicha.  
  Para clasificación multiclase con $n$ clases y salida softmax:  
  $$
  \text{Coste} = - \sum_{i=1}^n y_i \log \hat{y}_i
  $$

- **Importancia:**  
  Cambiar de error cuadrático a entropía cruzada fue un avance clave para mejorar el aprendizaje en redes con salidas sigmoides y softmax.

---

### 🔄 Backpropagation (Retropropagación)

- **Definición:**  
  Algoritmo para calcular el gradiente de la función de coste con respecto a cada parámetro $\theta$, usando la regla de la cadena.

- **Regla de la cadena (simplificada):**  
  Para funciones compuestas $z = f(g(x))$, la derivada es:  
  $$
  \frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}
  $$

- **Para vectores:**  
  Si $\mathbf{y} = g(\mathbf{x})$ y $z = f(\mathbf{y})$,  
  $$
  \nabla_{\mathbf{x}} z = \left( \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \right)^\top \nabla_{\mathbf{y}} z
  $$  
  donde $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ es la matriz Jacobiana de $g$.

- **Funcionamiento:**  
  Comienza con el gradiente de la salida (1) y viaja hacia atrás en la red, multiplicando por las derivadas parciales en cada paso y sumando gradientes cuando hay múltiples caminos.

---

## ✨ Puntos Finales

* Las redes neuronales combinan transformaciones lineales y no lineales para modelar funciones complejas.  
* Las funciones de activación no lineales son esenciales para que la red aprenda patrones útiles.  
* La propagación hacia adelante calcula la salida, mientras que la retropropagación calcula gradientes para optimizar la red.  
* Entender estos conceptos es clave para construir y entrenar redes neuronales con herramientas como NumPy.

---


# 🧠 Analógica Sencilla para Entender Capas y Neuronas en Redes Neuronales

> "Visualizar una red neuronal como un equipo de analistas hace que su funcionamiento sea mucho más claro."

---  
  ### Resumen Clave 📝
  Una capa en una red neuronal es como un equipo de analistas que trabajan simultáneamente con la misma información. Cada neurona recibe toda la entrada de la capa anterior y produce su propio resultado. La capa de entrada es solo la recepción de datos, no realiza cálculo. La capa de salida da la decisión final, por ejemplo, la probabilidad de pertenencia a una clase.

  ### Analogía para Entenderlo Mejor 💡
  Imagina que una capa es una fila de analistas en una oficina. El informe con los datos de entrada es distribuido a todos ellos, y cada analista (neurona) genera una conclusión. Así, una capa es un grupo de neuronas trabajando en paralelo, todas viendo el mismo "informe" pero aportando diferentes perspectivas.



---

## 📖 Explicación Completa y Sencilla

### 1. Capas y Neuronas: La Capa de Entrada

- **La Entrada:**  
  Imagina que recibes un informe con varias páginas; cada página es un dato de entrada.  
- **La Capa de Entrada:**  
  No es realmente una capa que hace cálculos, sino un "buzón de entrada" que distribuye la información. Por ejemplo, si el informe tiene 2 páginas, el buzón tendrá espacio para 2 datos.  
- **Primera Capa Oculta:**  
  Esta es la primera fila de analistas que toman todo el informe (todos los datos de entrada) y producen una conclusión individual cada uno. Si hay 4 analistas, habrá 4 conclusiones basadas en los mismos datos.

**Clave:** Una capa es un conjunto de neuronas que trabajan en paralelo, todas reciben la misma información de la capa anterior.

---

### 2. La Capa de Salida: Tomando Decisiones

- **El Objetivo:**  
  Queremos que la red decida si un punto pertenece a la luna A o la luna B (decisión binaria).  
- **La Herramienta:**  
  Usamos una neurona con activación **Sigmoide**, que convierte cualquier número en un valor entre 0 y 1.  
- **Interpretación:**  
  El resultado es una probabilidad. Por ejemplo:  
  - Resultado 0.9 → muy probable que sea la luna B  
  - Resultado 0.1 → muy poco probable que sea la luna B (probablemente luna A)  
  - Resultado 0.5 → indecisión total

**Pregunta:** Para obtener ese único número de probabilidad, ¿cuántas neuronas crees que necesitamos en la capa final?

---

### 3. La Capa Oculta: Decidiendo la Complejidad

- Usaremos **4 neuronas** en la capa oculta para comenzar.  
- Esto no es "la respuesta correcta", sino un **hiperparámetro**, una elección de diseño.  

**Importancia del número de neuronas ocultas:**  
- Pocas neuronas → red muy simple que no capta bien los patrones (subajuste).  
- Muchas neuronas → la red puede memorizar los datos, pero falla en datos nuevos (sobreajuste).  

4 neuronas es un buen punto medio para este problema: ni muy simple ni muy complejo.

---

### Resumen de Nuestra Arquitectura

| Capa           | Número de Neuronas             | Descripción                       |
|----------------|-------------------------------|---------------------------------|
| Capa de Entrada| `n_x` (definido por el usuario)| Número de datos de entrada       |
| Capa Oculta    | 4                             | Número fijo para empezar         |
| Capa de Salida | `n_y` (definido por el usuario)| Número de salidas (por ejemplo, 1 para binaria) |

---

## ✨ Puntos Finales

* La capa de entrada es solo el punto de partida para recibir datos, sin cálculos propios.  
* Cada neurona en una capa trabaja con toda la entrada que recibe, pero produce su propio valor único.  
* El número de neuronas en capas ocultas es una decisión clave que afecta la capacidad de aprendizaje y generalización de la red.  
* La capa de salida traduce la información en decisiones concretas, usualmente en forma de probabilidades con funciones de activación específicas.


## ⚙️ Implementación: Preparando los Parámetros Iniciales

### 🧱 Nuestra Estructura

Para implementar la red, vamos a utilizar la siguiente arquitectura:

- **Capa de Entrada** $(n_x)$: 2 neuronas  
  *(porque cada punto tiene 2 coordenadas: X e Y)*

- **Capa Oculta** $(n_h)$: 4 neuronas  
  *(suficiente capacidad para aprender la forma curva de las lunas)*

- **Capa de Salida** $(n_y)$: 1 neurona  
  *(queremos una única probabilidad como resultado final)*

---

### 🧰 Paso Siguiente: Obtener los "Materiales"

Para que la red neuronal funcione, necesitamos definir sus **parámetros de entrenamiento**: los **pesos** ($W$) y los **biases** ($b$).

Tendremos **dos conjuntos** de parámetros:

---

#### 🔗 1. De la **Capa de Entrada** a la **Capa Oculta**:

- Matriz de pesos:  
  $$
  W_1 \in \mathbb{R}^{(4,\ 2)}
  $$

- Vector de bias:  
  $$
  b_1 \in \mathbb{R}^{(4,\ 1)}
  $$

---

#### 🔗 2. De la **Capa Oculta** a la **Capa de Salida**:

- Matriz de pesos:  
  $$
  W_2 \in \mathbb{R}^{(1,\ 4)}
  $$

- Vector de bias:  
  $$
  b_2 \in \mathbb{R}^{(1,\ 1)}
  $$

---

Con estos parámetros listos, estaremos en condiciones de construir la **función de propagación hacia adelante**, calcular el **costo**, y luego ajustar los parámetros mediante **backpropagation**.



## 🔢 Matriz de Pesos y Vector de Bias: Cómo se Calculan y Por Qué



### 🧠 Conexión entre Capas: ¿Qué es una Matriz de Pesos?

Cuando conectamos dos capas en una red neuronal, necesitamos una **matriz de pesos** que defina cómo se transmite la información desde una capa a otra. Esta matriz representa la "fuerza" de cada conexión entre neuronas.

---

### 📐 Dimensiones de la Matriz de Pesos **W₁**

- **Contexto:** Conectamos la **capa de entrada** (de tamaño 2) con la **capa oculta** (de tamaño 4).  
- **Forma de la matriz:**  
  $$
  \text{Shape de } W_1 = (4,\ 2)
  $$

- **¿Por qué (4, 2)?**  
  - Cada una de las 4 neuronas en la capa oculta recibe 2 entradas (una por cada neurona de entrada).
  - Esto significa que necesitamos **4 filas** (una por cada neurona de destino) y **2 columnas** (una por cada entrada).
  - **Regla general:**  
    $$
    \text{Shape de } W = (\text{dimensión de la capa de destino},\ \text{dimensión de la capa de origen})
    $$

---

### 🧮 Multiplicación Matricial

Para que las operaciones funcionen correctamente en NumPy:

- **Vector de entrada:**  
  $$
  x = (2,\ 1)
  $$

- **Matriz de pesos:**  
  $$
  W_1 = (4,\ 2)
  $$

- **Multiplicación:**  
  $$
  z = W_1 \cdot x \Rightarrow \text{Shape de } z = (4,\ 1)
  $$

Así, obtenemos un vector de 4 elementos: uno para cada neurona de la capa oculta. ¡Justo lo que esperábamos!

---

### ➕ Vector de Bias **b₁**

Después de la multiplicación con la matriz de pesos, cada neurona también recibe un **bias** individual que se suma a su entrada ponderada. Es como un "desplazamiento personal" para cada neurona.

- **Forma del bias:**  
  $$
  \text{Shape de } b_1 = (4,)
  $$

- **¿Por qué (4,)?**  
  Porque la **capa de destino** (la capa oculta) tiene 4 neuronas, y cada una necesita su propio bias.

---

### 🧩 Recapitulación Visual

| Elemento         | Shape       | Significado                                       |
|------------------|-------------|--------------------------------------------------|
| `W₁`             | (4, 2)      | 4 neuronas de salida, 2 entradas cada una        |
| `x`              | (2, 1)      | Vector de entrada con 2 características          |
| `W₁ · x`         | (4, 1)      | Resultado: un valor por cada neurona oculta      |
| `b₁`             | (4,)        | Bias para cada una de las 4 neuronas ocultas     |

---

**¿Por qué todo esto importa?**  
Porque estas dimensiones aseguran que las operaciones matemáticas se realicen correctamente y que la red tenga la capacidad de aprender adecuadamente. Si las dimensiones no coinciden, ¡el código simplemente no funcionará!

## 🔗 Conexión Final: Capa Oculta → Capa de Salida


### 🧠 Matriz de Pesos **W₂**

Ahora conectamos la **capa oculta** con la **capa de salida**. Recordemos:

- La capa oculta tiene **4 neuronas**.
- La capa de salida tiene **1 sola neurona** (porque estamos haciendo una **clasificación binaria**: luna A o luna B).

#### 🔢 Dimensión de la Matriz de Pesos:

$$
\text{Shape de } W_2 = (1,\ 4)
$$

- **¿Por qué (1, 4)?**
  - La matriz conecta 4 valores de entrada (uno por cada neurona oculta) con 1 neurona de salida.
  - Siguiendo la convención:
    $$
    \text{Shape de } W = (\text{dimensión de la capa de destino},\ \text{dimensión de la capa de origen})
    $$

---

### ➕ Vector de Bias **b₂**

Cada neurona en la capa de destino necesita su **bias** individual.

- Como la capa de salida tiene **una sola neurona**, el vector de bias tendrá:
  $$
  \text{Shape de } b_2 = (1,)
  $$

---

### 🧩 Recapitulación Visual

| Elemento         | Shape       | Significado                                          |
|------------------|-------------|-----------------------------------------------------|
| `W₂`             | (1, 4)      | 1 neurona de salida, conectada a 4 neuronas ocultas |
| `a₁`             | (4, 1)      | Activaciones de la capa oculta                      |
| `W₂ · a₁`        | (1, 1)      | Resultado: activación de la única neurona de salida |
| `b₂`             | (1,)        | Bias para la única neurona de salida                |

---

**🧠 ¿Qué obtiene la red al final?**

Un único número entre 0 y 1 (gracias a la activación sigmoide) que interpreta como la **probabilidad de que el punto pertenezca a la luna B**.


## 🎯 ¿Por qué inicializamos los pesos aleatoriamente?


### 💥 El Problema Real: *Ruptura de Simetría* (Symmetry Breaking)

Supongamos que cometemos el error de inicializar **todos los pesos con ceros**.

---

### 🔁 Escenario Desastroso (Inicialización con ceros)

Tenemos una **capa oculta con 4 neuronas**. Si todos los pesos de $W_1$ y los biases $b_1$ son cero, pasaría lo siguiente:

#### 1. **Forward Pass**
Recibimos una entrada: $(x_1,\ x_2)$  
Cada neurona de la capa oculta calculará:

- Neurona 1: $x_1 \cdot 0 + x_2 \cdot 0 + 0 = 0$
- Neurona 2: $x_1 \cdot 0 + x_2 \cdot 0 + 0 = 0$
- Neurona 3: $x_1 \cdot 0 + x_2 \cdot 0 + 0 = 0$
- Neurona 4: $x_1 \cdot 0 + x_2 \cdot 0 + 0 = 0$

✅ Todas producen **exactamente la misma salida**. Son **clones perfectos**.

---

#### 2. **Backward Pass (Retropropagación)**

Ahora calculamos los gradientes. Como todas las neuronas dieron el mismo resultado, **reciben el mismo gradiente**.

- Todas serán ajustadas **de la misma forma**.

---

#### 3. **Actualización de Pesos**

Supón que el gradiente para cierto peso es $g$ y usamos una tasa de aprendizaje $\alpha$.

- Peso actualizado: $0 - \alpha \cdot g$
- Resultado: **todos los pesos siguen siendo idénticos**

---

### 🚨 Consecuencia: La red nunca aprende

Estas neuronas nunca se diferenciarán unas de otras.  
Siempre producirán la misma salida → recibirán el mismo error → actualizarán sus pesos de la misma forma.

👉 Es como tener **una sola neurona repetida 4 veces**, desperdiciando completamente la capacidad de la red.

---

### ✅ La Solución: Inicialización Aleatoria

Para evitar ese desastre, **rompemos la simetría**:

- Los pesos $W_1$ y $W_2$ se inicializan con **valores pequeños y aleatorios**
  $$
  W_1 \sim \mathcal{N}(0,\ \epsilon),\quad W_2 \sim \mathcal{N}(0,\ \epsilon)
  $$

Esto garantiza que:

- Cada neurona de la capa oculta empieza con una perspectiva distinta.
- Generan salidas diferentes desde el primer paso.
- Reciben gradientes diferentes y comienzan a **especializarse**.

---

### ℹ️ ¿Y los Biases?

No hay problema en inicializarlos a cero.

- Cada $b_i$ afecta **solo a su neurona**.
- No hay "simetría entre biases" que deba romperse.

---

### 🧠 Conclusión

**Nunca inicialices todos los pesos a cero**.  
Si lo haces, tu red se volverá una fábrica de clones inútiles.

✅ Al usar pesos aleatorios, le das a cada neurona **una oportunidad de aprender cosas diferentes**, convirtiendo la red en un sistema verdaderamente inteligente.

como buena practica hacemos un diccionario donde podamos guardar los valores de la capas para nuestra duncion de inicialiacion de parametros de nuestra red neuronal.

## 🚀 Implementar la Propagación hacia Adelante (Forward Propagation)


Ya tenemos nuestros parámetros (`W1`, `b1`, `W2`, `b2`), así que ahora toca usarlos para hacer que la red funcione: **convertir una entrada $X$ en una predicción final**.

---

### 🔁 ¿Qué es la Propagación hacia Adelante?

Es el proceso de pasar los datos de entrada **capa por capa**, hasta llegar a la salida de la red.

Para **cada capa**, el proceso tiene dos pasos:

1. **Cálculo Lineal**  
   Se multiplican las activaciones de la capa anterior por los pesos y se les suma el bias:  
   **Z = W ⋅ A(anterior) + b**

2. **Cálculo de Activación**  
   Se aplica una función no lineal sobre $Z$ para obtener la activación $A$:
   **A = activación(Z)**

---

### 🔐 Capa Oculta

Usaremos `tanh` como función de activación.  
Este es el código:

```python
Z1 = np.dot(W1, X1) + b1
A1 = np.tanh(Z1)
````

* `W1`: matriz de pesos con forma (4, 2)
* `X1`: vector o matriz de entrada con forma (2, m)
* `b1`: vector de bias con forma (4, 1)
* `A1`: salida activada de la capa oculta (4, m)

---

### 🎯 Capa de Salida

Aquí usamos la función **sigmoide**, ya que queremos una **salida entre 0 y 1**, útil para tareas de clasificación binaria (como "¿Sí o No?", "¿Clase 1 o Clase 0?").

```python
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
```

Y previamente definimos la función sigmoide:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

* `W2`: matriz de pesos con forma (1, 4)
* `A1`: activación de la capa oculta (4, m)
* `b2`: bias con forma (1, 1)
* `A2`: **predicción final de la red**, forma (1, m)

---

### 🧠 Detalle Técnico: ¿Por qué el bias es (n, 1)?

Cuando calculamos:

```python
Z1 = np.dot(W1, X1) + b1
```

* `np.dot(W1, X1)` da una matriz de forma **(4, m)**.
* `b1` tiene forma **(4, 1)**.

Gracias al **broadcasting** de NumPy:

* NumPy extiende automáticamente el vector columna `b1` para que se pueda sumar a cada columna del resultado sin error.
* Si en vez de `(4, 1)` usaras `(4,)`, NumPy **podría fallar** al hacer broadcasting correctamente, especialmente si cambian las dimensiones de entrada o se vectoriza el código más adelante.

✅ **Usar forma (n, 1) en el bias es la opción más robusta y clara.**

---

### 🧪 Conclusión

La propagación hacia adelante nos da la **predicción** de la red en función de sus pesos y biases. Este paso es crucial porque:

* Nos permite evaluar qué tan bien está funcionando la red.
* Es la base para calcular el error y hacer la retropropagación en el siguiente paso.

## 📉 Calcular la Pérdida o Coste



### ¿Cómo sabemos si la red está haciendo bien las predicciones?

Nuestra red toma un conjunto de datos de entrada **X** y devuelve predicciones **A2**. Pero para saber qué tan buena o mala es esa predicción, necesitamos **medir el error**.

Esto lo hacemos con una **función de coste**, que compara las predicciones **A2** con las etiquetas reales **Y** (los valores verdaderos) y nos devuelve un número que indica qué tan "equivocada" está la red.

---

### ⚖️ Función de coste elegida: Entropía Cruzada Binaria (BCE)

La función de coste **Binary Cross-Entropy (BCE)** es ideal para problemas de clasificación binaria (salida 0 o 1) y está definida así:

$
J = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]
$

donde:

- $m$ = número de ejemplos de entrenamiento
- $y^{(i)}$ = etiqueta verdadera del ejemplo $i$
- $a^{(i)}$ = predicción de la red para el ejemplo $i$

---

### 📝 Notación: ¿Por qué usamos \(Y\) mayúscula en lugar de \(y\) minúscula?

- \( y \) minúscula normalmente se usa para **un solo vector** o ejemplo.
- \( Y \) mayúscula indica que estamos trabajando con un **conjunto de datos completo** (matriz), con muchas etiquetas al mismo tiempo.

Nuestra función de coste está diseñada para calcular el error en **todos los ejemplos a la vez**, operando con matrices \(X\) y \(Y\).

---

### 📊 Relación con Machine Learning clásico

En modelos clásicos de ML, la función de coste suele recibir parámetros como:

- $X$ (datos de entrada)
- $\theta$ (parámetros del modelo)
- $\lambda$ (término de regularización)

En nuestras redes neuronales, estos parámetros están desglosados en múltiples pasos:

- $X$: datos de entrada
- $W_1, b_1, W_2, b_2$: nuestros parámetros (equivalente a $\theta$)
- Funciones intermedias: $Z_1$, $A_1$, $(Z_2)$, $A_2$ (cálculos dentro de la red)
- Más adelante, se puede agregar $\lambda$ para regularización

Por eso, **nuestro "theta" es el conjunto de parámetros $\{W_1, b_1, W_2, b_2\}$** y **$\lambda$** lo añadiremos cuando hablemos de regularización.

---

### 🧮 Implementación en Python

```python
def comput_cost(A2, Y):
    m = A2.shape[1]
    cost_sum = np.sum((Y * np.log(A2)) + ((1 - Y) * np.log(1 - A2)))
    cost_orig = - (1 / m) * cost_sum
    return cost_orig
```

* `A2`: matriz (1, m) con predicciones de la red
* `Y`: matriz (1, m) con etiquetas reales


### 🧠 Resumen

* La función coste mide el error entre lo que predice la red y la realidad.
* Minimizar el coste es el objetivo del entrenamiento.
* Usamos entropía cruzada binaria para clasificación binaria.
* Los parámetros y cálculos de la red se distribuyen en varias variables, pero todo forma parte del modelo.



Con esta función ya puedes saber cuán bien está funcionando tu red en cada paso. El siguiente paso será usar esta medida para ajustar tus pesos y biases (¡retropropagación!).

## 🔄 Implementar Backpropagation (Retropropagación)

### 🎯 Objetivo

Queremos ajustar los **pesos (`W`)** y los **sesgos (`b`)** de nuestra red para que cometa **menos error**. Para eso, necesitamos saber cómo cada parámetro influye en el coste total, y eso se logra calculando los **gradientes**.

---

### ❓ ¿Qué es un gradiente y por qué lo necesitamos?

El **gradiente** de un parámetro responde a esta pregunta:

> "Si cambio este parámetro un poquito, ¿cuánto y en qué dirección cambia el coste total?"

Este conocimiento nos permite modificar los parámetros **en la dirección correcta para reducir el error** usando el algoritmo de **descenso de gradiente**.



## 🧠 Intuición detrás de la retropropagación

Piensa en una "cadena de culpa":

1. Si el coste es alto, quiere decir que la predicción fue mala.
2. Retropropagamos ese error hacia atrás en la red para ver cuánta **"culpa"** tiene cada parámetro (peso o sesgo).
3. Así sabemos **qué ajustar** y **cuánto**.

Este proceso se basa en la **regla de la cadena** del cálculo diferencial.

---

### 🔹 Paso 1: calcular el error de salida (`dZ2`)

La última capa usa **función sigmoide**, por eso su derivada es simple:

$
dZ2 = A2 - Y
$

Esto representa **el error entre lo que predijo la red (`A2`) y la verdad (`Y`)**.

---

### 🔹 Paso 2: calcular gradientes de los parámetros de salida

Sabemos que:

$
Z2 = W2 \cdot A1 + b2
$

Entonces aplicamos las derivadas:

- **`dW2 = dZ2 ⋅ A1ᵀ`** → mide cuánto debe ajustarse cada peso
- **`db2 = suma(dZ2)`** → mide cuánto debe ajustarse cada sesgo


### 🔁 **Ese es el corazón de la *retro*propagación**

La **predicción final (`A2`)** es donde se manifiesta el **error**. Ahí es donde **podemos medir cuánto se equivocó la red** comparando con las etiquetas reales (`Y`).

Pero los parámetros que generaron esa predicción (los pesos y sesgos de todas las capas) están **más atrás** en la red.

---

### 🧠 Entonces, ¿por qué ir desde atrás hacia adelante?

1. **El error lo puedes calcular solo cuando tienes la predicción (`A2`)**.
2. Para saber **cómo ese error fue "causado" por los pesos anteriores**, debes ir **hacia atrás**, aplicando la **regla de la cadena** de la derivada.
3. Así descubres:

   * Qué tan responsables son `W2` y `b2` del error.
   * Luego, qué tanto contribuyeron `W1` y `b1` a generar la activación que luego generó el error.

---

### 🔗 Metáfora sencilla:

Piensa en una fábrica de botellas defectuosas.

* El defecto se nota **al final**, en la salida del producto.
* Pero para saber **dónde estuvo el problema** (moldeo, llenado, tapado...), tienes que **revisar la línea de montaje en reversa** hasta encontrar qué paso contribuyó al defecto.

Eso es retropropagación: **"propagar el error hacia atrás"** para corregir donde realmente importa.

---

### ✅ En resumen:

* Iniciamos con `dZ2 = A2 - Y` porque es **donde podemos medir el error**.
* Retrocedemos para calcular los efectos de ese error en los **parámetros anteriores**.
* De ahí el nombre: **retro** + **propagación** = propagar el error hacia atrás.


## Implementación de la Retropropagación (Backpropagation)

Ahora que sabemos cómo calcular la predicción de nuestra red (con la **propagación hacia adelante**), necesitamos aprender cómo **corregir los errores** que cometió. Para eso usamos la **retropropagación**.


### ¿Cómo lo hacemos?

1. **Primero vemos el error en la salida**, es decir, cuánto se equivocó la predicción de la red respecto a la verdad.

2. Luego, **repartimos ese error hacia atrás** para saber cuánto afectaron los parámetros de la última capa, y luego los de la capa anterior, y así sucesivamente.

---

### Explicación del código paso a paso

```python
def backward_propagation(parameters, cache, X, Y):
```

Esta función toma:

* `parameters`: los pesos de la red (en este caso sólo `W2` que es el de la última capa),
* `cache`: los valores que guardamos en la propagación hacia adelante (las activaciones `A1` y `A2`),
* `X`: los datos de entrada originales,
* `Y`: las respuestas correctas (las etiquetas verdaderas).

---

```python
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
```

Aquí simplemente sacamos de la memoria los valores que necesitamos para hacer el cálculo.

---

```python
    m = A1.shape[1]  # número de ejemplos
```

`m` es la cantidad de datos que estamos procesando a la vez. Lo usamos para hacer un promedio y no que un solo dato influya demasiado.

---

### Capa de salida (la última capa)

```python
    dZ2 = A2 - Y
```

* Esto es el **error directo en la salida**.
* `A2` es la predicción que hizo la red,
* `Y` es la respuesta correcta.
* Restamos para saber cuánto nos equivocamos.

---

```python
    dW2 = (1/m) * (dZ2 @ A1.T)
```

* Aquí calculamos cuánto hay que cambiar cada peso `W2`.
* Multiplicamos ese error `dZ2` por lo que salió de la capa anterior (`A1`), pero transpuesto para que las dimensiones cuadren.
* Dividimos por `m` para promediar.

---

```python
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
```

* Aquí calculamos cuánto cambiar el sesgo `b2`.
* Sumamos todos los errores de los datos para cada neurona y hacemos promedio.

---

### Capa oculta (la capa anterior)

```python
    dA1 = W2.T @ dZ2
```

* Retropropagamos el error hacia atrás.
* Ahora vemos cómo el error en la salida afecta a la capa oculta.
* Multiplicamos el error por la transpuesta de `W2` para repartir la “culpa” entre las neuronas de la capa oculta.

---

```python
    dZ1 = dA1 * (1 - A1**2)
```

* Aquí aplicamos la derivada de la función de activación **tanh**.
* La derivada de tanh(z) es (1 - tanh(z)^2), y multiplicamos por `dA1` para ajustar el error a la activación.

---

```python
    dW1 = (1/m) * np.dot(dZ1, X.T)
```

* Calculamos cuánto cambiar los pesos de la capa oculta.
* Multiplicamos el error ajustado (`dZ1`) por la entrada original `X` (transpuesta).
* Promediamos con `m`.

---

```python
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
```

* Calculamos cuánto cambiar el sesgo de la capa oculta.
* Sumamos todos los errores y promediamos.

---

### Guardamos todos los resultados para usarlos después

```python
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads
```

---

### En resumen:

* Calculamos el **error final** en la salida (`dZ2`).
* Usamos ese error para saber cuánto cambiar los pesos y sesgos de la capa final (`dW2`, `db2`).
* Luego, calculamos cuánto ese error afecta a la capa oculta (`dA1`).
* Ajustamos ese error con la derivada de la función de activación (`dZ1`).
* Calculamos cuánto cambiar los pesos y sesgos de la capa oculta (`dW1`, `db1`).
* Devolvemos todos estos cambios para poder actualizar nuestros parámetros y mejorar la red.

## 🔄 Actualizar los Parámetros: El Paso Clave para que la Red Aprenda

### ❓ ¿Por qué actualizamos los parámetros?

Imagina que estás en la cima de una montaña 🏔️, pero hay una **niebla tan densa** que no puedes ver hacia dónde ir para bajar. Tu objetivo es encontrar el punto más bajo — el **valle**, donde el error de la red sea mínimo.

* Cuando **inicializamos los parámetros** (`initialize_parameters`), es como caer en un lugar aleatorio de esa montaña.
* La **altura** donde estás parado representa el **coste** o el **error** de la red.

  * Si estás **muy alto** ➡️ el error es **grande**.
  * Si estás **bajo** ➡️ el error es **pequeño**.

---

### 🧭 ¿Cómo sabemos hacia dónde bajar?

No puedes ver el valle por la niebla, pero sí puedes **sentir la pendiente** justo donde estás parado. Esa pendiente te indica la dirección de la subida más fuerte, y tú quieres ir en la dirección contraria, para bajar.

* Eso es lo que hace la **retropropagación** (`backward_propagation`):
  🔍 **Te dice cuál es la dirección de la pendiente (gradientes) más pronunciada,**
  es decir, hacia dónde sube más el error.

* El resultado son los **gradientes** (`grads`), que nos dicen:
  👉 **“Qué tanto”** y **“en qué dirección”** están subiendo los errores para cada parámetro.

---

### 🏃‍♂️ ¿Y luego qué hacemos?

Sabemos la dirección de subida, entonces **damos un paso pequeño en la dirección opuesta** para bajar la montaña.

* Esto es la **actualización de parámetros** (`update_parameters`).
* La fórmula clave es:

```markdown
parámetro_nuevo = parámetro_viejo - tasa_de_aprendizaje × gradiente
```

* 📏 **La tasa de aprendizaje** es el tamaño del paso que damos.

  * Si es muy grande ⚠️, podríamos pasarnos del valle.
  * Si es muy pequeño 🐢, tardamos mucho en llegar.

---

### 🌟 Resumen de la analogía:

| 🛠️ Paso                   | 🏔️ En la montaña                        | 🤖 En la red neuronal                                      |
| -------------------------- | ---------------------------------------- | ---------------------------------------------------------- |
| 🎲 Inicializar parámetros  | Caer en un punto aleatorio               | Asignar pesos y sesgos aleatorios                          |
| 👣 Sentir la pendiente     | Sentir la inclinación del suelo          | Calcular gradientes con retropropagación                   |
| ⬇️ Dar un paso hacia abajo | Caminar en dirección opuesta a la subida | Actualizar parámetros con gradientes y tasa de aprendizaje |
| 🔁 Repetir muchas veces    | Seguir caminando hasta el valle          | Entrenar la red para minimizar el error                    |

---

### 🎯 ¿Qué logramos con esto?

Cada paso que damos hace que el **coste baje poco a poco**. Nuestra red mejora sus predicciones y aprende a partir de los datos.

Este ciclo:
🔁 **Calcular error → calcular gradientes → actualizar parámetros**
es lo que permite a la red **aprender de sus errores** y ajustar sus "conexiones" para ser cada vez más precisa.


## 🔄 Iteración: ¿Por qué repetir los pasitos?

Una sola vez NO alcanza para que la red aprenda bien. ¡Es como querer bajar una montaña con un solo paso! 🏔️👣

---

### 🎯 El entrenamiento es un proceso repetitivo:

1. 🚀 **Forward pass:**
   Calculamos la salida de la red y cuánto se equivocó.

2. 🎯 **Calcular el costo:**
   Medimos qué tan grande es ese error.

3. 🔙 **Backward pass:**
   Calculamos los gradientes, es decir, hacia dónde y cuánto ajustar para mejorar.

4. 🏃‍♂️ **Actualizar parámetros:**
   Damos un pequeño paso ajustando pesos y sesgos para bajar el error.

---

### 🔁 ¿Y qué pasa con todo esto?

Hay que repetir este ciclo **muchísimas veces**, para que la red mejore poco a poco. Cada repetición se llama:

* **Iteración** o
* **Epoch**

---

### 🧗‍♂️ Metáfora para entenderlo mejor:

Imaginá que estás bajando una montaña en plena niebla.

* Un solo paso no te lleva hasta el valle.
* Necesitás dar **muchos pasitos pequeños, uno tras otro**.

Solo así, poco a poco, vas acercándote al punto más bajo donde el error es mínimo.

---

### ⚙️ Entonces, armamos un bucle (loop) que repite:

> Forward → Costo → Backward → Actualizar

... muchas veces, para que la red aprenda de verdad.

## 🧠 Entrenar Nuestra Red Neuronal

Ahora vamos a juntar todo lo que aprendimos para construir y **entrenar nuestra red neuronal**.
Esto incluye: preparar los datos, definir la arquitectura, entrenar el modelo y visualizar los resultados.

---

### 📥 Paso 1: Cargar el Dataset `make_moons`

Usamos `make_moons` porque genera datos con forma de medialuna:
una medialuna para la clase 0, y otra para la clase 1. ¡Ideal para probar modelos no lineales!

```python
X, Y = make_moons(n_samples=400, noise=0.2)
```

* `n_samples=400`: generamos 400 puntos.
* `noise=0.2`: le agregamos ruido para hacerlo más realista.

---

### 🔄 Paso 2: Reorganizar los Datos

> 💡 **¿Por qué reorganizamos X y Y?**

Porque al trabajar con redes neuronales y NumPy, **es más eficiente que cada columna sea un ejemplo**, en vez de cada fila.

```python
X = X.T        # De (400, 2) a (2, 400)
Y = Y.reshape(1, -1)  # De (400,) a (1, 400)
```

Esto permite hacer operaciones vectorizadas como:

```python
Z = np.dot(W, X) + b
```

¡Sin bucles! Más rápido y más limpio 🧼

---

### 👀 Paso 3: Visualizar los Datos

Antes de entrenar, **miramos cómo se ven los datos**:

![alt text](<Scatter plot of make_moon dataset.png>)

---

### 🧠 Paso 4: Entrenar la Red Neuronal

Ahora sí, ¡el corazón del proyecto!
Entrenamos la red con 1 capa oculta de 4 neuronas, durante 10.000 iteraciones:

```python
trained_parameters, costs = nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.2)
```

* `n_h=4`: 4 neuronas en la capa oculta.
* `learning_rate=1.2`: cuán grandes son los pasitos que damos.
* `num_iterations=10000`: cuántas veces vamos a repetir el proceso de aprendizaje.

---

### 📊 Paso 5: Visualizar la Frontera de Decisión

¿Qué tan bien aprendió nuestra red?
Veamos cómo separa las dos clases visualmente 👇

```python
plot_decision_boundary(lambda x: predict(trained_parameters, x), X, Y)
```

Esto genera una gráfica como esta:

![alt text](<Decision boundary.png>)

---

### 📉 Paso 6: Ver cómo baja el Costo

Durante el entrenamiento, registramos cómo fue bajando el error:

```python
plt.plot(costs)
plt.xlabel("iterations (per thousands)")
plt.ylabel("cost")
plt.title("Cost reduction over time")
plt.show()
```

Esto nos permite ver si el modelo **está aprendiendo** correctamente o no:

* 📉 Si baja → vamos bien.
* 📈 Si sube o no baja → problema (por ejemplo, learning rate muy alto).

![alt text](<Cost reduction overtime.png>)

---

### 🧪 Paso Extra: Semilla Fija para Resultados Reproducibles

Para que siempre obtengamos el mismo resultado (útil para pruebas y debugging):

```python
np.random.seed(42)
```

---

## 🧩 ¿Qué hace la función `nn_model()`?

Esta es la función que **entrena todo el modelo**.
Combina todos los pasos clave:

* Inicializar los parámetros
* Forward propagation
* Cálculo del costo
* Backward propagation
* Actualizar los parámetros
* Repetir todo esto por muchas iteraciones

