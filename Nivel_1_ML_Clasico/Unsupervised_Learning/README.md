# 🔵 K-Means Clustering (Agrupamiento K-Medias)

**K-Means** es una técnica de aprendizaje no supervisado utilizada para dividir un conjunto de datos en **K grupos (clústeres)** distintos pero no superpuestos. El objetivo es que las observaciones dentro de cada grupo sean muy similares entre sí y diferentes de las observaciones en otros grupos.

Para aplicarlo, es necesario **especificar previamente el número de clústeres K** que se desean encontrar.

---

## 🎯 Objetivo de Minimización en Clustering

El propósito principal del algoritmo es que la **variación interna de cada clúster** sea lo más pequeña posible. Se busca minimizar la **suma total de las variaciones internas** de los clústeres.

Esto se expresa mediante la siguiente función objetivo:

$\min_{C_1, \dots, C_K} \left\{ \sum_{k=1}^{K} W(C_k) \right\}$

Donde:

* $C₁, ..., C_K$ son los clústeres formados.
* $W(C_k)$ representa la **variación interna** del clúster * $C_k$.
* La suma total recorre todos los K clústeres.

---

### 📏 Definición de la Variación Interna de un Clúster: W(Cₖ)

La variación dentro de un clúster $W(C_k)$ se define como la suma de las **distancias euclidianas al cuadrado** entre cada punto del clúster y su centroide (la media de sus características)**(inercia)**:

$W(C_k) = \sum_{i \in C_k} \sum_{j=1}^{p} (x_{ij} - \bar{x}_{kj})^2$

Donde:

* $i ∈ C_k$: Recorre todas las observaciones del clúster $C_k$.
* $j = 1,...,p$: Recorre todas las **p características** del conjunto de datos.
* $x_{ij}$: Valor de la característica **j** de la observación **i**.
* $𝑥̄_{kj}$: Media de la característica **j** en el clúster $C_k$.

---

## ⚙️ ¿Cómo Funciona el Algoritmo?

Encontrar una solución óptima global para K-Means es computacionalmente difícil. Por eso, se utiliza un algoritmo **iterativo** que converge a un **óptimo local**.

### Pasos del Algoritmo

1. **Inicialización aleatoria**:
   Se asigna aleatoriamente un número del 1 al K a cada observación (asignación inicial de clústeres).

2. El Proceso Iterativo de K-Means

El algoritmo K-Means alterna entre dos pasos fundamentales hasta que converge (es decir, hasta que las asignaciones de clústeres ya no cambian):

   1. **🧭 Paso de Asignación:**
        Cada punto de datos se asigna al **clúster cuyo centroide esté más cercano**, típicamente utilizando la **distancia euclidiana** como medida de cercanía.

   2. **📍 Paso de Actualización:**
        Se **recalcula la posición de cada centroide**. El nuevo centroide es simplemente la **media (promedio vectorial)** de todas las observaciones que fueron asignadas a ese clúster en el paso anterior.

Este proceso garantiza que la **variación total dentro de los clústeres disminuya (o al menos no aumente)** en cada iteración, y finaliza cuando ya no hay cambios en las asignaciones: el algoritmo ha llegado a un **óptimo local**.

---

## 🧠 Resumen del Algoritmo K-Means (Iterativo)

El algoritmo **K-Means** sigue un enfoque iterativo para encontrar grupos (clústeres) en los datos. Su funcionamiento puede resumirse en los siguientes pasos:

1. **📌 Elegir K:**
   Se decide cuántos **clústeres (K)** se desean encontrar en los datos.

2. **🎯 Inicializar Centroides:**
   Se seleccionan **K puntos iniciales** como centroides. (Existen distintas estrategias para esto, como la inicialización aleatoria o el método *k-means++*).

3. **🔁 Repetir hasta convergencia:**

   * **Paso de Asignación:**
     Cada punto se asigna al **clúster cuyo centroide esté más cercano**, generalmente usando la **distancia euclidiana**.

   * **Paso de Actualización:**
     Se **recalcula cada centroide** como la **media de todos los puntos asignados** a ese clúster.

4. **🛑 Detener:**
   El algoritmo finaliza cuando:

   * Las asignaciones de los puntos ya no cambian, o
   * Los centroides apenas se mueven, o
   * Se alcanza un **número máximo de iteraciones**.

> 💡 Este proceso garantiza que la variación interna (dentro de los clústeres) se reduzca en cada paso, hasta llegar a un **óptimo local**.

---
### ⚠️ Precaución: Óptimos Locales

El resultado final depende de la asignación aleatoria inicial. El algoritmo puede converger a diferentes soluciones dependiendo del punto de partida.

**Solución recomendada:**

* Ejecutar el algoritmo múltiples veces con diferentes inicializaciones.
* Elegir la solución con menor variación total.

📌 En R: la función `kmeans()` incluye el argumento `nstart`.
Se recomienda usar valores altos como `nstart = 20` o `nstart = 50`.

---

## ❓ La Importancia de Elegir K

K-Means requiere que especifiques **de antemano** cuántos clústeres deseas encontrar (**valor de K**).
Elegir el valor óptimo de K es un problema complejo y se analiza con más profundidad en la sección **"Practical Issues in Clustering"**.


## 📚 Aprendizaje Supervisado vs No Supervisado

### 1. ✅ Aprendizaje Supervisado

(Lo que hemos hecho hasta ahora)

En el **aprendizaje supervisado**, el algoritmo aprende a partir de datos que **ya están etiquetados**.
Veamos un ejemplo práctico:

> Supongamos que utilizamos algoritmos como **Regresión Logística**, **SVC** o **Random Forest** con el conjunto de datos de **cáncer de mama**.

#### 🎯 Objetivo:

Predecir si un tumor es **maligno** o **benigno**.

#### 🧠 ¿Con qué información entrena el modelo?

* **Características (X):** radio medio, textura media, etc.
* **Variable de respuesta (y):** si el tumor es benigno (0) o maligno (1).

El algoritmo **aprende la relación entre X e y** y la utiliza para hacer predicciones sobre nuevos datos en los que solo conocemos X.

#### 🧩 ¿Qué significa “supervisado”?

Significa que el algoritmo **recibe una “supervisión”** en forma de etiquetas correctas (y). Aprende comparando sus predicciones con las respuestas verdaderas durante el entrenamiento.

> 📌 En el caso del cáncer de mama, la variable de respuesta es la columna `target` del dataset `load_breast_cancer`, que contiene los valores 0 y 1.

---

### 2. 🔍 Aprendizaje No Supervisado

(Aquí entra K-Means)

En el **aprendizaje no supervisado**, **no se proporcionan etiquetas ni categorías**.
El objetivo es **descubrir patrones ocultos o estructura natural** en los datos.

#### 📦 Imagina:

Tienes un conjunto de datos sobre clientes de una tienda:

* Edad, gasto promedio, frecuencia de compra, productos visualizados...

Pero **no tienes una columna** que diga si el cliente es "VIP", "ocasional", o "en riesgo".
**No hay una variable de respuesta.**

#### 🔍 ¿Qué hace el algoritmo?

* Analiza las **características (X)**.
* Intenta **encontrar grupos o patrones** sin saber de antemano cuántos grupos existen ni cómo deberían estar etiquetados.

> No hay una “respuesta correcta” para aprender. El modelo **descubre por sí mismo** la estructura de los datos.

---

### 🧭 Conexión con el Clustering

El **clustering** (agrupamiento), como el algoritmo **K-Means**, es un ejemplo clásico de aprendizaje no supervisado.

#### 🎯 Objetivo del Clustering:

Agrupar observaciones **sin etiquetas previas**, de forma que:

* Las observaciones **dentro del mismo clúster** sean muy similares entre sí.
* Las observaciones de **clústeres distintos** sean muy diferentes entre sí.

> El algoritmo **K-Means no sabe de antemano** cuántos grupos hay ni a qué grupo pertenece cada punto.
> Su tarea es **descubrir** esa estructura.

---

## 🎯 K-Means++: Una Inicialización Más Inteligente

La **inicialización aleatoria simple** en K-Means puede llevar a soluciones pobres o a convergencia lenta. Para mejorar esto, se utiliza una técnica llamada **K-Means++**, que busca seleccionar los centroides iniciales de forma más estratégica.

### ¿Cómo funciona K-Means++?

1. **🧩 Elegir el primer centroide aleatoriamente** de entre los puntos de datos.
2. **🎯 Elegir los siguientes centroides** uno a uno, seleccionando preferentemente los puntos que están **más lejos** de los centroides ya elegidos.

   * La probabilidad de seleccionar un nuevo punto como centroide es **proporcional al cuadrado de su distancia** al centroide más cercano.
3. Este procedimiento **distribuye mejor los centroides iniciales**, evitando que todos caigan muy cerca unos de otros.

### 🧠 ¿Por qué es mejor?

* Mejora la **estabilidad** del algoritmo.
* Aumenta la **probabilidad de converger a una mejor solución** (óptimo más cercano al global).
* Reduce la **sensibilidad a la aleatoriedad inicial**.

> 💡 **Importante:**
> Los centroides iniciales no son los definitivos. Solo son **puntos de partida** para el proceso iterativo. A partir de ellos, el algoritmo K-Means se encarga de refinar su ubicación mediante asignaciones y actualizaciones sucesivas.

---



## Métodos principales de un objeto `KMeans`

| Método                 | Descripción                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `fit(X)`               | Ajusta el modelo a los datos `X` (entrena el clustering).                |
| `fit_predict(X)`       | Ajusta el modelo y devuelve las etiquetas de clúster para `X`.           |
| `predict(X)`           | Asigna nuevos datos `X` a los clusters ya entrenados.                    |
| `fit_transform(X)`     | Ajusta el modelo y devuelve la distancia de cada punto a cada centroide. |
| `transform(X)`         | Devuelve la distancia de los datos `X` a los centroides ya entrenados.   |
| `set_params(**params)` | Establece parámetros del modelo.                                         |
| `get_params()`         | Obtiene los parámetros actuales del modelo.                              |

---

## Atributos importantes (después de hacer `.fit`)

| Atributo           | Descripción                                                                     |
| ------------------ | ------------------------------------------------------------------------------- |
| `cluster_centers_` | Coordenadas de los centroides encontrados.                                      |
| `labels_`          | Etiquetas asignadas a cada muestra (clúster).                                   |
| `inertia_`         | Suma de las distancias al cuadrado a los centroides (criterio de optimización). |
| `n_iter_`          | Número de iteraciones que tomó el algoritmo para converger.                     |

---

### Ejemplo rápido:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print(kmeans.cluster_centers_)  # centroides
print(kmeans.labels_)           # etiquetas
print(kmeans.inertia_)          # suma de distancias al cuadrado
```


## 📊 Inercia y el Método del Codo

### 📌 ¿Qué es la Inercia?

La **inercia** es una métrica utilizada en K-Means para evaluar qué tan bien se han agrupado los datos. Se define como:

> **La suma de las distancias al cuadrado** de cada punto a su **centroide asignado**.

* Valores **más bajos** de inercia indican que los **clusters son más compactos** y que los puntos están más cerca de su centroide.
* Sin embargo, la inercia **siempre disminuye** al aumentar el número de clusters K (o, en el peor caso, permanece igual), lo que puede inducir a error si solo se minimiza esta métrica.

---

### 📈 Gráfica Inercia vs. Número de Clusters (K)

Al graficar la inercia en función de diferentes valores de K, ocurre lo siguiente:

* Inicialmente, la inercia **disminuye rápidamente** conforme agregamos más clusters.
* Luego, la curva **comienza a aplanarse**: los beneficios de seguir aumentando K se vuelven **marginales**.
* Si K fuera igual al número total de puntos, la inercia sería **cero** (cada punto sería su propio cluster).

---

# 🦾 El Método del Codo

El objetivo es encontrar un **punto óptimo** donde obtener una buena segmentación sin usar más clusters de los necesarios.

* Este punto se llama **"el codo"** de la curva.
* Representa un **equilibrio entre compacidad y simplicidad**.
* Más allá del codo, **añadir más clusters no mejora significativamente la inercia** (ley de rendimientos decrecientes).

> 🧠 **Consejo práctico**:
> El método del codo es una guía visual. Siempre es buena idea **complementarlo con otros métodos** o con conocimiento del dominio para decidir el valor óptimo de K.

## 🎯✨ **Cómo usar el Método del Codo para encontrar el mejor K en KMeans**

Imaginemos que queremos agrupar datos en grupos (clusters) y no sabemos cuántos grupos son los ideales. El **Método del Codo** nos ayuda a descubrir ese número perfecto, llamado **K**.

---

### Paso a paso para encontrar el "codo" 🦶

1. **Elegir un rango de K para probar**

   * Por ejemplo, probamos desde 1 hasta 10 grupos.
   * Así evaluamos varias opciones y no nos quedamos con una sola suposición.

2. **Probar cada valor de K**

   * Para cada número de grupos (K):

     * Creamos un modelo KMeans con esa cantidad de grupos.

       > *Consejo:* Usamos `init='k-means++'` y `random_state=42` para que los resultados sean consistentes cada vez.
     * Ajustamos el modelo a nuestros datos (X).
     * Calculamos la **inercia**, que es una medida de qué tan bien los datos están agrupados:

       * Inercia baja = grupos compactos y bien definidos.
     * Guardamos el valor de inercia para cada K.

3. **Hacer un gráfico con los resultados**

   * En el eje horizontal (X) ponemos los valores de K (1, 2, 3, …).
   * En el eje vertical (Y) ponemos la inercia calculada para cada K.

4. **Buscar el "codo" en la gráfica**

   * El "codo" es el punto donde la disminución de la inercia se vuelve menos pronunciada, formando una curva parecida a un brazo doblado.
   * Ese punto indica que agregar más grupos no mejora mucho la agrupación.
   * Elegir ese K es la mejor opción para balancear simplicidad y precisión.

---

🎨 **Ejemplo simple:**
Imaginá que tirás bolas en cajas: al principio, cada caja está muy llena y juntar más bolas en menos cajas mejora mucho. Pero llega un momento en que agregar más cajas no ayuda casi nada a ordenar mejor las bolas. Ese momento es el "codo".

---

📌 **Recordatorio rápido:**

* El Método del Codo es como un detective que busca el punto justo donde dejar de dividir en grupos.
* Usar la gráfica es clave, porque no hay fórmula mágica: ¡es cuestión de observar!

---
# 🎨🔍 **Entendiendo el Coeficiente de Silueta para evaluar grupos (clusters)**

Cuando agrupamos datos, queremos saber qué tan bien están organizados esos grupos. El **Coeficiente de Silueta** es una forma simple y visual de medir esto para cada punto y para todo el conjunto.

---

### ¿Qué es el Coeficiente de Silueta? 🤔

* Es un número que va desde **−1 hasta +1**.
* Nos dice qué tan bien un punto está ubicado en su grupo comparado con otros grupos cercanos.

---

### ¿Qué significan los valores?

- **🟢 Cerca de +1:**  
  El punto está muy bien ubicado dentro de su grupo y lejos de otros grupos. ¡Perfecto! 🎯

- **🟡 Cerca de 0:**  
  El punto está justo en el borde, entre dos grupos. No está claro a cuál pertenece. 🤷

- **🔴 Cerca de −1:**  
  El punto probablemente fue asignado al grupo equivocado, porque está más cerca de otro grupo diferente. 🚩


---

### ¿Qué mide el Coeficiente de Silueta para cada punto?

Se fija en dos cosas importantes:

1. **Cohesión dentro del grupo** $a_i$

   * Qué tan cerca está el punto de los otros puntos de su propio grupo.
   * Se calcula como la distancia promedio entre el punto $i$ y todos los demás puntos en el mismo cluster.
   * Si $a_i$ es pequeño, el punto está bien "metido" en su grupo.

2. **Separación entre grupos** $b_i$

   * Qué tan lejos está el punto $i$ de los puntos del cluster vecino más cercano que no es el suyo.
   * Se calcula como la distancia promedio más pequeña entre $i$ y todos los puntos de otro cluster.
   * Si $b_i$ es grande, el cluster del punto está bien separado de los demás.

---

### La fórmula del Coeficiente de Silueta para un punto $i$

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
$$

* Si $b_i \gg a_i$, entonces $s_i \to +1$ (ideal, buen punto).
* Si $a_i \gg b_i$, entonces $s_i \to -1$ (mala asignación).
* Si $a_i \approx b_i$, entonces $s_i \approx 0$ (punto en el límite).

---

### Silhouette Score general para todo el conjunto

El **Silhouette Score** para un número dado de clusters $K$ es el promedio de los coeficientes de silueta de todos los puntos:

$$
S = \frac{1}{N} \sum_{i=1}^N s_i
$$

Donde:

* $N$ es el número total de puntos.
* Un $S$ cercano a +1 indica clusters bien definidos.
* Un $S$ cercano a 0 o negativo indica agrupamientos pobres o mal definidos.

---

### 🎯 Resumen fácil:

* **Alta cohesión ($a_i$ pequeño) + buena separación ($b_i$ grande) = buen clustering**
* El Coeficiente de Silueta nos da un número para comprobarlo.
* Es una herramienta visual y numérica para decidir cuántos grupos elegir y si están bien formados.

---

💡 **Ejemplo simple:**
Imaginá que estás en un recreo con varios grupos de amigos.

* Si estás muy cerca de tus amigos y lejos de otros grupos, tu silueta es cercana a +1.
* Si estás justo en medio, tu silueta está cerca de 0.
* Si estás más cerca de otro grupo que del tuyo, tu silueta es negativa, y deberías cambiar de grupo.

---


# 🧩 Implementando el Cálculo del Coeficiente de Silueta

El **Coeficiente de Silueta** nos ayuda a elegir el número ideal de grupos (K) cuando usamos el método de agrupamiento KMeans. El proceso es parecido al del Método del Codo, pero con algunas diferencias importantes.

---

### Paso 1: Definir un rango de valores para K

* No podemos usar $K=1$ porque el coeficiente de silueta compara clusters entre sí.
* Por eso, probamos desde $K=2$ hasta un máximo razonable, por ejemplo $K=10$.

---

### Paso 2: Iterar para cada valor de K

Para cada $K$ dentro del rango definido:

1. Crear y entrenar un modelo KMeans con:

   * `n_clusters=K`
   * `init='k-means++'` (mejor inicialización)
   * `n_init='auto'` (número de inicializaciones para estabilidad)
   * `random_state=42` (para reproducibilidad)

2. Obtener las etiquetas de grupo que asigna el modelo a cada punto (con `model.labels_` o `model.fit_predict(X)`).

3. Calcular el **Silhouette Score promedio** usando los datos y etiquetas con la función `silhouette_score` de `sklearn.metrics`.

4. Guardar ese Silhouette Score para ese valor de $K$.

---

### Paso 3: Graficar y elegir el mejor K

* Hacer un gráfico de línea con:

  * Eje X: valores de $K$
  * Eje Y: Silhouette Score promedio para cada $K$

* Buscar el **valor de $K$ que tenga el puntaje más alto**, porque indica la mejor agrupación según la silueta.

---

✨ **Así, con este proceso podemos elegir un número de clusters que agrupe bien los datos, con grupos compactos y separados.**

---

# 🚀 Aplicación de K-Means y Métodos de Evaluación en Datasets Sintéticos y Reales


## 📊 1. Aplicación para el Dataset `make_blobs`

### Paso 1: Cargar y visualizar los datos

* Cargamos los datos del dataset sintético `make_blobs` en variables `X` (características) y `y_true` (etiquetas reales).
* Imprimimos los puntos coloreados para distinguir visualmente los grupos reales.

### Paso 2: Entrenar K-Means con $n\_clusters=3$

* Elegimos arbitrariamente $K=3$ para probar.
* Entrenamos el modelo y extraemos:

  * `y_kmeans_pred = kmeans_model.labels_`: etiquetas asignadas a cada muestra.
  * `centroides_kmeans = kmeans_model.cluster_centers_`: coordenadas de los centroides (centros) de los grupos.
  * `inertia_kmeans = kmeans_model.inertia_`: suma de distancias cuadradas entre puntos y sus centroides (menor inercia = grupos más compactos).

### Paso 3: Visualizar resultados

* Graficamos los puntos con colores según su grupo asignado por K-Means.
* Mostramos los centroides con marcadores rojos grandes.
* Este gráfico ayuda a ver si los grupos están bien separados y centrados.

![Datos y centroides KMeans para make\_blobs](<Data generated for make_blobs data.png>)
![Clusters y centroides KMeans make\_blobs](<Kmeans Clustering with Centroids for Makeblobs data.png>)

---

## 📈 2. Aplicación del Método del Codo (`Elbow method`)

* Probamos diferentes valores de $K$ para K-Means (por ejemplo, de 1 a 10).
* Calculamos la inercia para cada $K$ (suma de distancias cuadradas).
* Graficamos la inercia vs. $K$.
* Buscamos el “codo” en la curva: el punto donde añadir más grupos no reduce mucho la inercia.

![Método del Codo para make\_blobs](<Elbow Method for Makeblobs data.png>)

---

## 📐 3. Cálculo del Coeficiente de Silueta

* Calculamos el coeficiente de silueta para valores de $K$ desde 2 en adelante.
* Este coeficiente mide la calidad del agrupamiento: qué tan bien cada punto encaja en su grupo y qué tan separado está de otros grupos.
* Valores:

  * 🔹 Cerca de 1: excelente agrupamiento (puntos bien dentro de su grupo).
  * 🔸 Cerca de 0: puntos en el límite entre grupos.
  * 🔻 Cerca de -1: puntos posiblemente asignados mal.

![Coeficiente de Silueta para make\_blobs](<Silhouette Coefficient Method for Makeblobs set.png>)

---

## 🌸 4. Aplicación para el Dataset Iris

### Paso 1: Cargar datos

* Cargamos las características y etiquetas reales de las especies del dataset Iris.
* Imprimimos dimensiones y nombres para conocer el dataset.

### Paso 2: Método del Codo en Iris

* Entrenamos modelos K-Means con $K = 1$ a $10$.
* Calculamos inercia para cada modelo y graficamos.
* Identificamos el codo para elegir $K$ óptimo.

![Método del Codo para Iris](<Elbow method for Iris Data.png>)

### Paso 3: Coeficiente de Silueta en Iris

* Calculamos y graficamos el coeficiente de silueta para varios $K$.
* Valor máximo sugiere el mejor número de grupos para Iris.

![Coeficiente de Silueta para Iris](<Silhouette Coefficient Method for Iris Data.png>)

---

## 🎯 5. Visualización y Comparación de Resultados en Iris

* Entrenamos K-Means con $K=3$ (número real de especies).

* Obtenemos etiquetas predichas, centroides y inercia.

* Graficamos dos subplots:

  1. Puntos coloreados según etiquetas predichas por K-Means + centroides en rojo.
  2. Puntos coloreados según etiquetas reales de especies.

* Esto permite comparar visualmente la agrupación del modelo con las clases reales.

![Comparación Clusters Iris](<Comparation Clusters.png>)

---

✨ **Resumen:**
Este recorrido muestra cómo aplicar K-Means y evaluar la cantidad óptima de grupos con métodos visuales y métricas cuantitativas, usando datasets sintéticos (`make_blobs`) y reales (Iris). Los gráficos y métricas ayudan a entender y validar los resultados.

---