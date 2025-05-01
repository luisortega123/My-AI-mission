# My-AI-mission
# Misión 1: Álgebra Lineal con Python Puro


2. **$ \theta $**: Vector de parámetros (coeficientes).
   - Contiene los pesos que el modelo aprende durante el entrenamiento.
   - $ \theta_0 $: Intercepto (valor predicho cuando todas las características son cero).
   - $ \theta_1, \theta_2, \dots $: Coeficientes que indican la influencia de cada característica en la predicción.

Breve descripción del proyecto: Implementación de operaciones básicas de matrices (suma, multiplicación por escalar, multiplicación de matrices, traspuesta) usando solo Python, junto con la explicación de conceptos fundamentales.

## Código Python (`Algebra_lineal.py`)

Este archivo contiene las funciones desarrolladas para realizar las operaciones matriciales solicitadas.

## Explicaciones Conceptuales
### 1. Vectores y Combinaciones Lineales

* **Vector:** Los vectores son listas ordenadas de números (componentes). Representan magnitudes y direcciones.
* **Combinación Lineal:** Consiste en multiplicar cada vector por un escalar (un número) y luego sumar los vectores resultantes.

### 2. Multiplicación de Matrices

* **¿Cómo funciona?:** Cada elemento de la matriz resultante se calcula mediante el producto punto de una fila de la primera matriz y una columna de la segunda matriz (se multiplican los elementos correspondientes y se suman los resultados).
* **¿Por qué importan las dimensiones?:** Son cruciales para determinar si la multiplicación es posible y cuál será el tamaño del resultado.
    * **Condición:** Para multiplicar A (m x n) por B (p x q), es necesario que `n = p` (el número de columnas de A debe ser igual al número de filas de B).
    * **Tamaño del Resultado:** Si la condición se cumple, la matriz resultado será de tamaño m x q (filas de A x columnas de B).
    * **Razón:** La condición `n = p` es necesaria para poder realizar la operación fila-por-columna (producto punto).
      
### 3. Traspuesta de una Matriz

* **¿Cómo se obtiene?:** Se obtiene intercambiando las filas por las columnas de la matriz original. La fila `i` se convierte en la columna `i` de la traspuesta, y la columna `j` se convierte en la fila `j`. Se puede visualizar como "voltear" la matriz sobre su diagonal principal.
* **¿Para qué sirve?:** Es útil para reorganizar la disposición de los datos, simplificar ciertas fórmulas matemáticas (por ejemplo, en estadística o al resolver sistemas de ecuaciones) y se usa frecuentemente en aplicaciones como Machine Learning (por ejemplo, para manipular vectores y matrices de pesos o calcular productos punto de manera conveniente).

### ¿Por qué la multiplicación de matrices NO es conmutativa en general (AB=BA)?
Porque las matrices representan transformaciones, representan rotaciones, etc. básicamente cuando multiplicas estas aplicando una transformación por lo tanto no puede ser conmutativa. También podemos pensar AxB sea posible pero tal vez BxA no, porque tal vez no sean compatibles (columna de la primera con filas de la segunda). Aun así en el caso de que sea posible AxB y BxA, puede que el tamaño resultante no sea el mismo.
### ¿Cuál es la intuición geométrica (si la hay) detrás de la traspuesta?
* Perpectiva: podriamos pensarlo como un cambio de perspectiva, como por ejemplo si tienes filas como si fuera una persona y columnas con caracterisiticas, si haces una traspuesta, ahora tendriamos filas con caracteriticas y personas como columnas que compartan ensas caracteristicas.
* Reflejo: si pudiermos pensarlo de manera espacial o en una tabla basicamente la traspuesta seria como reflejarla en su diagonal principal. 


# Preguntas de la Misión 1:

## ¿Por qué la multiplicación de matrices NO es conmutativa en general (AB != BA)?

**Porque** las matrices representan transformaciones, representan rotaciones, etc. **básicamente** cuando multiplicas estas aplicando una **transformación**, por lo tanto no puede ser conmutativa. **También** podemos pensar A\*B sea posible pero tal vez B\*A no, **porque** tal vez no sean compatibles (columna de la primera con filas de la segunda). **Aun así**, en el caso de que sea posible A\*B y B\*A, puede que el tamaño resultante no sea el mismo.

## ¿Cuál es la intuición geométrica (si la hay) detrás de la traspuesta?

* **Perspectiva:** **Podríamos** pensarlo como un cambio de perspectiva, como por **ejemplo** si **tienen** filas como si fueran una persona y columnas con **características**, si haces una traspuesta, ahora **tendríamos** filas con **características** y personas como columnas que compartan **esas** **características**.
* **Reflejo:** Si **pudiéramos** pensarlo de manera espacial o en una tabla, **básicamente** la traspuesta **sería** como reflejarla en su diagonal principal.

# Misión 2

## ¿Qué es PCA?

PCA, por sus siglas (**Análisis** de Componentes Principales), es una técnica que se usa para reducir la cantidad de variables en un conjunto de datos, sin perder demasiada información.

## ¿Para qué sirve?

* Nos permite visualizar datos complejos. Por ejemplo, podemos pasar de 4 dimensiones a 2 y graficarlos.
* **También** podemos eliminar variables que no aportan mucho o que no son tan importantes.

## ¿Cómo se elige cuántos componentes usar (el valor de k)?

Al hacer PCA para reducir dimensiones (por ejemplo, de 4 a `k`), la gran pregunta es: ¿cuántos componentes (`k`) debemos conservar para quedarnos con la información más importante sin perder demasiado? Aquí entran los conceptos de **varianza explicada** y **varianza acumulada**.

* **Varianza Explicada:** Cada componente principal (CP) "explica" un cierto porcentaje de la variación total de los datos. El primer CP explica la mayor parte, el segundo un poco menos, y así sucesivamente. Este porcentaje está directamente relacionado con el tamaño de su valor propio (eigenvalue) o su valor singular al cuadrado ($s^2$) comparado con la suma total de todos ellos. Es como preguntarse: "¿Cuánto de la 'historia completa' de los datos me cuenta esta dirección principal?"

* **Varianza Acumulada:** Es simplemente ir sumando los porcentajes de varianza explicada de los primeros componentes. Por ejemplo, la varianza acumulada por los 2 primeros CP es (Varianza del CP1) + (Varianza del CP2).

**Métodos para elegir `k`:**

1.  **Umbral de Varianza Acumulada (El más común):**
    * Decidimos qué porcentaje de la varianza total original queremos conservar (un valor típico es entre 90% y 99%, por ejemplo, 95%).
    * Calculamos la varianza explicada acumulada al usar 1 componente, luego 2, luego 3...
    * Elegimos el **menor número `k`** de componentes cuya varianza acumulada **alcance o supere** nuestro umbral (ej: el primer `k` que explique al menos el 95% de la varianza).

2.  **Método del "Codo" (Visual):**
    * Se grafica la varianza explicada por cada componente (ordenados de mayor a menor).
    * Se busca un punto en el gráfico donde la curva "se dobla" como un codo y empieza a aplanarse. El "codo" sugiere el punto donde añadir más componentes ya no aporta una cantidad significativa de información nueva. El valor de `k` se elige en ese codo.

## Pasos clave de mi implementación

1.  **Cargamos** el dataset de Iris, que tiene 4 variables por flor.
2.  **Centramos** los datos, restando la media de cada característica.
3.  **Calculamos** la matriz de covarianza:
    * Aquí calculamos la matriz de covarianza, que **básicamente** nos dice cómo se mueven las variables entre sí. Es como una tabla que nos muestra si dos variables tienden a aumentar o disminuir juntas. Es clave para ver las relaciones entre las características.
4.  **Calculamos** los valores propios y vectores propios:
    * Los valores propios nos dicen **cuánta** "importancia" tiene cada dirección de variación.
    * Los vectores propios nos indican en qué dirección ocurren esas variaciones.
5.  **Ordenamos** los vectores propios de mayor a menor según la varianza que explican. Así sabemos qué direcciones son las más importantes y capturan la mayor parte de la información.
6.  **Seleccionamos** los dos primeros componentes principales para proyectar los datos, y así poder visualizarlos.
7.  **Aquí utilizamos** una alternativa usando SVD (Descomposición en Valores Singulares):
    * Como alternativa, usamos SVD, que es otra forma de descomponer los datos. En lugar de calcular la matriz de covarianza, directamente obtenemos las direcciones principales usando SVD. Los vectores fila de `Vh` nos dan las direcciones principales.
8.  **Proyección final alternativa usando SVD**:
    * Proyectamos los datos usando los primeros dos componentes principales que conseguimos a través de SVD. Esto también nos reduce la dimensionalidad y nos permite ver los datos en 2D.
9.  **Visualización**:
    * Finalmente, **graficamos** los datos proyectados en 2D, usando los dos componentes principales seleccionados. Esto nos da una visualización más sencilla de los datos, para poder ver patrones o relaciones.

## Conexión: Matriz de Covarianza y Varianza

Pensemos en los **números** de la diagonal principal (de arriba a la izquierda hacia abajo a la derecha), estos son la **varianza**, te dicen **cuánto varía** cada variable por sí sola; si es grande, cambia mucho entre los datos.
Los **números** que quedan fuera de la diagonal son la **covarianza** entre pares de variables, estas nos dicen **cómo varían** juntas dos **características**.

* Si el número es positivo y grande, cuando una sube, la otra también tiende a subir.
* Si es negativo, cuando una sube, la otra baja.
* Si es cerca de cero, no hay mucha relación.

### ¿Por qué nos ayudan a entender la estructura de datos?

**Porque** nos dan pista de **cómo están** distribuidos los datos y sus relaciones.

* Si una varianza es muy grande, quiere decir que esa variable es muy dispersa, cambia mucho entre muestra y muestra. **Tal vez** sea importante.
* Si una covarianza fuera de la diagonal es grande y positiva, significa que esas dos variables **están** relacionadas, se mueven “juntas”. Eso es útil porque **podríamos** reducir dimensiones, ya que **están** diciendo cosas parecidas.

### BÁSICAMENTE

* **Qué** variables cambian mucho (varianzas).
* **Qué** variables **están** conectadas entre sí (covarianzas).

## Conexión: Eigenvectores/Eigenvalores y Varianza

Los **vectores propios** son como flechas que te dicen hacia dónde se **extienden** los datos, o sea, por donde se dispersan. Los **valores propios** nos dicen **cuánta variación** hay en la **dirección** de su **vector propio** correspondiente.

* Si el valor propio es grande, hay mucha varianza (los datos **están** muy esparcidos en esa dirección).
* Si es pequeño, hay poca varianza (los datos **están** más concentrados).

El **vínculo** es: los vectores propios de la matriz de covarianza nos dan las **direcciones** donde la varianza es máxima, y los valores propios nos dicen **cuánta** varianza hay en cada una de esas direcciones.
En resumen: Los **vectores propios** te dicen por dónde se **están** moviendo más los datos. Los **valores propios** te dicen **cuánto** se **están** moviendo por esas direcciones.

## Conexión: SVD y Varianza

Cuando usamos `U, s, Vh = np.linalg.svd(X_centrado)` estamos haciendo algo muy parecido a lo que hicimos con la matriz de covarianza, pero más directo y más estable.
* La U nos dice **cómo** se ven los datos originales sobre las nuevas direcciones (Vh).
* Las filas de Vh son las mismas direcciones principales que **habíamos** encontrado con los vectores propios, o sea, por dónde más se esparcen los datos.
* Los valores de s (valores singulares) **están** ligados a la varianza:
    * Si haces $s^2$ (s al cuadrado), eso te da una idea de **cuánta** varianza hay en cada dirección.

## En resumen (SVD):

* Vh: Hacia dónde mirar (las direcciones principales).
* s: **Cuánta** importancia tiene cada dirección (relacionado con la varianza a través de $s^2$).
* U: **Cómo** cada punto del dataset se ve desde esas nuevas direcciones.

SVD te da otra forma (más precisa y directa) de encontrar esas direcciones principales importantes (Vh) y **cuánta** info hay en cada una ($s^2$). Es como hacer PCA, pero sin tener que **calcular** la matriz de covarianza.

## Cómo Elegir k (Número de Componentes)

Cuando hacemos PCA necesitamos preguntarnos "**¿cuántas direcciones** necesito para obtener lo **más** importante?". **Aquí** entramos en los **conceptos** de **varianza explicada** y **varianza acumulada**.

* **Varianza explicada:** Cada componente principal (esas nuevas direcciones que PCA encuentra) explica una parte de la variación total que hay en tus datos. Es como decir: "**¿Cuánto** de la info original me **está** mostrando esta **dirección**?"
* **Varianza acumulada:** Es la suma de la varianza explicada de los primeros `k` componentes. Nos ayuda a saber **cuánta información** estamos obteniendo si usamos solo `k` componentes.

## ¿Cómo saber cuántos componentes necesito para conservar, por ejemplo, el 95% de la info (varianza)?

Primero, **calculás** la varianza que explica cada componente, y **después** vas sumando una por una (eso se llama varianza acumulada).

* **Si usás los `valores_propios` (eigenvalues):**
    * **Sumás** todos los valores propios, eso te da la varianza total.
    * Para cada componente, **dividís** su valor propio entre la varianza total; eso te da el porcentaje de varianza que explica ese componente.
    * Vas sumando esos porcentajes hasta que llegues o pases el umbral del 95%.
* **Si usás los `s` de SVD (valores singulares):**
    * **Elevás** al cuadrado cada número de `s` ($s^2$), eso te dice **cuánta** "varianza" (información) tiene cada componente.
    * **Sumás** todos esos cuadrados; eso te da la información total que hay en los datos.
    * **Después**, para cada componente, **dividís** su valor al cuadrado ($s^2$) entre la suma total → así ves qué porcentaje de info aporta ese componente.
    * Vas sumando esos porcentajes, uno por uno, hasta que la suma llegue al 95%.


# Explicaciones Tarea 3

## Descripción de las operaciones realizadas con Pandas

En la primera parte del script, **utilicé** la **librería** Pandas para realizar varias operaciones de **manipulación** y **análisis** sobre el conjunto de datos de **Iris**. A **continuación**, se **describen** las **principales** operaciones que **realicé**:

* **Carga de datos:** **Utilicé** el **método** `pd.read_csv()` para cargar el archivo `Iris.csv` en un DataFrame llamado `df`. Este DataFrame contiene las **características** de las flores de **Iris**.

* **Visualización de las primeras y últimas filas:** **Usé** `df.head()` para mostrar las primeras 5 filas del DataFrame y obtener una vista **rápida** de los primeros registros. **También** **utilicé** `df.tail()` para ver las **últimas** 5 filas y verificar **cómo** **terminan** los datos.

* **Resumen del DataFrame:** Con `df.info()`, **obtuve** un resumen general del DataFrame, incluyendo el **número** de entradas, el **tipo** de datos de cada columna y la cantidad de valores no nulos.

* **Estadísticas descriptivas:** **Utilicé** `df.describe()` para obtener **estadísticas** descriptivas de las columnas **numéricas** del DataFrame, como la media, **desviación** **estándar**, **mínimos**, **máximos** y percentiles. Posteriormente, **redondeé** los resultados a tres decimales con `.round(3)`.

* **Conteo de categorías:** Para saber **cuántas** veces **aparece** cada valor en la columna `Species`, **utilicé** `df['Species'].value_counts()`, lo que me **dio** la **distribución** de las especies en el conjunto de datos.

* **Selección de filas y columnas por posición:** **Usé** `df.iloc[-1, [1, 3]]` para seleccionar la **última** fila y las columnas en las **posiciones** 1 y 3 (correspondientes a `SepalWidthCm` y `PetalWidthCm`).

* **Selección de filas y columnas por etiquetas:** Con `df.loc[149, ['SepalWidthCm', 'PetalWidthCm']]`, **seleccioné** la fila con el **índice** 149 y las columnas `SepalWidthCm` y `PetalWidthCm` por sus nombres.

* **Filtrado de datos con condiciones combinadas:** **Apliqué** un filtro combinado con `df[(df['Species'] == 'Iris-setosa') & (df['SepalLengthCm'] < 5)]` para seleccionar las filas donde la **especie** es `Iris-setosa` y la **longitud** del **sépalo** es menor a 5.

Estas **operaciones** son fundamentales para explorar y entender los datos antes de pasar a las siguientes etapas del **análisis** o **modelado**.

## Naive Bayes

Podemos pensar en Naive Bayes como un detective que trata de resolver un caso: descubrir qué tipo de flor es una nueva flor que encontró, basándose en la evidencia (sus medidas) y su experiencia previa.

### Pasos del Algoritmo (Implementación)

1.  **Cargar Datos:** Es muy importante empezar importando los datos necesarios, en este caso, usando `load_iris()`. A este resultado le damos un nombre (ej: `datos`). Accedemos a los datos de dos maneras principales:
    * `.data` (guardado en `X`): Contiene las características numéricas (las 4 medidas de cada flor).
    * `.target` (guardado en `Y`): Contiene las etiquetas de clase (0, 1 o 2) para cada flor.

2.  **Separar Datos por Clase:** Tomamos la matriz `X` y, usando las etiquetas `Y`, la separamos en tres grupos: uno para cada clase (0, 1 y 2). Así tenemos `X_clase0`, `X_clase1`, `X_clase2`.

3.  **Calcular Estadísticas ("Entrenamiento"):** Una vez separados los datos por clase, calculamos para cada una:
    * **Media (`np.mean(..., axis=0)`):** El valor promedio de cada una de las 4 características para esa clase específica.
    * **Desviación Estándar (`np.std(..., axis=0)`):** Cuánto varían o se dispersan las medidas de cada característica alrededor de su media, para esa clase.
    * **Probabilidad Previa (Prior):** Qué tan común es cada especie en el conjunto total de datos (calculado como: número de flores de la clase / número total de flores).

4.  **Agrupar Estadísticas:** Guardamos todas las medias, desviaciones estándar y priors calculados en listas (`lista_medias`, `lista_stds`, `lista_priors`) para usarlas fácilmente en la predicción.

5.  **Función `gaussian_pdf`:** Definimos una función auxiliar importante. Esta calcula la Densidad de Probabilidad Gaussiana: dado un valor `x`, una media `mu` y una desviación estándar `std`, nos dice qué tan "probable" o "típico" es ese valor `x` si perteneciera a una distribución normal (Campana de Gauss) con esa `mu` y `std`. Es como preguntarle a la campana: "¿Qué altura tenés en este punto `x`?".

6.  **Función `predecir_clases_nb` (Predicción):** Esta es la función principal. Recibe una flor nueva (`flor_nueva`) y las listas de estadísticas. Para decidir la clase:
    * Crea una lista vacía (`posteriors`) para guardar los "scores".
    * Usa un bucle `for` para recorrer cada clase posible (0, 1, 2).
    * **Dentro del bucle:**
        * Obtiene las estadísticas (`media_actual`, `std_actual`, `prior_actual`) de la clase actual.
        * Calcula el **Likelihood**: Usa `gaussian_pdf` para obtener la probabilidad de cada una de las 4 características de la `flor_nueva` según la media y std de la clase actual. Luego, multiplica estas 4 probabilidades (¡la asunción "naive"!) usando `np.prod` para obtener la probabilidad total de observar esas características si la flor fuera de esta clase. (Nota: en el código final usamos logaritmos para evitar problemas numéricos, sumando log(PDFs) en lugar de multiplicar PDFs).
        * Calcula el **Score Posterior**: Multiplica el `likelihood` por el `prior_actual` (o suma sus logaritmos).
        * Guarda este `posterior_actual` en la lista `posteriors`.
    * **Después del bucle:** Compara los 3 scores guardados en `posteriors` y elige el **índice** (0, 1 o 2) del score más alto usando `np.argmax`.
    * Devuelve ese índice como la clase predicha.

7.  **Probar Nuestro Clasificador:**
    * Hacemos un bucle que recorre todas las flores del dataset original (`X`).
    * Para cada flor, llamamos a `predecir_clases_nb` para obtener su predicción.
    * Guardamos todas las predicciones.
    * Calculamos la **Precisión (Accuracy)**: comparamos nuestras predicciones con las etiquetas reales (`Y`) y vemos el porcentaje de aciertos.

8.  **Comparar con Scikit-learn:**
    * Usamos la implementación `GaussianNB` de Scikit-learn, la entrenamos (`fit`) y predecimos (`predict`) con los mismos datos `X` e `Y`.
    * Calculamos su precisión para tener una referencia y ver si nuestro modelo manual da resultados similares.


## Conceptos Clave Relacionados

### Teorema de Bayes (Regla General)

* **¿Qué es?:** Es una fórmula matemática para actualizar una probabilidad o creencia inicial (`Prior`) basándonos en nueva evidencia (`Datos`) para obtener una probabilidad final (`Posterior`).
* **Idea:** $P(\text{Clase } | \text{ Datos}) \propto P(\text{Datos } | \text{ Clase}) \times P(\text{Clase})$ (El posterior es proporcional al likelihood por el prior).
* **En resumen:** Nos dice cómo combinar lo que ya sabíamos con la nueva evidencia.

### Naive Bayes (El Algoritmo de Clasificación)

* **¿Qué es?:** Un método de clasificación que **usa** el Teorema de Bayes para decidir a qué clase pertenece una nueva muestra.
* **La parte "Naive" (Ingenua):** Su característica principal es que **asume (ingenuamente)** que todas las características (las 4 medidas de la flor) son **independientes entre sí** dada la clase. Esto simplifica muchísimo el cálculo del Likelihood ($P(Datos|Clase)$), permitiendo multiplicar las probabilidades individuales de cada característica (o sumar sus logaritmos).
* **En resumen:** Aplica Bayes con una simplificación clave (independencia) para clasificar.

### Campana de Gauss (Distribución Normal)

* **¿Qué es?:** Una forma matemática muy común (la curva en forma de campana) que describe cómo se distribuyen muchos datos numéricos alrededor de una media, con una cierta dispersión (desviación estándar).
* **En Gaussian Naive Bayes:** Hacemos la **suposición** de que las características numéricas *dentro de cada clase* siguen esta distribución Gaussiana. Esto nos permite usar la fórmula de la Campana de Gauss (nuestra función `gaussian_pdf`) para calcular las probabilidades $P(\text{característica } k | \text{ Clase})$ que necesitamos para el Likelihood.
* **En resumen:** Es el modelo de probabilidad que usamos para las características numéricas en esta versión específica de Naive Bayes.

### Resumen de la Relación

El **Teorema de Bayes** nos da el marco general. El algoritmo **Naive Bayes** lo aplica para clasificar, añadiendo la suposición "naive" de independencia. La **Campana de Gauss** es la herramienta que usamos en *Gaussian* Naive Bayes para calcular las probabilidades de las características numéricas dentro de ese marco.
