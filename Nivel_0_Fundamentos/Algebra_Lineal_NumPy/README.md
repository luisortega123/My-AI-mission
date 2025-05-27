# Álgebra Lineal con Python Puro

Breve descripción del proyecto: Implementación de operaciones básicas de matrices (suma, multiplicación por escalar, multiplicación de matrices, traspuesta) usando únicamente Python puro, junto con explicaciones de los conceptos fundamentales involucrados.

## 📂 Código Python (`Algebra_lineal.py`)

Este archivo contiene las funciones desarrolladas para realizar las operaciones matriciales solicitadas: suma de matrices, multiplicación por escalar, multiplicación de matrices y trasposición.

## 📘 Explicaciones Conceptuales

### 1. Vectores y Combinaciones Lineales

- **Vector:** Una lista ordenada de números (componentes) que representa una magnitud con dirección en un espacio.
- **Combinación Lineal:** Es una operación en la que cada vector se multiplica por un escalar y luego se suman los vectores resultantes.

### 2. Multiplicación de Matrices

* **¿Cómo funciona?:** Cada elemento de la matriz resultante se calcula mediante el producto punto de una fila de la primera matriz y una columna de la segunda matriz (se multiplican los elementos correspondientes y se suman los resultados).
* **¿Por qué importan las dimensiones?:** Son cruciales para determinar si la multiplicación es posible y cuál será el tamaño del resultado.
    * **Condición:** Para multiplicar A (m x n) por B (p x q), es necesario que `n = p` (el número de columnas de A debe ser igual al número de filas de B).
    * **Tamaño del Resultado:** Si la condición se cumple, la matriz resultado será de tamaño m x q (filas de A x columnas de B).
    * **Razón:** La condición `n = p` es necesaria para poder realizar la operación fila-por-columna (producto punto).
      
### 3. Traspuesta de una Matriz

- **¿Cómo se obtiene?:** Se intercambian filas por columnas. Es decir, la fila \( i \) de la matriz original se convierte en la columna \( i \) de la traspuesta.
  
- **¿Para qué sirve?:** Facilita reorganizar datos, simplificar fórmulas matemáticas y es muy usada en áreas como Machine Learning para manipular vectores, pesos y operaciones matriciales..

### ❓ ¿Por qué la multiplicación de matrices *no* es conmutativa en general (\( AB \neq BA \))?

Las matrices representan transformaciones (como rotaciones o escalados), y aplicar una transformación seguida de otra no necesariamente da el mismo resultado si se invierte el orden. Además:

- Puede que \( A \times B \) sea posible pero \( B \times A \) no, debido a la incompatibilidad de dimensiones.
- Incluso si ambas multiplicaciones son posibles, el tamaño del resultado puede ser diferente.
- Y aun si el tamaño coincide, el contenido generalmente **no será el mismo**.

### ❓ ¿Cuál es la intuición geométrica detrás de la traspuesta?

- **Perspectiva:** Se puede ver como un cambio de enfoque: si las filas representan personas y las columnas características, al trasponer la matriz, ahora las filas representan características y las columnas personas que comparten esas características.
  
- **Reflejo:** Visualmente, es como reflejar la matriz sobre su **diagonal principal**, intercambiando filas por columnas.


# Preguntas de la Misión 1:

## ¿Por qué la multiplicación de matrices NO es conmutativa en general (AB != BA)?

**Porque** las matrices representan transformaciones, representan rotaciones, etc. **básicamente** cuando multiplicas estas aplicando una **transformación**, por lo tanto no puede ser conmutativa. **También** podemos pensar A\*B sea posible pero tal vez B\*A no, **porque** tal vez no sean compatibles (columna de la primera con filas de la segunda). **Aun así**, en el caso de que sea posible A\*B y B\*A, puede que el tamaño resultante no sea el mismo.

### ¿Cuál es la intuición geométrica (si la hay) detrás de la traspuesta?

* **Perspectiva:** Podemos pensarlo como un cambio de enfoque. Por ejemplo, si las filas representan personas y las columnas características, al trasponer la matriz, las filas pasarían a representar las características, y las colvumnas a las personas que las poseen.
* **Reflejo:** También puede visualizarse como reflejar la matriz en su diagonal principal, intercambiando filas por columnas.

## ¿Qué es PCA?

PCA, por sus siglas **(Análisis de Componentes Principales)**, es una técnica que se usa para reducir la cantidad de variables en un conjunto de datos, sin perder demasiada información.

## ¿Para qué sirve?

* Nos permite **visualizar datos complejos.** Por ejemplo, podemos pasar de 4 dimensiones a 2 y graficarlos.
* **También** podemos eliminar variables que no aportan mucho o que no son tan importantes.

## ¿Cómo se elige cuántos componentes usar (el valor de $k$)?

Al hacer PCA para reducir dimensiones (por ejemplo, de 4 a $k$), la gran pregunta es: ¿cuántos componentes ($k$) debemos conservar para quedarnos con la información más importante sin perder demasiado? Aquí entran los conceptos de **varianza explicada** y **varianza acumulada**.

* **Varianza Explicada:** Cada componente principal (CP) "explica" un cierto porcentaje de la variación total de los datos. El primer CP explica la mayor parte, el segundo un poco menos, y así sucesivamente. Este porcentaje está directamente relacionado con el tamaño de su valor propio (eigenvalue) o su valor singular al cuadrado ($s^2$) comparado con la suma total de todos ellos. Es como preguntarse: "¿Cuánto de la 'historia completa' de los datos me cuenta esta dirección principal?"

* **Varianza Acumulada:** Es simplemente ir sumando los porcentajes de varianza explicada de los primeros componentes. Por ejemplo, la varianza acumulada por los 2 primeros CP es (Varianza del CP1) + (Varianza del CP2).

**Métodos para elegir $k$:**

1.  **Umbral de Varianza Acumulada (El más común):**
    * Decidimos qué porcentaje de la varianza total original queremos conservar (un valor típico es entre 90% y 99%, por ejemplo, 95%).
    * Calculamos la varianza explicada acumulada al usar 1 componente, luego 2, luego 3...
    * Elegimos el **menor número $k$** de componentes cuya varianza acumulada **alcance o supere** nuestro umbral (ej: el primer $k$ que explique al menos el 95% de la varianza).

2.  **Método del "Codo" (Visual):**
    * Se grafica la varianza explicada por cada componente (ordenados de mayor a menor).
    * Se busca un punto en el gráfico donde la curva "se dobla" como un codo y empieza a aplanarse. El "codo" sugiere el punto donde añadir más componentes ya no aporta una cantidad significativa de información nueva. El valor de $k$ se elige en ese codo.

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

* Qué variables **cambian mucho** (varianzas).
* Qué variables **están** conectadas entre sí (covarianzas).

## Conexión: Eigenvectores/Eigenvalores y Varianza

Los **vectores propios** son como flechas que te dicen hacia dónde se **extienden** los datos, o sea, por donde se dispersan. Los **valores propios** nos dicen **cuánta variación** hay en la **dirección** de su **vector propio** correspondiente.

* Si el valor propio es grande, hay mucha varianza (los datos **están** muy esparcidos en esa dirección).
* Si es pequeño, hay poca varianza (los datos **están** más concentrados).

El **vínculo** es: los vectores propios de la matriz de covarianza nos dan las **direcciones** donde la varianza es máxima, y los valores propios nos dicen **cuánta** varianza hay en cada una de esas direcciones.
En resumen: Los **vectores propios** te dicen por dónde se **están** moviendo más los datos. Los **valores propios** te dicen **cuánto** se **están** moviendo por esas direcciones.

## Conexión: SVD y Varianza

Cuando usamos `U, s, Vh = np.linalg.svd(X_centrado)` estamos haciendo algo muy parecido a lo que hicimos con la matriz de covarianza, pero más directo y más estable.
* La U nos dice **cómo** se ven los datos originales sobre las nuevas direcciones (`Vh`).
* Las filas de `Vh` son las mismas direcciones principales que **habíamos** encontrado con los vectores propios, o sea, por dónde más se esparcen los datos.
* Los valores de `s` (valores singulares) **están** ligados a la varianza:
    * Si haces $s^2$ (s al cuadrado), eso te da una idea de **cuánta** varianza hay en cada dirección.

## En resumen (SVD):

* `Vh`: Hacia dónde mirar (las direcciones principales).
* `s`: **Cuánta** importancia tiene cada dirección (relacionado con la varianza a través de $s^2$).
* `U`: **Cómo** cada punto del dataset se ve desde esas nuevas direcciones.

SVD te da otra forma (más precisa y directa) de encontrar esas direcciones principales importantes (`Vh`) y **cuánta** info hay en cada una ($s^2$). Es como hacer PCA, pero sin tener que **calcular** la matriz de covarianza.

## Cómo Elegir $k$ (Número de Componentes)

Cuando hacemos PCA necesitamos preguntarnos "**¿cuántas direcciones** necesito para obtener lo **más** importante?". **Aquí** entramos en los **conceptos** de **varianza explicada** y **varianza acumulada**.

* **Varianza explicada:** Cada componente principal (esas nuevas direcciones que PCA encuentra) explica una parte de la variación total que hay en tus datos. Es como decir: "**¿Cuánto** de la info original me **está** mostrando esta **dirección**?"
* **Varianza acumulada:** Es la suma de la varianza explicada de los primeros $k$ componentes. Nos ayuda a saber **cuánta información** estamos obteniendo si usamos solo $k$ componentes.

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

## Puedes explicar con una analogía simple o geométrica qué representan los componentes principales? ¿Cómo se relaciona la pérdida de información con la reducción de dimensiones?

**Componentes principales:**
Los componentes principales pueden entenderse como nuevas direcciones o ejes que nos permiten describir los datos de manera más compacta.
Una forma de verlo es imaginar que las fotos representan información. Lo que los componentes principales buscan es organizar esta información de manera que no se pierda demasiado detalle.

En términos visuales, podríamos pensar en dibujar una línea imaginaria que pase por el punto de mayor dispersión de los datos. Esta línea representaría el primer componente principal, la dirección con mayor varianza. Es decir, la línea que captura la mayor parte de la variación en los datos.

Después, podemos dibujar una segunda línea que debe ser ortogonal a la primera. Esta segunda línea captura la mayor cantidad restante de información, es decir, la segunda mayor varianza. Así sucesivamente para cada componente principal.

**Cómo se relaciona la pérdida de información con la reducción de dimensiones:**
Reducir dimensiones es comparable a resumir una historia: retienes lo más importante, pero inevitablemente se pierde parte de la información.
Siguiendo la analogía de los datos como una nube de puntos, el primer componente principal captura la mayor parte de la información contenida en los datos. Sin embargo, como se descartan los componentes restantes, se pierde información.
Esta información perdida corresponde a la varianza explicada por los componentes principales que decidimos no conservar.