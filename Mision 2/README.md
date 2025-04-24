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

*(Nota: Aquí faltaría tu explicación de cómo elegir k, podrías añadirla)*

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