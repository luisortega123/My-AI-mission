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
    * `.data` (guardado en `$X$`): Contiene las características numéricas (las 4 medidas de cada flor).
    * `.target` (guardado en `$Y$`): Contiene las etiquetas de clase (0, 1 o 2) para cada flor.

2.  **Separar Datos por Clase:** Tomamos la matriz `$X$` y, usando las etiquetas `$Y$`, la separamos en tres grupos: uno para cada clase (0, 1 y 2). Así tenemos `X_clase0`, `X_clase1`, `X_clase2`.

3.  **Calcular Estadísticas ("Entrenamiento"):** Una vez separados los datos por clase, calculamos para cada una:
    * **Media (`np.mean(..., axis=0)`):** El valor promedio de cada una de las 4 características para esa clase específica.
    * **Desviación Estándar (`np.std(..., axis=0)`):** Cuánto varían o se dispersan las medidas de cada característica alrededor de su media, para esa clase.
    * **Probabilidad Previa (Prior):** Qué tan común es cada especie en el conjunto total de datos (calculado como: número de flores de la clase / número total de flores).

4.  **Agrupar Estadísticas:** Guardamos todas las medias, desviaciones estándar y priors calculados en listas (`lista_medias`, `lista_stds`, `lista_priors`) para usarlas fácilmente en la predicción.

5.  **Función `gaussian_pdf`:** Definimos una función auxiliar importante. Esta calcula la Densidad de Probabilidad Gaussiana: dado un valor `$x$`, una media `$mu$` y una desviación estándar `$std$`, nos dice qué tan "probable" o "típico" es ese valor `$x$` si perteneciera a una distribución normal (Campana de Gauss) con esa `$mu$` y `$std$`. Es como preguntarle a la campana: "¿Qué altura tenés en este punto `$x$`?".

6.  **Función `predecir_clases_nb` (Predicción):** Esta es la función principal. Recibe una flor nueva (`flor_nueva`) y las listas de estadísticas. Para decidir la clase:
    * Crea una lista vacía (`posteriors`) para guardar los "scores".
    * Usa un bucle `for` para recorrer cada clase posible (0, 1, 2).
    * **Dentro del bucle:**
        * Obtiene las estadísticas (`media_actual`, `std_actual`, `prior_actual`) de la clase actual.
        * Calcula el **Likelihood**: Usa `gaussian_pdf` para obtener la probabilidad de cada una de las 4 características de la `flor_nueva` según la media y `$std$` de la clase actual. Luego, multiplica estas 4 probabilidades (¡la asunción "naive"!) usando `np.prod` para obtener la probabilidad total de observar esas características si la flor fuera de esta clase. (Nota: en el código final usamos logaritmos para evitar problemas numéricos, sumando $\log(\text{PDFs})$ en lugar de multiplicar PDFs).
        * Calcula el **Score Posterior**: Multiplica el `likelihood` por el `prior_actual` (o suma sus logaritmos).
        * Guarda este `posterior_actual` en la lista `posteriors`.
    * **Después del bucle:** Compara los 3 scores guardados en `posteriors` y elige el **índice** (0, 1 o 2) del score más alto usando `np.argmax`.
    * Devuelve ese índice como la clase predicha.

7.  **Probar Nuestro Clasificador:**
    * Hacemos un bucle que recorre todas las flores del dataset original (`$X$`).
    * Para cada flor, llamamos a `predecir_clases_nb` para obtener su predicción.
    * Guardamos todas las predicciones.
    * Calculamos la **Precisión (Accuracy)**: comparamos nuestras predicciones con las etiquetas reales (`$Y$`) y vemos el porcentaje de aciertos.

8.  **Comparar con Scikit-learn:**
    * Usamos la implementación `GaussianNB` de Scikit-learn, la entrenamos (`fit`) y predecimos (`predict`) con los mismos datos `$X$` e `$Y$`.
    * Calculamos su precisión para tener una referencia y ver si nuestro modelo manual da resultados similares.


## Conceptos Clave Relacionados

### Teorema de Bayes (Regla General)

* **¿Qué es?:** Es una fórmula matemática para actualizar una probabilidad o creencia inicial (`Prior`) basándonos en nueva evidencia (`Datos`) para obtener una probabilidad final (`Posterior`).
* **Idea:** $P(\text{Clase} | \text{Datos}) \propto P(\text{Datos} | \text{Clase}) \times P(\text{Clase})$ (El posterior es proporcional al likelihood por el prior).
* **En resumen:** Nos dice cómo combinar lo que ya sabíamos con la nueva evidencia.

### Naive Bayes (El Algoritmo de Clasificación)

* **¿Qué es?:** Un método de clasificación que **usa** el Teorema de Bayes para decidir a qué clase pertenece una nueva muestra.
* **La parte "Naive" (Ingenua):** Su característica principal es que **asume (ingenuamente)** que todas las características (las 4 medidas de la flor) son **independientes entre sí** dada la clase. Esto simplifica muchísimo el cálculo del Likelihood ($P(\text{Datos}|\text{Clase})$), permitiendo multiplicar las probabilidades individuales de cada característica (o sumar sus logaritmos).
* **En resumen:** Aplica Bayes con una simplificación clave (independencia) para clasificar.

### Campana de Gauss (Distribución Normal)

* **¿Qué es?:** Una forma matemática muy común (la curva en forma de campana) que describe cómo se distribuyen muchos datos numéricos alrededor de una media, con una cierta dispersión (desviación estándar).
* **En Gaussian Naive Bayes:** Hacemos la **suposición** de que las características numéricas *dentro de cada clase* siguen esta distribución Gaussiana. Esto nos permite usar la fórmula de la Campana de Gauss (nuestra función `gaussian_pdf`) para calcular las probabilidades $P(\text{característica}_k | \text{Clase})$ que necesitamos para el Likelihood.
* **En resumen:** Es el modelo de probabilidad que usamos para las características numéricas en esta versión específica de Naive Bayes.

### Resumen de la Relación

El **Teorema de Bayes** nos da el marco general. El algoritmo **Naive Bayes** lo aplica para clasificar, añadiendo la suposición "naive" de independencia. La **Campana de Gauss** es la herramienta que usamos en *Gaussian* Naive Bayes para calcular las probabilidades de las características numéricas dentro de ese marco.
