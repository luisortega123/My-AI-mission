# 🧠 Implementación de SVM con Scikit-learn

## ✅ Definición de SVM:
Una Máquina de Vectores de Soporte (SVM) es un modelo de aprendizaje supervisado que encuentra el hiperplano óptimo que maximiza el margen entre las clases en un conjunto de datos etiquetados.

## ✂️ Clasificador de Máximo Margen

El **Clasificador de Máximo Margen** es el núcleo del algoritmo **SVM (Support Vector Machine)**. Su misión es clara:
👉 **Encontrar el hiperplano que separa dos clases con el mayor margen posible**, es decir, con la máxima “zona de seguridad” entre los puntos de cada clase.

---

## 🔵🔴 Analogía intuitiva: separación entre dos grupos

Imagina dos grupos de puntos en un plano:

* 🔴 Grupo de puntos **rojos**
* 🔵 Grupo de puntos **azules**

Queremos trazar una **línea de separación** que divida ambos grupos correctamente. En SVM, esta línea se llama:

### ✳️ **Hiperplano Separador**

> Aunque en 2D es una línea, en dimensiones superiores es un **hiperplano** (una superficie que generaliza la noción de línea recta).

Este **hiperplano** funciona como una **navaja** que corta el espacio en dos mitades.
🛡️ Cuanto más lejos esté de los puntos cercanos de cada clase, más **robusta** será esta separación.


## 🧱 ¿Qué son las fronteras de decisión o el Hiperplano en SVM?

### 📌 Definición sencilla:

> **Una frontera de decisión es la línea (o superficie) que separa las distintas clases en los datos.**

En el caso de una **SVM lineal en 2D**, es una **línea recta**.
En una SVM no lineal (usando *kernels*), puede ser una **curva, superficie o forma más compleja**.

---

## 🎯 ¿Por qué es importante?

La **frontera de decisión es lo que el modelo aprende**. Su objetivo es:

* Separar las clases de forma que **se maximice el margen** entre ellas.
* **Predecir la clase de nuevos puntos** según de qué lado de la frontera caen.

---

## 🖼️ Intuición visual (2D)

Imaginá dos grupos de puntos:

🔴 Rojos
🔵 Azules

Una SVM encuentra la **línea (hiperplano en general)** que los separa **dejando el mayor espacio posible entre ambos grupos**.

Así:

```
🔴🔴🔴         🔵🔵🔵
🔴🔴🔴  |     🔵🔵🔵
🔴🔴🔴         🔵🔵🔵
        ↑
     Frontera
     de decisión
```

* Todo lo que queda a la izquierda → se predice como clase roja.
* Todo lo que queda a la derecha → se predice como clase azul.

---

## 🧠 Características clave de la frontera en SVM

| Característica                    | Explicación                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| **Lineal o no lineal**            | Depende del kernel: lineal → hiperplano; no lineal → curva/superficie    |
| **Definida por vectores soporte** | Solo los puntos más cercanos (vectores soporte) influyen en su posición  |
| **Máximo margen**                 | SVM la posiciona para maximizar la distancia con los puntos más cercanos |

---

## 🔁 Frontera de decisión vs. Márgenes

* **Frontera de decisión**: la línea central que divide las clases.
* **Márgenes**: las dos líneas paralelas que pasan por los vectores soporte (uno por cada clase).

```
   margen         frontera         margen
   -----        | (decisión)        -----
🔴🔴🔴🔴        |                🔵🔵🔵🔵
```


---

## 📏 ¿Qué es el **margen**?

🧩 El **margen** es la distancia entre el hiperplano separador y los puntos más cercanos de cada clase.
Un margen amplio significa:

* ✅ Mayor tolerancia al ruido
* ✅ Mayor robustez ante nuevos datos
* ✅ Mejor generalización

🎯 **El objetivo de SVM es maximizar este margen**.

---

## 🧷 ¿Qué son los **vectores de soporte**?

Los **vectores de soporte** son:

* 🔺 Los puntos **más cercanos** al hiperplano
* 🧱 Los que **definen los bordes del margen**
* 🧭 Los que **determinan la posición exacta del hiperplano**

🛠️ Si mueves uno de estos puntos, el hiperplano y el margen cambiarán.
Los demás puntos no tienen influencia directa.

> 📌 Son literalmente los que “soportan” (support) el hiperplano.

---

## 🛣️ Visualización mental: una calle

Visualiza esto como una **calle**:

* 🟢 El **hiperplano separador** es la **línea central** de la calle
* 🟥 y 🟦 Los **vectores de soporte** están en **los bordes** de la calle
* 📐 El **margen** es el **ancho total** de la calle

📎 El mejor hiperplano es el que está **a la misma distancia** de los puntos rojos y azules más cercanos.

---

## 🧾 En resumen

El **Clasificador de Máximo Margen** busca:

✅ Un **hiperplano** que separe correctamente las clases
📐 Que esté **equidistante** de los vectores de soporte de cada clase
📏 Que **maximice el margen** para mayor estabilidad
🔧 Solo los vectores de soporte afectan su posición

💡 Esta estrategia ofrece una **separación óptima y robusta**, ideal para evitar errores con nuevos datos.



## 🌀 ¿Qué sucede si los datos son más complejos?

Imagina una nueva situación más desafiante:

* 🔵 Tienes un grupo de **puntos azules en el centro** de tu hoja de papel.
* 🔴 Y un grupo de **puntos rojos formando un anillo alrededor** de los puntos azules.

🎨 En este escenario, una simple línea recta (hiperplano en 2D) **no puede separarlos** correctamente.

---

## 🧠 Aquí es donde entra la **idea genial de los *Kernels*** ✨

¿Qué pasaría si pudieras:

* 💫 Tomar tus datos en ese plano 2D (la hoja de papel),
* 🔁 Y **transformarlos** mágicamente a un **espacio con más dimensiones**,
* Donde sí se puedan **separar con un plano recto**?

🎩 Eso es exactamente lo que hacen los **Kernels** en SVM.

---

## ⛰️ Analogía Visual: Valle y Colina

Piensa en esto:

* 📄 En tu hoja 2D, un **círculo** no se puede dividir con una línea recta.
* Pero... ¿y si los puntos del **centro (azules)** los pudieras **"elevar"** a una tercera dimensión, como si estuvieran en una **colina**?
* Mientras tanto, los puntos del **anillo rojo** se mantienen abajo, en el **valle**.

🧱 En ese espacio 3D, ahora puedes simplemente **cortar horizontalmente con un plano plano**,
y separar perfectamente los de la colina (azules) de los del valle (rojos).

> 🔍 En 2D: separación imposible ❌
> 🛸 En 3D con kernel: separación simple ✔️

---

## 🔁 ¿Qué hace el **Kernel** entonces?

* Es una **función matemática** que transforma tus datos originales a un espacio de mayor dimensión **sin que tú tengas que hacerlo manualmente**.
* Gracias a esta transformación, el algoritmo SVM puede encontrar un **hiperplano separador** incluso cuando los datos no eran separables antes.

🎯 Resultado:
Separación lineal **en un espacio transformado**,
aunque en el espacio original **no lo parecía**.

---

## 💬 En resumen

* 🧩 Algunos conjuntos de datos **no se pueden separar linealmente** en su forma original.
* 🧠 Los **kernels** permiten transformar esos datos a un **espacio donde sí lo son**.
* 🧗‍♂️ Como si **elevaras una parte del plano**, logrando una separación clara con un plano simple en esa nueva dimensión.

---

## ✨ Esto es precisamente la magia de los **Kernels** en SVM:

### 🔮 Transformación Implícita (el famoso *Kernel Trick*)

En lugar de tener que calcular manualmente nuevas coordenadas (por ejemplo, una altura `z` en el ejemplo de la colina 🏔️), el **Kernel** es una función matemática que permite al **SVM** comportarse **como si** los datos ya estuvieran en un espacio transformado de mayor dimensión…
¡pero **sin tener que calcular esas coordenadas explícitamente**!

A esto se le conoce como el **`Kernel Trick`** 🧙‍♂️

> 💡 Lo que hace el *Kernel Trick* es calcular **productos escalares** (medidas de similitud) **directamente en el espacio original**,
> de una forma que **equivale** a haber hecho las transformaciones complejas a otro espacio.

---

## 🧪 Diferentes Tipos de Transformaciones (Kernels más comunes)

Así como puedes imaginar diferentes maneras de “elevar” o “doblar” tus datos,
también existen distintos **tipos de kernels**, cada uno con su propia forma de transformación implícita:

---

### 1. 📏 **Kernel Lineal**

* ✅ El más simple.
* No transforma nada: trabaja **en el espacio original**.
* Es el que usamos cuando los datos **ya son linealmente separables**.
* Útil para problemas simples donde una **línea recta o un plano** bastan.

---

### 2. 📐 **Kernel Polinomial**

* 🧮 Crea **combinaciones polinómicas** de las variables originales.
* Ejemplo: transforma (x₁, x₂) en algo como (x₁, x₂, x₁², x₂², x₁·x₂).
* Esto permite generar **fronteras de decisión curvas**: parábolas, circunferencias, etc.
* 🔄 Ideal para capturar relaciones **no lineales suaves**.

---

### 3. 🌊 **Kernel RBF (Radial Basis Function)**

* 🎯 Es el más potente y más usado.
* Tiene una intuición parecida a una colina ⛰️:
  mide **la influencia de cada punto**, que **disminuye con la distancia** (como ondas en un estanque).
* Mapea los datos a un **espacio de dimensiones infinitas**. 😮
* Permite que el SVM dibuje **fronteras muy complejas y precisas**.

---

## 🎨 ¿Entonces qué hacen los Kernels?

Podemos pensarlos como:

> 🧪 **Recetas matemáticas** que le dicen al SVM cómo comparar puntos como si estuvieran en un espacio de mayor dimensión.

Gracias a los kernels, el SVM puede encontrar una **frontera de decisión lineal en un espacio invisible** y, cuando la proyectamos de vuelta a nuestro espacio original, vemos una **frontera curva o compleja**, como el **círculo** que separa el centro del anillo.

---

## 🧭 Resumen Visual

| Kernel        | ¿Transforma los datos?    | ¿Cómo son las fronteras?            | Ideal para...                   |
| ------------- | ------------------------- | ----------------------------------- | ------------------------------- |
| 📏 Lineal     | ❌ No                      | Líneas rectas / planos              | Datos linealmente separables    |
| 📐 Polinomial | ✅ Sí                      | Curvas suaves (parábolas, círculos) | Relación no lineal moderada     |
| 🌊 RBF        | ✅ Sí (dimensión infinita) | Fronteras complejas y adaptativas   | Problemas no lineales complejos |

---

# 🛠️ Parámetros Clave en scikit-learn para SVM: C, kernel, degree, y gamma.
Cuando usamos SVC() de scikit-learn, hay varios parámetros importantes que determinan cómo se comporta el clasificador SVM. Aquí te explico los más relevantes:

## 🧪 Parámetro `kernel` en SVM (con `scikit-learn`)

> **¿Qué le estás diciendo al modelo cuando defines el parámetro `kernel`?**
> Le estás diciendo **cómo transformar el espacio** para que una separación lineal sea posible... incluso si los datos **no lo son en el espacio original**.

---

### 🧠 ¿Qué es un *kernel*?

Un **kernel** es como una **receta matemática 🧮** que define **cómo se calcula la similitud** entre puntos de datos.
Detrás de escena, permite transformar (implícitamente) los datos a un espacio de mayor dimensión, **sin tener que hacer la transformación explícita**.
Eso es lo que se conoce como el famoso **kernel trick** 🪄.

---

### 📌 ¿Qué opciones puedes darle a `kernel` en `scikit-learn`?

| Valor de `kernel`        | ✨ Transformación que aplica                         | 🔍 Cuándo usarlo                                                                             |
| ------------------------ | --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `"linear"`               | ➖ No transforma los datos. Usa el espacio original. | ✅ Ideal si tus datos ya son linealmente separables. Rápido y eficiente.                      |
| `"poly"`                 | 🧩 Aplica una transformación polinómica             | 🌀 Útil cuando las fronteras de decisión tienen curvaturas suaves o patrones complejos.      |
| `"rbf"` (o `"gaussian"`) | 🌊 Usa funciones de base radial (como ondas)        | 🔥 Muy potente para datos no lineales. Puede crear fronteras muy complejas. Es el más común. |
| `"sigmoid"`              | 🔁 Usa una función tipo tangente hiperbólica        | 🧠 Inspirado en redes neuronales. Poco usado, pero útil en contextos específicos.            |
| `custom callable`        | 🧠 Puedes pasar tu propia función como kernel       | 👨‍🔬 Para experimentación o necesidades muy particulares.                                   |

---

### 🧩 ¿Cómo lo usas en código?

```python
from sklearn.svm import SVC

# Ejemplo con kernel RBF (por defecto)
model = SVC(kernel='rbf')
```

---

### 🎓 Conclusión rápida

> El parámetro `kernel` define **la forma del mundo donde el SVM va a trabajar**.
> Elegir el kernel correcto es como elegir **los lentes adecuados** 👓 para ver la estructura de tus datos.



## Parámetro C:
"C: Parámetro de regularización (similar a 1/λ). Un C pequeño implica mayor regularización (margen más amplio, más errores de clasificación permitidos en el margen). Un C grande implica menor regularización (margen más estrecho)."

### 🧮 Parámetro **C** en SVM: Controlando el equilibrio entre precisión y generalización

En Support Vector Machines, el parámetro **C** es como un **control deslizante** 🎚️ que ajusta cuánto le permitimos al modelo equivocarse durante el entrenamiento.

---

### ⚖️ ¿Qué significa ajustar **C**?

| Valor de **C**   | 🔍 Comportamiento del modelo                                       | 🎯 Consecuencias                                                                                                         |
| ---------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| 🔹 **C pequeño** | 🔧 **Mayor regularización**<br>🔄 Permite errores de clasificación | ✅ Mejor capacidad de generalización<br>📉 Menor riesgo de *overfitting*<br>⚠️ Puede cometer más errores en entrenamiento |
| 🔸 **C grande**  | 🚫 **Menor regularización**<br>🔍 Penaliza fuertemente los errores | 🎯 Alta precisión en entrenamiento<br>📈 Mayor riesgo de *overfitting*<br>⚠️ Puede memorizar ruido o *outliers*          |

---

### 🧠 ¿Cómo interpretarlo?

* 🛡️ **C pequeño** → El modelo busca un **margen amplio** aunque tenga que **ignorar algunos puntos** mal clasificados. Más tolerancia, más robustez.
* 🧷 **C grande** → El modelo intenta **clasificar todo perfectamente**, ajustando el margen para abarcar incluso casos extremos o ruidosos.

---

### 🎓 Conclusión rápida

> **C pequeño** = Más tolerancia, mejor generalización
> **C grande** = Menos tolerancia, más precisión en entrenamiento (pero cuidado con el *overfitting*)

---

## 🎛️ Parámetros `degree` y `gamma` en SVM

Estos parámetros controlan **la forma y complejidad** de la frontera de decisión cuando usas ciertos tipos de kernel.

---

### 📐 `degree` – Grado del Polinomio (`kernel='poly'`)

Este parámetro **solo se aplica** si estás utilizando el **kernel polinomial** (`'poly'`).

#### 🧠 ¿Qué hace?

Transforma tus datos al combinar y elevar las características originales.
Por ejemplo, un vector como `(x₁, x₂)` se convierte en:
→ `(x₁, x₂, x₁², x₂², x₁·x₂, ...)`

#### 🔢 ¿Qué efecto tiene?

| Degree | Forma de la frontera | Ejemplo visual               |
| ------ | -------------------- | ---------------------------- |
| 1      | Lineal               | Una línea recta              |
| 2      | Cuadrática           | Parábolas, círculos, elipses |
| 3      | Cúbica               | Ondas suaves o curvas        |
| >3     | Muy compleja         | Fronteras con muchas curvas  |

📌 **Cuanto mayor sea el `degree`**, más curvas y ondulaciones puede tener la frontera de decisión. Pero cuidado: demasiada complejidad → ⚠️ sobreajuste(Overfitting).

---

### 🌌 `gamma` – Alcance de la Influencia (`kernel='rbf'`)

#### 🧠 ¿Qué hace?

Controla **cuánta influencia tiene cada punto de entrenamiento** sobre la forma de la frontera.
Es como si cada punto “irradiara” un campo alrededor de sí mismo:

* **Gamma alto** → campo pequeño, muy localizado 🔍
* **Gamma bajo** → campo grande, más extendido 🛰️

### 📏 Intuición:
| Valor de `gamma`          | Significado         | Efecto                                                                                               |
| ------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| **Pequeño** (ej. `0.001`) | **Alcance grande**  | Cada punto afecta a muchos vecinos → modelo más **suave**, generaliza más                            |
| **Grande** (ej. `10`)     | **Alcance pequeño** | Cada punto afecta solo a sí mismo → modelo más **ajustado**, riesgo de **sobreajuste** (overfitting) |

---

#### 🔁 Imagina esto:

* Cada punto "deforma" el espacio a su alrededor.
* Si gamma es alto, la deformación ocurre en un área pequeña.
* Si gamma es bajo, esa influencia llega más lejos.

#### 📊 Comparativo resumen:

| Gamma   | Alcance de Influencia 🌐 | Forma de la Frontera 🧭 | Riesgo de Overfitting ⚠️         |
| ------- | ------------------------ | ----------------------- | -------------------------------- |
| 🔹 Bajo | Amplio (global)          | Suave y generalizada    | Bajo (puede haber underfitting)  |
| 🔸 Alto | Localizado (fino)        | Muy ajustada al detalle | Alto (riesgo de memorizar ruido) |

---


| Kernel          | Usa `degree` | Usa `gamma` | Comentarios clave                                              |
| --------------- | ------------ | ----------- | -------------------------------------------------------------- |
| `'linear'`      | No           | No          | Kernel lineal, no usa ni `degree` ni `gamma`                   |
| `'poly'`        | Sí           | Sí          | Kernel polinomial, `degree` define el grado, `gamma` la escala |
| `'rbf'`         | No           | Sí          | Kernel RBF (radial), solo usa `gamma`                          |
| `'sigmoid'`     | No           | Sí          | Kernel sigmoide, usa `gamma` y `coef0`                         |
| `'precomputed'` | No           | No          | Se usa matriz de kernel precalculada, no usa estos parámetros  |
---

## ⚙️ Implementación de un Clasificador SVC con `scikit-learn`

### 1. 📥 Importación del Dataset

Primero, importamos el dataset de **cáncer de mama** usando `load_breast_cancer`. Luego, cargamos los datos en las variables `X` (características) e `y` (etiquetas).

### 2. ✂️ ¿Por qué dividir los datos?

Separar el dataset en **entrenamiento** y **prueba** tiene un propósito fundamental:

> 🧪 **Evaluar la capacidad de generalización del modelo.**

Esto nos permite saber si el modelo realmente aprendió a clasificar correctamente o solo memorizó los datos.

---

### 3. 🧪 División del Dataset

Usamos la función `train_test_split` para dividir los datos.
Nos devuelve 4 conjuntos:

* `X_train`: características de entrenamiento
* `X_test`: características de prueba
* `y_train`: etiquetas de entrenamiento
* `y_test`: etiquetas de prueba

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### ✅ Parámetros útiles:

* `test_size=0.2`: reserva el 20% de los datos para prueba.
* `random_state=42`: fija una **semilla** para garantizar que la división sea **reproducible**.

---

### 4. 🖨️ Buenas prácticas

Es recomendable imprimir las formas de los conjuntos para verificar:

```python
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
```

---

### 5. 🤖 Crear y entrenar el modelo

Creamos una instancia de un clasificador SVC lineal:

```python
from sklearn.svm import SVC

modelo_svc = SVC(kernel='linear')
```

Entrenamos el modelo con el método `.fit()`:

```python
modelo_svc.fit(X_train, y_train)
```

---

### 6. 🔮 Realizar predicciones

Utilizamos `.predict()` para predecir etiquetas:

```python
y_pred_test = modelo_svc.predict(X_test)   # predicciones en test
y_pred_train = modelo_svc.predict(X_train) # opcional: predicciones en entrenamiento
```

📝 `.predict()` recibe un array del mismo formato que `X_train` o `X_test`:
`(n_muestras, n_características)`

---

### 7. 📏 Medir la exactitud (accuracy)

#### Método 1: Usando `accuracy_score` de `sklearn.metrics`

```python
from sklearn.metrics import accuracy_score

accuracy_test = accuracy_score(y_test, y_pred_test)
print("Exactitud en test:", accuracy_test)
```

#### Método 2: Usando `.score()` del modelo

```python
print("Exactitud en test (con .score()):", modelo_svc.score(X_test, y_test))
```


## 🔁 Paso 1 Bloque de Acciones Repetitivas 

Cada experimento con una configuración distinta de SVM sigue el mismo bloque de **tres pasos fundamentales**:

### 1. 🧱 Instanciar el modelo

Seleccionamos un kernel (por ejemplo, `'linear'`, `'rbf'`, `'poly'`) y establecemos los valores de sus **hiperparámetros**:

```python
modelo = SVC(kernel='rbf', C=1.0, gamma=0.01)
```

### 2. 🧠 Entrenar el modelo

Ajustamos el modelo a los datos de entrenamiento:

```python
modelo.fit(X_train, y_train)
```

### 3. 📊 Evaluar el modelo

Calculamos la exactitud en **ambos conjuntos**:

```python
accuracy_train = modelo.score(X_train, y_train)
accuracy_test = modelo.score(X_test, y_test)
```

Este bloque de tres pasos se **repite sistemáticamente** para cada combinación de parámetros que deseemos evaluar.

---

## 🔄 Paso 2: Identificar el Elemento Variable

En nuestra experimentación, los elementos que vamos a variar son:

| Parámetro | ¿Cuándo aplica?         | Efecto esperado                                 |
| --------- | ----------------------- | ----------------------------------------------- |
| `kernel`  | Siempre                 | Tipo de transformación del espacio de datos     |
| `C`       | Siempre                 | Controla la regularización (rigidez del margen) |
| `gamma`   | Solo si `kernel='rbf'`  | Alcance de la influencia de cada muestra        |
| `degree`  | Solo si `kernel='poly'` | Grado del polinomio que define la curvatura     |

Esta identificación permite crear una **rejilla de combinaciones** que serán exploradas experimentalmente.

---

## 🔍 Paso 3: Evaluar Casos Aislados

### Ejemplo de Evaluación Crítica

> Configuración: `kernel='rbf'`, `C=10`, `gamma=0.1`
> Resultado:
>
> * Exactitud en entrenamiento: **100.00%**
> * Exactitud en prueba: **62.28%**

Este comportamiento es una **señal clara de sobreajuste (overfitting)**.

### ¿Por qué ocurre?

* **C=10**:
  Valor alto → menor regularización → el modelo se esfuerza por clasificar perfectamente los datos de entrenamiento.
  Resultado: fronteras complejas que pueden capturar ruido.

* **gamma=0.1**:
  Valor relativamente alto → la "influencia" de cada punto de entrenamiento es muy localizada.
  Resultado: fronteras muy sensibles a puntos individuales → se generan ondulaciones innecesarias.

### 💡 Reflexión

No hay un valor universalmente "bueno" o "malo" para `C` o `gamma`.

> Todo depende del dataset y de cómo interactúan estos hiperparámetros.

Por eso es esencial experimentar **sistemáticamente** con diferentes combinaciones:

> Solo así se encuentra el equilibrio ideal entre ajuste y generalización.


---

## 🔁 Paso 4: Construir el Bucle Esqueleto

El objetivo de este paso es crear un **bucle anidado** que recorra distintas combinaciones de:

* Tipos de `kernel`
* Valores del parámetro `C`
* Valores de `gamma` (si el kernel es `'rbf'`)
* Valores de `degree` (si el kernel es `'poly'`)

Este bucle nos permite **experimentar con múltiples configuraciones** para encontrar la "receta" óptima, es decir, la combinación de parámetros que proporcione el mejor rendimiento de generalización.

También creamos una **lista vacía** donde almacenaremos los resultados de cada experimento en forma de diccionario.

Durante la iteración, usamos estructuras `if` para decidir qué parámetros usar según el tipo de `kernel`. Por ejemplo:

* Si `kernel == 'rbf'`, usamos distintos valores de `gamma`.
* Si `kernel == 'poly'`, usamos distintos valores de `degree`.
* Si `kernel == 'linear'`, no se necesitan ni `gamma` ni `degree`.

---

## 🧩 Paso 5: Integración del Bucle y Evaluación

Dentro de cada `if`, seguimos siempre el **bloque de acciones repetitivas** definido previamente:

1. **Instanciamos** el modelo SVM con la configuración actual.
2. **Entrenamos** el modelo usando `fit()`.
3. **Evaluamos** el modelo con `.score()` tanto en el conjunto de entrenamiento como en el de prueba.

Para cada experimento, guardamos un diccionario con todos los parámetros utilizados y sus respectivas métricas:

```python
{
  'kernel': 'rbf',
  'C': 10,
  'gamma': 0.1,
  'train_accuracy': 1.0,
  'test_accuracy': 0.6228
}
```

Este diccionario se añade a la lista general de resultados.

---

## 📈 Paso 6: Manejo y Comparación de Resultados

Una vez almacenados todos los resultados, queremos **encontrar la mejor combinación** de parámetros según su rendimiento en el conjunto de prueba.

### Método Manual

1. Creamos dos variables:

   * Una para guardar la **mejor exactitud** encontrada hasta el momento.
   * Otra para guardar el **diccionario completo** del experimento asociado.

2. Recorremos la lista de resultados y comparamos:

```python
for experimento_actual in lista_de_resultados:
    exactitud_actual_prueba = experimento_actual["Exactitud Prueba"] 
    if exactitud_actual_prueba > mejor_exactitud:
        mejor_exactitud = exactitud_actual_prueba
        mejor_experimento = experimento_actual
```

3. Finalmente, imprimimos el `mejor_experimento`.

### Método con Pandas (más limpio y directo)

Si convertimos la lista de diccionarios en un `DataFrame`, el análisis se simplifica considerablemente:

```python
import pandas as pd

df = pd.DataFrame(lista_resultados)
mejor_experimento = df.loc[df["test_accuracy"].idxmax()]
print(mejor_experimento)
```

> Aunque ambos métodos son válidos, **pandas ofrece mayor eficiencia y legibilidad**, especialmente cuando se trabaja con grandes cantidades de experimentos.

---



# 🧪 Implementación de SVR y Evaluación de Métricas

## 🎯 Paso 1: Clarificar la Meta

Cargamos y preparamos los datos del conjunto `california_housing`, con el objetivo de predecir precios de viviendas. Utilizaremos un modelo **SVR (Support Vector Regressor)** con una configuración básica de hiperparámetros para realizar predicciones y calcular el **MSE** tanto en el conjunto de entrenamiento como en el de prueba.

---

## ⚖️ Paso 2: Escalar las Características

### ¿Por qué escalar para SVR?

El algoritmo **SVR**, especialmente con el kernel `RBF` (que es el valor por defecto), es **sensible a la escala de las características**. Si los rangos de valores varían mucho entre columnas, el modelo puede:

* Favorecer características con valores más grandes.
* Aprender patrones incorrectos.
* Ofrecer predicciones menos precisas.

### Diagnóstico previo

Podemos visualizar la dispersión de las características con:

```python
df_train = pd.DataFrame(X_train, columns=datos.feature_names).describe().round(2)
print("Descripción de las características:\n", df_train)
```

Esto nos ayuda a detectar si existe disparidad en las escalas de las variables, lo cual justificaría el escalado.

### ¿Cómo se escalan los datos?

1. **`fit()`** calcula la media y desviación estándar de `X_train`.
2. **`transform()`** aplica esta transformación para que cada característica tenga:

* 📍 **Media 0**: los valores se centran en torno a 0.
* 📈 **Desviación estándar 1**: los datos tienen una dispersión estándarizada.

Esto mejora significativamente el rendimiento de algoritmos sensibles a la escala.

> ⚠️ **Importante:** Solo se usa `fit()` con `X_train`. No debe aplicarse en `X_test`, para evitar que el modelo "vea" datos que deben ser desconocidos.

---

## 🧪 Evaluación de Modelos

### `.score()` en Clasificación vs Regresión

| Modelo | Método `.score()` devuelve |
| ------ | -------------------------- |
| `SVC`  | **Exactitud (Accuracy)**   |
| `SVR`  | **Coeficiente R²**         |

Esto se debe a que los modelos de **clasificación** buscan asignar la categoría correcta, mientras que los de **regresión** predicen valores numéricos.

### ¿Qué es el R²?

* Mide **cuánta varianza de la variable objetivo** puede ser explicada por el modelo.
* Un R² cercano a **1.0** indica un modelo con muy buena capacidad predictiva.
* No debe confundirse con la "exactitud".

---

## ⚙️ Paso 3: Etapa de Experimentación

Definimos un conjunto de hiperparámetros para explorar:

```python
lista_kernel = ["linear", "rbf", "poly"]
valores_C = [0.1, 1, 10, 100]
valores_gamma = [0.01, 0.1, 1]
valores_degree = [2, 3, 4]
```

Luego, construiremos un bucle para probar combinaciones y registrar métricas (R², MSE, MAE). Esto nos permitirá encontrar el conjunto de hiperparámetros más efectivo para nuestro problema de regresión.

---

## 📉 Métricas de Error: MAE vs MSE

Ambas métricas evalúan el **error** de las predicciones, pero de manera diferente:

| Métrica | Sensibilidad a outliers | Interpretación         | Penalización                    |
| ------- | ----------------------- | ---------------------- | ------------------------------- |
| MAE     | Baja                    | Error absoluto medio   | Lineal                          |
| MSE     | Alta                    | Error cuadrático medio | Cuadrática (outliers pesan más) |

Ejemplo de cálculo de **MAE**:

```python
# Paso a paso para MAE
diferencias = y_true - y_pred
errores_absolutos = np.abs(diferencias)
mae = np.mean(errores_absolutos)
```

---

## 🧠 Paso 4: Métricas Avanzadas para SVC

Una vez que hayas encontrado los mejores hiperparámetros para tu modelo `SVC` (clasificación), es importante complementar la **exactitud** con otras métricas:

* ✅ **Matriz de Confusión**
* ✅ **Precisión**
* ✅ **Recall**
* ✅ **F1-score**

Esto se logra con:

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
```

Estas métricas son especialmente importantes si tienes un **desequilibrio de clases**, ya que la exactitud por sí sola puede ser engañosa.

---

## ✅ Conclusión

* En tareas de **regresión**, usamos métricas como R², MAE y MSE.
* En tareas de **clasificación**, usamos exactitud, matriz de confusión y métricas derivadas.
* Escalar los datos mejora el rendimiento de modelos sensibles como `SVR`.
* Separar correctamente los conjuntos de entrenamiento y prueba es esencial para una evaluación confiable.

---


## Resumen de  Métricas de Evaluación en Clasificación Binaria: Precision, Recall y F1-Score

| Métrica   | Se enfoca en...                           | Qué quiere evitar                 |
| --------- | ----------------------------------------- | --------------------------------- |
| Precision | Predicciones positivas correctas          | Falsos positivos                  |
| Recall    | Positivos reales correctamente detectados | Falsos negativos                  |
| F1        | Balance entre Precision y Recall          | Cuando uno de los dos falla mucho |


