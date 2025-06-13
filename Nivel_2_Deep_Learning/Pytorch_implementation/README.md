# 🚀 Implementación de Redes Neuronales con PyTorch

> "De NumPy manual a PyTorch automático: transformando cálculos complejos en líneas elegantes de código."

---

<details>
  <summary><strong>TL;DR: Resumen y Analogía (Haz clic para expandir)</strong></summary>
  
  ### Resumen Clave 📝
  Estamos evolucionando nuestro modelo de red neuronal `run_moon` desde implementación manual con NumPy hacia PyTorch. PyTorch nos permite automatizar cálculos de gradientes y operaciones tensoriales de forma más intuitiva y eficiente, utilizando la herramienta **autograd** para el seguimiento automático de operaciones.

  ### Analogía para Entenderlo Mejor 💡
  Imagina que antes construías una casa ladrillo por ladrillo (NumPy manual), calculando cada ángulo y medida a mano. Ahora tienes una máquina constructora inteligente (PyTorch) que no solo coloca los ladrillos automáticamente, sino que también **recuerda** cada paso del proceso para poder deshacerlo o mejorarlo después (autograd). ¡Es como tener un asistente de construcción con memoria fotográfica!

</details>

---

## 📖 Explicación Completa y Sencilla

### 🎯 El Gran Salto: De NumPy a PyTorch

PyTorch revoluciona nuestro enfoque anterior al ofrecer **operaciones tensoriales** similares a NumPy, pero específicamente diseñadas para redes neuronales. La gran ventaja es que PyTorch puede realizar automáticamente todos esos cálculos de gradientes que antes hacíamos manualmente.

### 🔧 La Magia de Autograd

**Autograd** es el corazón de PyTorch. Su función principal es:

* **Seguimiento automático** de todas las operaciones matemáticas
* **Construcción del grafo computacional** necesario para calcular gradientes
* **Diferenciación automática** sin intervención manual

Para activar esta magia, necesitamos el atributo especial:

```python
tensor.requires_grad = True
```

### 🏗️ Configuración de Nuestros Componentes

Para que todo funcione correctamente, establecemos que:

**Parámetros principales:**
* `W1, b1, W2, b2` → tensores con `requires_grad=True`
* `X, Y` (datos) → también convertidos a tensores

### 🎨 Construyendo Nuestra Clase de Red Neuronal

Definimos nuestra **clase blueprint** para construir la red:

```python
class MyNeuralNetwork(nn.Module):
```

Esta clase hereda de `nn.Module`, que es la **base fundamental** para todas las redes en PyTorch.

### 🔨 El Constructor: Función `__init__`

La función especial `__init__` se ejecuta automáticamente al crear un objeto:

```python
def __init__(self, n_x, n_h, n_y):
```

**Parámetros de la arquitectura:**
* `n_x` → número de entradas (features de input)
* `n_h` → neuronas en la capa oculta
* `n_y` → número de salidas (predicciones finales)

### ⚡ Inicialización Correcta

```python
super().__init__()
```

Esta línea **inicializa correctamente** la clase base `nn.Module`, asegurando que todas las funcionalidades de PyTorch estén disponibles.

### 🧠 Definiendo las Capas de la Red

**Primera capa lineal:**
```python
self.layer_1 = nn.Linear(n_x, n_h)
```

Esto reemplaza todas las operaciones matriciales manuales que hacíamos antes. PyTorch maneja automáticamente la inicialización de pesos y las multiplicaciones matriz-vector.

**Función de activación no lineal:**
```python
self.activation_1 = nn.Tanh()
```

La **no linealidad es crucial** - sin ella, múltiples capas lineales se reducirían a una sola transformación lineal, limitando drásticamente las capacidades del modelo.

**Segunda capa lineal:**
```python
self.layer_2 = nn.Linear(n_h, n_y)
```

Esta capa final toma las `n_h` salidas de la capa oculta y produce las `n_y` predicciones finales.

## ✨ Puntos Finales

* **Automatización inteligente:** PyTorch maneja automáticamente cálculos complejos que antes requerían implementación manual
* **Autograd = superpoder:** El seguimiento automático de gradientes revoluciona el entrenamiento de redes neuronales
* **Arquitectura modular:** La herencia de `nn.Module` proporciona una estructura robusta y extensible para construir redes complejas



## 🧠 Definir la función `forward`

Después de definir nuestra red neuronal, el siguiente paso es crear el método `forward`. Esta función se encarga de pasar los datos de entrada `x` a través de las capas del modelo.

### ⚙️ ¿Qué hace el método `forward`?

1. 🧩 **Capa oculta**  
   Aplica una transformación lineal a los datos y luego una función de activación no lineal (por ejemplo, `ReLU`).

2. 🎯 **Capa de salida**  
   Por ahora **no** aplicamos ninguna función de activación aquí. Más adelante usaremos `BCEWithLogitsLoss`, que **ya incluye** la activación `sigmoid` automáticamente.



## 🛠️ Crear una instancia del modelo

A continuación, creamos un objeto (instancia) de nuestra red neuronal personalizada:

```python
model = MyNeuralNetwork(n_x=input_size, n_h=hidden_size, n_y=output_size)
print(model)
```

✅ Esto inicializa nuestra red con la arquitectura definida y nos permite ver su estructura.


## 📚 ¿Qué necesitamos para entrenar el modelo?

Antes de comenzar a entrenar, asegurémonos de tener todos los componentes clave. A continuación, los explicamos de forma sencilla:

1. 🧠 **El Modelo (`MyNeuralNetwork`)**

   > Es el "cerebro" del sistema. Ya lo construimos.

2. 📊 **Los Datos (`X` e `Y`)**

   > Son el "libro de texto". El modelo los usará para aprender.

3. ❗ **Función de Pérdida (Loss Function)**

   > Es como un "examen". Le dice al modelo qué tan equivocadas son sus predicciones.
   > En este caso, usamos `BCEWithLogitsLoss`, que combina:

   * 🌀 `Sigmoid`: convierte la salida en una probabilidad.
   * 📉 `Binary Cross-Entropy`: mide el error de clasificación binaria.

4. 🧮 **Optimizador (Optimizer)**

   > Es el "tutor". Ayuda al modelo a mejorar ajustando sus pesos en cada iteración.

5. 🔁 **Bucle de Entrenamiento (Training Loop)**

   > Es el "horario de estudio". Repite muchas veces el proceso de:

   * Aprender 🧠
   * Evaluarse ❗
   * Corregirse 🔧

🚀 ¡Con todo esto listo, ya podemos pasar al entrenamiento de la red neuronal!

# 🚀 IMPLEMENTACIÓN

En esta sección, vamos a construir y entrenar nuestra red neuronal utilizando PyTorch, junto con algunas librerías adicionales como NumPy y Matplotlib para el manejo de datos y visualización.

---

## 🔧 Importaciones necesarias

Primero importamos las librerías esenciales:

- `torch`: Para manejar tensores y construir el modelo.
- `numpy`: Para operaciones numéricas.
- `matplotlib`: Para graficar resultados.
- Nuestra clase `MyNeuralNetwork` desde el script donde definimos la arquitectura y funciones de entrenamiento.

De esta manera, tenemos listo el “plano de construcción” de nuestra red neuronal.

---

## 🧩 Crear la instancia del modelo

Creamos nuestro modelo con:

```python
model = MyNeuralNetwork(n_x=input_size, n_h=hidden_size, n_y=output_size)
```

¿Por qué hacemos esto?
Porque este objeto contiene toda la lógica y estructura de la red, y será la base para el entrenamiento. A diferencia de usar solo NumPy, PyTorch facilita el entrenamiento con menos código y de forma más eficiente.

---

## 📊 Preparar los datos

Usamos el conjunto de datos `make_moons` de `sklearn`.

Para evitar errores en las operaciones matriciales, hacemos un `reshape` a las etiquetas para que tengan forma `(1, n)`.

Luego convertimos los datos de NumPy a tensores de PyTorch, ya que PyTorch trabaja internamente con tensores para realizar cálculos rápidos y eficientes:

```python
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
```

Finalmente, renombramos las variables para mejorar la claridad en el código.

---

## ⚙️ Definir función de pérdida y optimizador

* Usamos la función de pérdida **BCEWithLogitsLoss** (Binary Cross-Entropy con logits).

  > Esta función es ideal para clasificación binaria y ya incluye internamente la activación `sigmoid`, por lo que no necesitamos aplicarla manualmente.

* Para optimizar, usamos el optimizador **Adam** con tasa de aprendizaje 0.01, que ajusta los pesos para minimizar la pérdida.

---

## 🔄 Bucle de entrenamiento

```python
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()                 # 1️⃣ Limpiar gradientes del paso anterior
    y_pred = model(X_train)               # 2️⃣ Forward pass: obtener predicciones
    loss = criterion(y_pred, y_train)    # 3️⃣ Calcular pérdida
    loss.backward()                      # 4️⃣ Backpropagation: calcular gradientes
    optimizer.step()                     # 5️⃣ Actualizar pesos

    # Mostrar la pérdida cada 100 épocas
    if epoch % 100 == 0:
        print(f"Loss (epoch {epoch}): {loss.item():.4f}")
```

### Paso a paso del loop:

* 🔹 **`zero_grad()`**: Limpiamos los gradientes acumulados para evitar sumas incorrectas.
* 🔹 **Forward pass**: Pasamos los datos por el modelo para obtener las predicciones.
* 🔹 **Calcular pérdida**: Comparamos las predicciones con los valores reales.
* 🔹 **Backpropagation**: Calculamos cómo cambiar cada peso para reducir la pérdida.
* 🔹 **Actualizar pesos**: El optimizador ajusta los pesos con base en los gradientes.
* 🔹 **Monitoreo**: Imprimimos la pérdida periódicamente para seguir el progreso.

---

## 🔮 Función de predicción

```python
def predict(x):
    model.eval()               # Cambiar a modo evaluación (sin calcular gradientes)
    with torch.no_grad():      # Evitar cálculo de gradientes para optimizar memoria y velocidad
        logits = model(x)      
        probs = torch.sigmoid(logits)     # Convertir logits en probabilidades
        prediction = (probs >= 0.5).float()  # Umbral para clasificar en 0 o 1
    model.train()              # Volver a modo entrenamiento
    return prediction
```

> **¿Por qué usar `torch.no_grad()`?**
> Durante el entrenamiento PyTorch guarda información para calcular gradientes. En predicciones esto no es necesario y consumiría recursos extra, por eso desactivamos el cálculo de gradientes.

---

## 🎨 Función para graficar la frontera de decisión

```python
def plot_decision_boundary(predict_fn, X, y):
    # 1️⃣ Crear una malla fina que cubra todo el espacio de características
    # 2️⃣ Predecir la clase para cada punto en la malla usando predict_fn
    # 3️⃣ Graficar la frontera de decisión coloreando según la clase
    # 4️⃣ Dibujar los puntos originales para comparar con la clasificación
```

Esta función permite visualizar cómo el modelo divide el espacio de características entre las distintas clases.

---

# 💡 Abstracción vs. Control Manual en Redes Neuronales


## 🏗️ Definición de capas: `nn.Linear` vs. creación manual de matrices `W` y `b`

- **Control Manual con NumPy:**  
  - Inicializas tú mismo las matrices de pesos `W` y vectores de sesgo `b` (por ejemplo, con ceros, valores aleatorios o distribuciones específicas).  
  - Escribes manualmente todas las operaciones (multiplicación matricial, sumas, activaciones).  
  - Ventaja: comprensión profunda de **cómo funciona** una red neuronal.  
  - Desventaja: mucho código y posibilidad de errores humanos, especialmente al escalar a redes grandes.

- **Con `nn.Linear` en PyTorch:**  
  - Es un módulo de alto nivel que **abstracta y automatiza** la creación y actualización de pesos y sesgos.  
  - Maneja internamente las operaciones matriciales y la inicialización adecuada.  
  - Ideal para proyectos reales porque reduce la complejidad y minimiza errores en operaciones complejas.  
  - El rango de error es mucho menor y facilita trabajar con redes neuronales profundas y grandes.

---

## 🔄 Cálculo de Gradientes: función manual `backward()` vs. `loss.backward()`

- **Función `backward()` con NumPy:**  
  - Implementas tú mismo el cálculo de derivadas parciales de la función de pérdida con respecto a cada parámetro (pesos y sesgos).  
  - Esto puede ser complicado y propenso a errores, sobre todo con operaciones más complejas o muchas capas.  
  - El código es extenso y difícil de mantener.

- **`loss.backward()` en PyTorch:**  
  - PyTorch usa **autograd**: calcula automáticamente los gradientes de manera eficiente y segura.  
  - Profesional y optimizado para modelos reales y grandes.  
  - Reduce drásticamente errores y permite enfocarse en diseño y experimentación en lugar de cálculos manuales.

---

## 🔄 El "Ecosistema": Coexistencia de NumPy/Sklearn y PyTorch

- **NumPy y Sklearn:**  
  - Herramientas maduras y estables para manipulación, preprocesamiento, generación y evaluación de datos.  
  - Permiten, por ejemplo, normalizar datos, hacer splits de entrenamiento/prueba, crear datasets sintéticos, etc.

- **PyTorch:**  
  - Fuerte en modelado, cálculo automático de gradientes, entrenamiento y despliegue de redes neuronales.

- **Uso conjunto:**  
  - Preparas y procesas datos con NumPy/Sklearn.  
  - Luego conviertes datos a tensores con `torch.from_numpy()` para alimentar a PyTorch.  
  - Esto reutiliza librerías especializadas y evita reinventar la rueda.

- **Cuidado:**  
  - La conversión constante entre formatos puede provocar errores sutiles (tipos de datos, dimensiones, CPU vs GPU).  
  - En producción, intenta minimizar el cruce de frameworks para evitar bugs y mejorar rendimiento.

---

## 📈 Curva de Aprendizaje: Mi experiencia personal

- Aprender NumPy y construir redes neuronales **desde cero** te da una base sólida para entender **por qué** y **cómo** funcionan internamente.  
- Es fundamental para construir conocimiento profundo y evitar “cajas negras”.  
- Pero para **implementaciones prácticas y profesionales**, PyTorch es una ventaja enorme: automatiza cálculos, optimización y reduce errores.  
- Además, ahorra muchísimo tiempo, especialmente en proyectos reales con redes complejas.

---

# 🔑 **Conclusión:**  
El control manual es excelente para aprender y entender, pero la abstracción que ofrece PyTorch es clave para eficiencia, escalabilidad y confiabilidad en proyectos reales. Lo ideal es combinar ambos enfoques según la etapa de aprendizaje o desarrollo en la que te encuentres.
