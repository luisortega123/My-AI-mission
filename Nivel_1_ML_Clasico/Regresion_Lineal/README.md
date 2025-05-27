# Regresión Lineal

## ¿Qué es?

La **regresión lineal** es una técnica estadística que busca la relación entre una variable cuantitativa ($Y$) y una o más variables predictoras ($X$).
El **objetivo** es **predecir valores numéricos continuos**, basándose en la suposición de que existe una relación lineal entre las variables explicativas ($X$) y la variable objetivo ($Y$).

---

## ¿Qué tipo de resultado produce?

**Variable cuantitativa continua:**
No clasifica en categorías, sino que entrega un valor numérico que puede tomar cualquier valor dentro de un rango.

---

## Interpretación de los coeficientes

Los coeficientes del modelo ($\theta$) indican cómo cambia $Y$ ante variaciones en $X$.

---

## Función Hipótesis ($h_\theta(X) = X \theta$)

Ecuación fundamental en la regresión lineal: $h_\theta(X) = X \theta$. Define la relación entre las variables predictoras ($X$) y la variable objetivo ($Y$).

### Objetivo

Predecir valores numéricos continuos basándose en una combinación lineal de características de entrada y coeficientes asociados.

### Elementos

1.  **$X$**: Matriz de características (inputs)
    * Cada fila representa una observación (ejemplo de entrenamiento)
    * Cada columna corresponde a una característica (variable predictora)
    * Se agrega una columna de unos para incluir el intercepto ($\theta_0$)

2.  **$\theta$**: Vector de parámetros (coeficientes)
    * Contiene los pesos que el modelo aprende durante el entrenamiento
    * Incluye el intercepto ($\theta_0$)
    * $\theta_1, \theta_2, \dots$ indican la influencia de cada característica

3.  **$h_\theta(X)$**: Predicción del modelo
    * Resultado del producto matricial $X\theta$
    * Cada valor $h_\theta(x^{(i)})$ es la predicción para la observación $i$

---

## Función de Coste (MSE)

**Fórmula:**
$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$

Mide el **promedio del error cuadrático** entre las predicciones del modelo y los valores reales en todo el conjunto de datos.

* **$m$**: Número de observaciones
* **$h_\theta(x^{(i)})$**: Predicción para la observación $i$
* **$y^{(i)}$**: Valor real para la observación $i$

---

### ¿Por qué se eleva al cuadrado la diferencia?

1.  **Evita errores negativos:** Las diferencias se vuelven positivas.
2.  **Penaliza errores grandes:** Un error de 2 pesa más (4) que uno de 1 (1).
3.  **Facilita la optimización:** La función cuadrática es convexa y garantiza un mínimo global.

---

## ¿Por qué queremos minimizar $J(\theta)$?

Minimizar $J(\theta)$ significa ajustar los parámetros $\theta$ para que las predicciones sean lo más cercanas posible a los valores reales.

### Métodos comunes:

* **Mínimos Cuadrados Ordinarios (OLS):** Solución analítica.
* **Descenso de Gradiente (GD):** Método iterativo.

---

## ¿Qué implica un $J(\theta)$ pequeño?

* **Buen ajuste:** Las predicciones están cerca de los valores reales.
* **Alta precisión:** El modelo generaliza bien.
* **Menor incertidumbre:** Los errores (residuos) tienen baja variabilidad.

---

# Descenso de Gradiente (Gradient Descent)

El **descenso de gradiente** busca minimizar $J(\theta)$, ajustando iterativamente los parámetros $\theta$ para reducir el error.

### En resumen:

Es un método que permite a un modelo **aprender** los mejores valores de los parámetros $\theta$, optimizando la predicción. El resultado del entrenamiento es el vector `theta_final`, que contiene los coeficientes óptimos encontrados por el modelo.

---

## Regla de Actualización

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$ (Esta es la regla implícita que se describe)

### Componentes

**$\alpha$ (Tasa de Aprendizaje):**

* Controla el tamaño del paso en cada iteración.
* $\alpha$ alto → puede hacer que el algoritmo no converja.
* $\alpha$ bajo → puede hacer que la convergencia sea muy lenta.

**Derivada Parcial ($\frac{\partial}{\partial \theta_j} J(\theta)$):**

* Mide la dirección de mayor aumento de $J(\theta)$
* Al restarla, el modelo se mueve en dirección descendente (hacia el mínimo).

---

## Gradiente Vectorizado

**Fórmula:**
$\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)$

### Términos

1.  **$X\theta$**: Vector de predicciones para todas las observaciones.
2.  **$X\theta - y$**: Vector de errores residuales.
3.  **$X^T$**: Multiplicación por la transpuesta de $X$ pondera los errores por cada característica.
4.  **$\frac{1}{m}$**: Promedia el gradiente sobre el conjunto de datos.

---

## ¿Qué representa el resultado?

El vector $\nabla J(\theta)$ nos dice:

* **Dirección:**
    * Si $\frac{\partial J}{\partial \theta_j} > 0$ = disminuir $\theta_j$
    * Si $\frac{\partial J}{\partial \theta_j} < 0$ = aumentar $\theta_j$

* **Magnitud:**
    * Cuánto influye ese $\theta_j$ en el error total

---

## Resumen

* **Objetivo:** Minimizar $J(\theta)$ ajustando $\theta$ para reducir errores
* **Tasa de Aprendizaje ($\alpha$):** Controla la velocidad de convergencia
* **Gradiente Vectorizado:** Forma eficiente de calcular el ajuste de todos los coeficientes a la vez

# 📘 Pasos del Algoritmo de Regresión Lineal (California Housing)

## 1. 🗂️ Importación de Datos

Importamos el dataset **California Housing** y las funciones necesarias de **NumPy**.

* `.data`: contiene las **características** (features), en forma de matriz ($X$).
* `.target`: contiene los **valores a predecir** (precios promedio de casas) ($Y$).

Se ajustó la forma de $Y$ usando `np.reshape` para trabajar con matrices columna, sin alterar los datos.

---

## 2. ⚖️ Normalización (Estandarización)

Calculamos la **media ($\mu$)** y **desviación estándar ($\sigma$)** de los datos originales para escalar las características. Esto es clave porque:

* Las variables originales tienen diferentes escalas.
* Sin escalar, el descenso de gradiente puede ser lento o ineficaz.
* El escalado mejora la velocidad y estabilidad del entrenamiento.
kv
**🔧 Sin este paso, el modelo devolvía `null` en `theta_final`, sin importar los hiperparámetros.**

---

## 3. ➕ Agregar Columna de Unos (Bias)

Se añadió una columna de unos a `X_scaled` para permitir que el modelo aprenda un **término independiente** ($\theta_0$), haciendo que la recta **no tenga que pasar por el origen (0,0)**.
Esto da **flexibilidad** al modelo.

---

## 4. 🔮 Función de Hipótesis

Se implementó la función `calcular_hipotesis(X, theta)`:

* Predice valores continuos.
* Es la base de la fórmula de **regresión lineal**:
    $h(\theta) = X \theta$

---

## 5. ❌ Función de Coste (MSE)

Se implementó la función de coste: **Error Cuadrático Medio (Mean Squared Error)**. $J(\theta)$.

Mide el **promedio del error al cuadrado** entre predicciones ($h_\theta(x^{(i)})$) y valores reales ($y^{(i)}$).

📉 El objetivo es **minimizarla**:

* **MSE alta** → el modelo se equivoca mucho.
* **MSE baja** → el modelo está aprendiendo bien.

---

## 6. 🔁 Descenso de Gradiente

Se implementó el **descenso de gradiente** para minimizar el error:

* Calcula predicciones, errores y gradiente ($\nabla J(\theta)$) en cada iteración.
* Actualiza $\theta$ con la regla de aprendizaje:
    $\theta := \theta - \alpha \nabla J(\theta)$

Se definieron:

* `theta_inicial`: vector de ceros
* $\theta$ (tasa de aprendizaje)
* `n_iteraciones`

También se graficó el **historial de coste** para visualizar la convergencia del algoritmo.
![alt text](<Regresion_Lineal.py/Grafico Historial de coste.png>)
**La gráfica mostró que el coste convergió de forma estable a un valor aproximado de X.XX después de unas YYY iteraciones.** (<- **¡RECUERDA REEMPLAZAR X.XX e YYY con tus valores!**)

---

## 7. 🤖 Función de Predicción

La función `predecir(X_nuevos, theta_final, mu, sigma)` permite usar el modelo entrenado con **nuevos datos**:

1.  Escala los nuevos datos con $\mu$ y $\sigma$ del entrenamiento.
2.  Añade la columna de unos (bias).
3.  Aplica la fórmula de regresión lineal ($h_\theta(X) = X \theta$) para predecir precios.

---

## ✅ Conclusión

Este modelo permite predecir el precio promedio de casas en California usando regresión lineal multivariable, correctamente entrenada y escalada.
Con el descenso de gradiente y la MSE como guía, podemos ajustar $\theta$ hasta encontrar una solución eficiente y precisa.

## Punto 1: Consolidar el Aprendizaje 🧠

Dependiendo del valor de **alpha**, podemos observar cuánto tiempo tarda en **converger** el algoritmo.

* Si el **alpha es muy pequeño**, el descenso de gradiente avanza muy lento.
* Si el **alpha es muy grande**, el algoritmo puede **omitir el aprendizaje** o incluso **divergir** (los valores crecen en lugar de estabilizarse).

📈
*Figura 1*
![alt text](Regresion_Lineal.py/Figure_1.png)

Gracias a la comparación de los valores en la gráfica, podemos encontrar un **alpha ideal**:


📉
*Figura 2*
![alt text](Regresion_Lineal.py/Figure_2.png)


Sin el **escalado**, el descenso de gradiente tarda más o simplemente **no converge** (pueden aparecer datos 'null').
En cambio, cuando **escalamos las características**:

* Las variables tienen una magnitud parecida.
* El algoritmo avanza mejor.
* Se pueden usar valores de alpha más grandes sin que se vuelva inestable.
* Todo converge de forma más rápida y eficiente.


*Gráfico del historial de coste*
![alt text](<Regresion_Lineal.py/Grafico Historial de coste.png>)


### ✅ En resumen:

* Escalar las características mejora la **eficiencia del algoritmo**.
* Elegir bien el valor de **alpha** hace que el modelo **converja más rápido** sin salirse de control.

##  Punto 2: Evaluar el Número de Iteraciones ⏱️

Con distintos valores de alpha, podemos observar que, aproximadamente a partir de las 2500 iteraciones, las curvas comienzan a aplanarse. Esto indica que el algoritmo empieza a converger, ya que el coste deja de disminuir significativamente.

En mi experimento utilicé 4000 iteraciones como número total. Elegí este valor porque, al probar varios valores de alpha, quería asegurarme de observar con claridad en qué punto cada curva se aplanaba por completo. Esto me permitió identificar con mayor precisión cuándo el algoritmo realmente comenzaba a converger en cada caso.

![alt text](Regresion_Lineal.py/Figure_1.2.png)



# Regresión Lineal: Ecuación Normal vs Descenso de Gradiente  

## Estructura de la Matriz **X** y el Vector **y**  
### **Matriz X (Diseño)**  
- **Contenido**:  
  - Columna de **unos (1)** para el intercepto (`θ₀`).  
  - Columnas de **características** (`X₁, X₂, ..., Xₙ`).  
- **Dimensiones**:  
  `m × (n+1)`  *(m observaciones, n características)*  

### **Vector y (Objetivo)**  
- **Contenido**: Valores reales a predecir.  
- **Dimensiones**:  
  `m × 1`  

## 🔍 **Comparaciones Clave**  
### 📊 Resultados Experimentales  
| **Escenario**            | Diferencia (Error) | Comparación Válida |  
|--------------------------|--------------------|--------------------|  
| Theta GD (escalado) vs Theta EN (**sin escalar**) | ≈ 111              | ❌ No (escalas distintas) |  
| Theta GD (escalado) vs Theta EN (**escalado**)    | ≈ 9.9              | ✅ Sí               |  

### ❓ **Interpretación**  
1. **Diferencia ≈ 111**:  
   - Ilustra cómo el **escalado afecta los valores absolutos de `θ`**.  
   - **No es válida técnicamente** (comparar `θ` en escalas diferentes no tiene sentido matemático).  

2. **Diferencia ≈ 9.9**:  
   - Muestra que el Descenso de Gradiente (**GD**) **no convergió totalmente** por falta de iteraciones.  

---

## 🧮 **Ecuación Normal: Fórmula e Implementación**  
### **Fórmula Analítica**  


θ = (Xᵗ X)⁻¹ Xᵗ y



### **Pasos de Implementación**  
1. Calcular `Xᵗ X`.  
2. Invertir la matriz resultante.  
3. Multiplicar por `Xᵗ y`.  

## ⚖️ **Pros y Contras**  
| **Método**           | **Ecuación Normal**                              | **Descenso de Gradiente**                     |  
|----------------------|--------------------------------------------------|-----------------------------------------------|  
| **Ventajas**         | - Solución exacta en 1 paso.<br>- Sin hiperparámetros.<br>- No requiere escalado. | - Escalable a grandes `n`.<br>- Funciona incluso si `Xᵗ X` es singular. |  
| **Desventajas**      | - Coste `O(n³)` (lento para `n > 10⁴`).<br>- Falla si `Xᵗ X` no es invertible. | - Necesita ajustar `α` e iteraciones.<br>- Requiere escalado para converger bien. |  

---

## 🚀 **¿Cuándo Usar Cada Método?**  
| **Criterio**               | **Ecuación Normal**          | **Descenso de Gradiente**       |  
|----------------------------|------------------------------|---------------------------------|  
| **Número de características** | `n < 10⁴`                 | `n ≥ 10⁴`                      |  
| **Estabilidad matricial**  | Evitar si `Xᵗ X` es singular | Funciona siempre               |  
| **Recursos computacionales** | Adecuado para CPU/GPU moderadas | Ideal para clusters distribuidos |  

---

**Notas Finales**:  
- Usar `np.linalg.pinv` en lugar de `inv` para manejar matrices singulares.  
- El escalado en GD es **crítico** para convergencia rápida y estable.  


## 🧪 Implementación de `train_test_split` y Comparación de Métodos de Regresión

### ✂️ División del Dataset

Para mejorar la estructura y la organización del flujo de trabajo, implementamos la función `train_test_split` de `sklearn.model_selection`. Esto nos permite separar claramente los datos de entrenamiento y prueba, y darles nombres consistentes como:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=...)
```

Este cambio mejora la legibilidad y facilita la gestión de las variables a lo largo del código.

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



### ⚖️ Comparación: Ecuación Normal vs Descenso de Gradiente

Se observaron métricas prácticamente idénticas para ambos métodos en este dataset, lo que es esperable bajo ciertas condiciones:

* El **Descenso de Gradiente (GD)** fue aplicado sobre datos **escalados**.
* La **Ecuación Normal (NE)** se aplicó sobre datos **sin escalar**, como es estándar cuando no se usa regularización.

Ambos métodos convergieron a soluciones **muy similares**, con diferencias menores en los decimales más significativos. Por ejemplo:

| Método | MSE (prueba) | Primeras 5 predicciones                   |
| ------ | ------------ | ----------------------------------------- |
| NE     | 0.5558915987 | \[0.7191, 1.7640, 2.7096, 2.8389, 2.6046] |
| GD     | 0.5558913314 | \[0.7191, 1.7640, 2.7096, 2.8389, 2.6046] |

Aunque los valores redondeados a 4 cifras son iguales (`0.5559`), un análisis más profundo muestra pequeñas diferencias, lo que indica que no son exactamente idénticos. Esto se visualizó claramente mediante un bloque de *debugging* con valores extendidos y predicciones parciales.

---

### 🛡️ Impacto de la Regularización: Cambio de λ (lambda)

Durante la experimentación con el parámetro de regularización `λ` (lambda) en el **Descenso de Gradiente**, se identificaron los siguientes comportamientos:

* Con **lambda = 0** o **valores muy pequeños**, el algoritmo a veces **diverge**, especialmente si no se escalan las características y se usa una **tasa de aprendizaje alta** (`α = 0.1`). Esto se manifiesta en la aparición de `nan` en los coeficientes (`theta`).
* Al **aumentar lambda** (por ejemplo, a 0.1 o más), la **regularización L2 estabiliza** las actualizaciones penalizando los valores grandes de `theta`. Esto reduce la magnitud de los saltos del algoritmo y permite una convergencia más estable, incluso sin escalado óptimo.

> Esta observación refuerza el papel de la regularización como herramienta no solo para evitar el sobreajuste, sino también para **mejorar la estabilidad numérica** del aprendizaje en condiciones no ideales.

---

