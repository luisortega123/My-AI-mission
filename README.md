# My-AI-mission
# Misión 1: Álgebra Lineal con Python Puro

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
