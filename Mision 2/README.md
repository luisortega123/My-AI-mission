## ¿Qué es PCA?
PCA, por sus siglas (Análisis de Componentes Principales), es una técnica que se usa para reducir la cantidad de variables en un conjunto de datos, sin perder demasiada información.

# ¿Para qué sirve?
* Nos permite visualizar datos complejos. Por ejemplo, podemos pasar de 4 dimensiones a 2 y graficarlos.

* También podemos eliminar variables que no aportan mucho o que no son tan importantes.

# Pasos clave de mi implementación
1. Cargamos el dataset de Iris, que tiene 4 variables por flor.
2. Centramos los datos, restando la media de cada característica.
3. Calculamos la matriz de covarianza. Esta matriz es como una tabla que encuentra relaciones entre varias variables al mismo tiempo (por ejemplo, si suben o bajan juntas).
4. Calculamos los valores propios y vectores propios:
    * Los valores propios nos dicen cuánta "importancia" tiene cada dirección de variación.

    * Los vectores propios nos indican en qué dirección ocurren esas variaciones.
5. Ordenamos por importancia y elegimos las direcciones con mayor varianza.
6. Seleccionamos los dos primeros componentes principales para proyectar los datos, y así poder visualizarlos.

