import pandas as pd
from sklearn.naive_bayes import GaussianNB

# llamamos el archivo csv con .read_csv()
df = pd.read_csv("Iris.csv")
# Esto es para ver los 5 primeras lineas
print(df.head())
# Esto para ver las ultimas 5 filas
print(df.tail())
# muestre directamente en la consola un resumen sobre tu tabla df.
df.info()
# Con esto obtenemos las estadisticas descriptivas del archivo csv
print(df.describe().round(3))
# Contar categorias
# .value_counts() para saber contar cuantas veces aparece cada valor en este caso 'Species'
print(df.value_counts("Species"))
print(df["Species"].value_counts())
# .iloc es para seleccionar filas y columnas por su posicion entera, por numeros
print(df.iloc[-1, [1, 3]])
# .loc es para seleccionar filas y columnas por sus etiquetas(nombres)
print(df.loc[149, ["SepalWidthCm", "PetalWidthCm"]])
# Filtrado conbinado queremos seleccionar/filtrar las filas que cumplan DOS condiciones a la vez:
print(df[(df["Species"] == "Iris-setosa") & (df["SepalLengthCm"] < 5)])


# Aqui empieza el Naives Bayes
# primero tenemos que cargar los datos asi como hicimos con lo csv
# Directamente desde numpy
import numpy as np
from sklearn.datasets import load_iris

# CARGAMOS LOS DATOS, cuando es con numpy es load_iris le damos un nombre
datos = load_iris()
# Para acceder a los datos numericos utilizamos .data
X = datos.data  # estos son arrays = NUMERO TOTAL DE FLORES
# Para acceder a etiquetas utilizamos .target las etiquetas de clase (0, 1, 2) para cada muestra.
Y = datos.target

# Siguiente Paso: SEPARAMOS DATOS POR CLASES
# Para nuestro Naive Bayes, necesitamos agrupar las flores por clase.
# y esto hacemos al tomar X (flores) y las agrupamos por clase Y (0,1 y 2), asi las ordenamos. EL NUMERO DE FLORES
X_clase0 = X[Y == 0]  #  crea una "máscara" de Verdadero/Falso (True/False), del mismo tamaño que Y. Es True donde Y vale 0, y False en el resto.
X_clase1 = X[Y == 1]
X_clase2 = X[Y == 2]

# Siguiente paso: CALCULAMOS ESTADISTICAS POR CLASE "ENTRENAMIENTO"
# ya separamos los datos ahora necesitamos calcular las estadisticas para cada clase
# Aqui calcularemos la MEDIA, el promedio  y se hace con np.mean(), se hace con las clases, para cada una de las 4 caracteristicas
media_clase0 = np.mean(X_clase0, axis=0)  # el axis = 0 Promedio por columnas (por características). Si fuera axis = 1, Promedio por filas (cada observación)
media_clase1 = np.mean(X_clase1, axis=0)
media_clase2 = np.mean(X_clase2, axis=0)
# Aqui calcularemos su desviación estándar np.std, La desviación estándar mide qué tanto se alejan los datos del promedio
std_clase0 = np.std(X_clase0, axis=0)
std_clase1 = np.std(X_clase1, axis=0)
std_clase2 = np.std(X_clase2, axis=0)
# Aqui calcularemos Su probabilidad previa (qué tan frecuente es esa clase).también llamada prior en estadística
# La PROBABILIDAD PREVIA es cuánto esperas que algo pase sin ver datos nuevos.
# Esto lo hacemos para saber que probabilidades tenemos con las flores por hacemos EL NUMERO  DE FLORES   /  EL NUMERO TOTAL DE FLORES, con las filas
prior_clase0 = (X_clase0.shape[0] / X.shape[0])  # El .shape te dice Te dice cuántas filas y columnas tiene. 0 = filas, 1 = columnas,
prior_clase1 = X_clase1.shape[0] / X.shape[0]
prior_clase2 = X_clase2.shape[0] / X.shape[0]
# ¡Fase de Cálculo de Estadísticas ("Entrenamiento") Lista! ✅
# Logramos los numeros nuestro modelo aprendio de los datos

# PREPARACION FINAL Antes de Predecir:
# AGRUPAMOS ESTADISTICAS, creando listas
lista_medias = [media_clase0, media_clase1, media_clase2]  # Estos son arrays
lista_stds = [std_clase0, std_clase1, std_clase2]
lista_priors = [prior_clase0, prior_clase1, prior_clase2]


def gaussian_pdf(x, mu, std):
    # Define un valor muy pequeño (épsilon) para añadir a la varianza.
    # Esto evita problemas numéricos, como la división por cero si la desviación estándar es muy cercana a cero.
    epsilon = 1e-9
    # Calcula el denominador del exponente en la fórmula de la PDF Gaussiana: 2 * (varianza + epsilon).
    denominador_exp = 2 * (std**2 + epsilon)
    # Calcula el exponente: -((x - mu)^2) / (2 * varianza).
    exponente = -((x - mu) ** 2) / denominador_exp
    # Calcula el factor de normalización: 1 / sqrt(2 * pi * varianza).
    factor_norm = 1 / np.sqrt(2 * np.pi * (std**2 + epsilon))
    # Calcula el valor final de la PDF multiplicando el factor de normalización por la exponencial del exponente.
    pdf_valor = factor_norm * np.exp(exponente)
    # Devuelve el valor o array de valores de la densidad de probabilidad calculada.
    return pdf_valor


# Primero def predecir las clases
def predecir_clases_nb(
    flor_nueva, lista_media, lista_stds, lista_priors
):  
    # Diseñada para usar el teorema de Bayes, que es la base de los clasificadores Naive Bayes.
    # crea una lista vacía nueva cada vez que llamamos a la función para predecir una flor. ✅
    
    posteriors = []  # Preparamos una "caja" vacía (posteriors) donde iremos metiendo los 3 scores (uno por cada clase) que vamos a calcular a continuación.
    # Ahora necesitamos hacer los cálculos para la clase 0, luego para la clase 1 y finalmente para la clase 2. Para eso, usamos un bucle for.
    for clase_idx in range(3):
        # Accede a los valores de la media, desviación estándar y probabilidad previa correspondientes a esa clase, utilizando el índice clase_idx.
        media_actual = lista_media[clase_idx]  # Esto selecciona la media correspondiente de la clase en la posición clase_idx.
        std_actual = lista_stds[clase_idx]  # Esto selecciona la desviación estándar correspondiente de la clase en la posición clase_idx.
        prior_actual = lista_priors[clase_idx]  # Esto selecciona la PROBABILIDAD PREVIA de la clase en la posición clase_idx.
        # Obtienes correctamente las estadísticas (media_actual, std_actual, prior_actual) para la clase específica (clase_idx) que se está procesando en esta vuelta del bucle. ✅
        """Lógica Explicada Fácil:
        En cada vuelta (para clase 0, luego 1, luego 2), sacamos de nuestras "listas maestras" los ingredientes específicos de esa clase: su perfil de medias, su perfil de dispersión (std dev) y qué tan común es (prior)."""

        # AHORA CALCULAMOS Likelihood
        # Es decir, qué tan "normal" o "esperable" serían las medidas de esta flor_nueva si realmente perteneciera a esta clase_idx.
        # Fórmula de Bayes: 
        # Llamaremos a nuestra funcion gaussian_pdf(x, mu, std), pero dandolo 3 variables (flor_nueva, media_actual y std_actual)
        # buscamos una nueva probabilidad, la de flor nueva
        probabilidades_pdf = gaussian_pdf(flor_nueva, media_actual, std_actual)
        # Ahora, tomamos las 4 probabilidades calculadas y las multiplicamos todas entre sí usando np.prod()
        # Esto nos da el "likelihood" o la probabilidad de que la flor nueva, con sus características, pertenezca a esta clase.
        # Esta es una aplicación de la asunción "naive" (independencia entre características).  
        likelihood = np.prod(probabilidades_pdf)
        # Posteriormente, multiplicamos el "likelihood" por el "prior" de la clase actual.
        # El "prior" es la probabilidad de que una flor pertenezca a esta clase sin tener en cuenta las características.
        # Esto nos da la probabilidad posterior (sin normalizar), que es la probabilidad de que la flor nueva
        # pertenezca a esta clase dada sus características.
        posterior_actual = likelihood * prior_actual
        # Finalmente, añadimos el valor de posterior_actual a la lista "posteriors".
        posteriors.append(posterior_actual)
        # ------ Fin de la funcion predecir_clases_nb -------

    # Paso Final: Decidir la Clase Predicha (DESPUÉS del Bucle)
    # el np.argmax nos ayuda a encontrar el índice del valor máximo en una lista o array
    # es buena idea convertirla a array NumPy, por que las operaciones vectorizadas sobre arrays son mucho más rápidas que los bucles manuales.
    posteriors_array = np.array(posteriors)
    clase_predicha = np.argmax(posteriors_array)
    return clase_predicha
    

# Probar nuestro clasificador predecir_clases_nb en todo el dataset X, calcular su precisión y compararla con la de scikit-learn.
# Parte 1: Probar NUESTRO clasificador
# Inicializa una lista vacía llamada 'predicciones' para almacenar las predicciones de clase para cada muestra del conjunto de datos.
predicciones = []
# Inicia un bucle 'for' que itera sobre los índices de las filas del array 'X' (desde 0 hasta el número total de muestras - 1).
for i in range(X.shape[0]):
    # Extrae la fila 'i' de los valores del array 'X', que corresponde a las características de la muestra actual.
    flor_actual = X[i]
    # Llama a la función 'predecir_clases_nb' para obtener la predicción de clase para la muestra 'flor_actual'.
    # Pasa la muestra actual y los parámetros del modelo (medias, stds, priors) a la función.
    prediccion = predecir_clases_nb(flor_actual, lista_medias, lista_stds, lista_priors)
    # Agregamos prediccion a la lista predicciones
    predicciones.append(prediccion)
    # ------- Fin del bucle for --------

# Siguiente Paso: Calcular la Precisión del Modelo
# convertimos a nuestra lista en un array para poder compararlo con Y
predicciones_np = np.array(predicciones)
# sumamos los datos de la lista y lo comparamos con Y
aciertos = np.sum(predicciones_np == Y)
# Aqui queremos saber como obtendriamos el numero total de muestras, con el numero total de filas 
total_muestras = X.shape[0]
# Aqui dividimos aciertos y total_muestras 
precision_propia = aciertos / total_muestras
# Imprimimos los valores con formato F-string para porcentaje
print(f"Precisión Naive Bayes propio {precision_propia * 100:.2f}% ")


# ----------------------------------------------------------------------------------
# Sección de comparación con la implementación de Naive Bayes de Scikit-learn
# ----------------------------------------------------------------------------------

# Traemos a nuestro modelo el GaussianNB()
modelo_sk = GaussianNB()

# Los modelos de Scikit-learn se "entrenan" trayendo el metodo .fit()
modelo_sk.fit(X, Y)
#  (Predecir): Una vez entrenado, para hacer predicciones usamos el método .predict()
predicciones_sk = modelo_sk.predict(X)
# sumamos los datos de la lista y lo comparamos con Y
aciertos_sk = np.sum(predicciones_sk == Y)
precision_sk = aciertos_sk / X.shape[0]
print(f"Precisión Naive Bayes Scikit-learn {precision_sk * 100:.2f}% ")
