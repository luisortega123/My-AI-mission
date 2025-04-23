# Función 1: Sumar Matrices


def sumar_matrices(matriz1, matriz2):
    # --- Validaciones Iniciales ---
    # Asegurar que la matriz 1 es una estructura válida (no vacía, filas no vacías)
    if not matriz1 or not matriz1[0]:
        raise ValueError("Error: La matriz 1 está vacía o mal formada.")

    # Asegurar que la matriz 2 es una estructura válida
    if not matriz2 or not matriz2[0]:
        raise ValueError("Error: La matriz 2 está vacía o mal formada.")

    # --- Obtener Dimensiones ---
    filas1 = len(matriz1)
    cols1 = len(matriz1[0])
    filas2 = len(matriz2)
    cols2 = len(matriz2[0])

    # --- Validación de Compatibilidad ---
    # Requisito para la suma: las matrices deben tener exactamente las mismas dimensiones.
    if filas1 != filas2 or cols1 != cols2:
        raise ValueError(
            "Error: Las dimensiones de las matrices no coinciden para la suma."
        )

    # --- Inicialización de Matriz Resultado ---
    # Crear la estructura de la matriz resultado con las dimensiones correctas (filas1 x cols1)
    # e inicializarla con ceros.
    matriz_suma = []
    for i in range(filas1):
        una_fila_de_ceros = [0] * cols1
        matriz_suma.append(una_fila_de_ceros)

    # --- Cálculo de la Suma ---
    # Recorrer cada celda (i, j) y calcular la suma elemento a elemento.
    for i in range(filas1):
        for j in range(cols1):
            matriz_suma[i][j] = matriz1[i][j] + matriz2[i][j]

    return matriz_suma


# Función 2: Multiplicar Matriz por Escalar


def multiplicar_matriz_por_escalar(matriz, escalar):
    # --- Validaciones ---
    # Asegurar que la matriz es válida
    if not matriz or not matriz[0]:
        raise ValueError("Error: La matriz está vacía o mal formada.")

    # Asegurar que el escalar es un tipo numérico
    if not isinstance(escalar, (int, float)):
        raise TypeError("Error: El escalar debe ser un número (entero o decimal).")

    # --- Obtener Dimensiones ---
    filas = len(matriz)
    columnas = len(matriz[0])

    # --- Inicialización de Matriz Resultado ---
    # Crear matriz resultado con las mismas dimensiones que la original, inicializada con ceros.
    matriz_resultado = []
    for i in range(filas):
        fila_ceros = [0] * columnas
        matriz_resultado.append(fila_ceros)

    # --- Cálculo de la Multiplicación ---
    # Recorrer cada celda (i, j) y multiplicar el elemento original por el escalar.
    for i in range(filas):
        for j in range(columnas):
            matriz_resultado[i][j] = matriz[i][j] * escalar

    return matriz_resultado


# Función 3: Obtener Traspuesta de Matriz


def matriz_traspuesta(matriz):
    # --- Validación ---
    # Asegurar que la matriz de entrada es válida
    if not matriz or not matriz[0]:
        raise ValueError("Error: La matriz está vacía o mal formada.")

    # --- Obtener Dimensiones Originales ---
    fila_org = len(matriz)
    col_org = len(matriz[0])

    # --- Crear Matriz Traspuesta (Inicializada) ---
    # La traspuesta tendrá dimensiones intercambiadas (col_org x fila_org).
    matriz_trasp = []
    for i in range(col_org):  # La traspuesta tiene 'col_org' filas
        fila_cero = [0] * fila_org  # Cada fila tiene 'fila_org' columnas
        matriz_trasp.append(fila_cero)

    # --- Rellenar Matriz Traspuesta (Intercambiando índices) ---
    # Recorrer la matriz ORIGINAL para copiar los elementos
    for i in range(fila_org):  # i recorre filas originales
        for j in range(col_org):  # j recorre columnas originales
            # El elemento original [i][j] se asigna a la posición traspuesta [j][i]
            matriz_trasp[j][i] = matriz[i][j]

    return matriz_trasp


# Función 4: Multiplicar Matrices


def multiplicar_matrices(matriz_a, matriz_b):
    # --- Validaciones Iniciales ---
    # Validar matriz A
    if not matriz_a or not matriz_a[0]:
        raise ValueError("Error: La matriz A está vacía o mal formada.")
    # Validar matriz B
    if not matriz_b or not matriz_b[0]:
        raise ValueError("Error: La matriz B está vacía o mal formada.")

    # --- Obtener Dimensiones ---
    fila_a = len(matriz_a)
    col_a = len(matriz_a[0])
    fila_b = len(matriz_b)
    col_b = len(matriz_b[0])

    # --- Validación de Compatibilidad ---
    # Requisito para multiplicación A(m x n) * B(p x q): n debe ser igual a p (col_a == fila_b).
    if col_a != fila_b:
        # Usar f-string para un mensaje más informativo con las dimensiones
        raise ValueError(
            f"Dimensiones incompatibles para multiplicación: A({fila_a}x{col_a}), B({fila_b}x{col_b})"
        )

    # --- Inicialización de Matriz Resultado C ---
    # El resultado C tendrá dimensiones fila_a x col_b.
    matriz_c = []
    for i in range(fila_a):  # C tiene fila_a filas
        fila_cero = [0] * col_b  # C tiene col_b columnas
        matriz_c.append(fila_cero)

    # --- Cálculo de la Multiplicación (Producto Punto) ---
    # Recorrer cada celda (i, j) de la matriz resultado C.
    for i in range(fila_a):
        for j in range(col_b):
            # Calcular el producto punto de la fila i de A y la columna j de B para C[i][j].
            suma_productos = 0
            # k recorre las columnas de A y las filas de B (col_a = fila_b).
            for k in range(col_a):
                suma_productos += matriz_a[i][k] * matriz_b[k][j]
            # Asignar el resultado del producto punto a la celda correspondiente de C.
            matriz_c[i][j] = suma_productos

    return matriz_c


# -----------------------------------------------------------------------------
# Sección de Pruebas con Assert
# -----------------------------------------------------------------------------
print("Ejecutando pruebas básicas...")

# --- Pruebas para sumar_matrices ---
print("Probando sumar_matrices...")
try:
    # Prueba 1: Suma simple 2x2
    m_s1 = [[1, 1], [1, 1]]
    m_s2 = [[2, 3], [4, 5]]
    resultado_s = sumar_matrices(m_s1, m_s2)
    esperado_s = [[3, 4], [5, 6]]
    assert (
        resultado_s == esperado_s
    ), f"Error en sumar_matrices: Se esperaba {esperado_s} pero se obtuvo {resultado_s}"
    print("  Suma simple: OK")

    # Prueba 2: Suma con negativos/cero
    m_s3 = [[-1, 0], [5, -2]]
    m_s4 = [[1, -3], [-5, 2]]
    resultado_s2 = sumar_matrices(m_s3, m_s4)
    esperado_s2 = [[0, -3], [0, 0]]
    assert (
        resultado_s2 == esperado_s2
    ), f"Error en sumar_matrices (negativos): Se esperaba {esperado_s2} pero se obtuvo {resultado_s2}"
    print("  Suma con negativos/cero: OK")

    # Prueba 3: Error dimensiones incompatibles (debería lanzar ValueError)
    m_s5 = [[1], [2]]  # 2x1
    m_s6 = [[1, 2]]  # 1x2
    error_detectado_suma = False
    try:
        sumar_matrices(m_s5, m_s6)
    except ValueError as e:
        # ¡Correcto! Esperábamos un ValueError
        error_detectado_suma = True
        print(
            f"  Suma incompatibles (ValueError): OK ({e})"
        )  # Imprime el mensaje del error
    assert (
        error_detectado_suma
    ), "Error en sumar_matrices: No lanzó ValueError con dimensiones incompatibles."

except AssertionError as e:
    # Si algún assert falla, se imprime el mensaje de error del assert
    print(e)
except Exception as e:
    # Captura cualquier otro error inesperado durante las pruebas de suma
    print(f"Ocurrió un error inesperado probando sumar_matrices: {e}")


# --- Pruebas para multiplicar_matriz_por_escalar ---
print("Probando multiplicar_matriz_por_escalar...")

# Prueba 1: Prueba escalar simple
m_test_e1 = [[1, 2], [3, 4]]
esc_test1 = 3
resultado_e1 = multiplicar_matriz_por_escalar(m_test_e1, esc_test1)
esperado_e1 = [[3, 6], [9, 12]]

assert (
    esperado_e1 == resultado_e1
), f"Error en prueba escalar simple: Se esperaba {esperado_e1} pero se obtuvo {resultado_e1}"
print("Prueba escalar simple: OK")

# Prueba 2: Prueba escalar cero
esc_test2 = 0
resultado_e2 = multiplicar_matriz_por_escalar(m_test_e1, esc_test2)
esperado_e2 = [[0, 0], [0, 0]]

assert (
    esperado_e2 == resultado_e2
), f"Error en prueba escalar cero: Se esperaba {esperado_e2} pero se obtuvo {resultado_e2}"
print("Prueba escalar cero: OK")

# Prueba 3: Prueba escalar negativo
m_test_e3 = [[1, -2], [0, 3]]
esc_test3 = -2
resultado_e3 = multiplicar_matriz_por_escalar(m_test_e3, esc_test3)
esperado_e3 = [[-2, 4], [0, -6]]

assert (
    esperado_e3 == resultado_e3
), f"Error en prueba escalar negativo: Se esperaba {esperado_e3} pero se obtuvo {resultado_e3}"
print("Prueba escalar negativo: OK")

# Prueba 4: try-except
try:
    multiplicar_matriz_por_escalar(m_test_e1, "hola")
    assert False, "Se esperaba TypeError pero no se lanzó."

except TypeError:
    print("Prueba error tipo escalar: OK")

# Prueba 5: try-except
try:
    multiplicar_matriz_por_escalar([], 5)
    assert False, "Se esperaba ValueError pero no se lanzó."

except ValueError:
    print("Error matriz inválida pasó OK.")


# --- Pruebas para matriz_traspuesta ---
print("\nProbando matriz_traspuesta...")  # (Añadí un \n para mejor espaciado)

# Prueba 1: Traspuesta matriz 2x3
m_t1 = [[1, 2, 3], [4, 5, 6]]
resultado_t1 = matriz_traspuesta(m_t1)
esperado_t1 = [[1, 4], [2, 5], [3, 6]]
assert (
    resultado_t1 == esperado_t1
), f"Error en prueba traspuesta 2x3: Se esperaba {esperado_t1} pero se obtuvo {resultado_t1}"
print("  Prueba traspuesta 2x3: OK")

# Prueba 2: Traspuesta matriz 2x2
m_t2 = [[10, 20], [30, 40]]
r_t2 = matriz_traspuesta(m_t2)
esp_t2 = [[10, 30], [20, 40]]
assert (
    r_t2 == esp_t2
), f"Error en prueba traspuesta 2x2: Se esperaba {esp_t2} pero se obtuvo {r_t2}"
print("  Prueba traspuesta 2x2: OK")

# Prueba 3: Traspuesta matriz 3x1
m_t3 = [[10], [20], [30]]
r_t3 = matriz_traspuesta(m_t3)
esp_t3 = [[10, 20, 30]]
assert (
    r_t3 == esp_t3
), f"Error en prueba traspuesta 1x3: Se esperaba {esp_t3} pero se obtuvo {r_t3}"
print("  Prueba traspuesta 1x3: OK")

# Prueba 4: Traspuesta matriz 1x3
m_t4 = [[10, 20, 30]]
r_t4 = matriz_traspuesta(m_t4)
esp_t4 = [[10], [20], [30]]
assert (
    r_t4 == esp_t4
), f"Error en prueba traspuesta 1x3: Se esperaba {esp_t4} pero se obtuvo {r_t4}"
print("  Prueba traspuesta 1x3: OK")

# Prueba 5: Traspuesta matriz invalida
try:
    matriz_traspuesta([[]])
    assert False, "Se esperaba ValueError pero no se lanzó."

except ValueError:
    print("  Prueba error tipo ValueError: OK")

# --- Pruebas para multiplicar_matrices ---
print("Probando multiplicar_matrices...")

# Prueba 1: Multiplicar matriz
m_a_t1 = [[1, 2], [3, 4]]
m_b_t1 = [[5, 6], [7, 8]]
m_r_t1 = multiplicar_matrices(m_a_t1, m_b_t1)
m_esp_t1 = [[19, 22], [43, 50]]
assert (
    m_r_t1 == m_esp_t1
), f"Error en prueba de multiplicar 2x2: Se esperaba {m_esp_t1} pero se obtuvo {m_r_t1}"
print("  Prueba multiplicar 2x2: OK")

# Prueba 2: Multiplicar matriz
m_a_t2 = [[1, 2, 3], [4, 5, 6]]
m_b_t2 = [[7, 8], [9, 1], [2, 3]]
r_m2 = multiplicar_matrices(m_a_t2, m_b_t2)
esp_m2 = [[31, 19], [85, 55]]
assert (
    r_m2 == esp_m2
), f"Error en prueba de multiplicar : Se esperaba {esp_m2} pero se obtuvo {r_m2}"
print("  Prueba multiplicar rectangular: OK")

# Prueba 3:
# Caso 3: Dimensiones incompatibles (ej: A 2x2 * B 3x1).
try:
    m_a_t3 = [[1, 2], [3, 4]]
    m_b_t3 = [[1], [2], [3]]
    multiplicar_matrices(m_a_t3, m_b_t3)
    assert False, "Se esperaba ValueError (mult incompatible) pero no se lanzó."
except ValueError as e:  # Captura ValueError
    print(f"  Prueba mult incompatibles: OK ({e})")  # Mensaje OK

# Prueba 4
# Caso 4: Matriz B inválida (ej: A válida * B []).
try:
    m_a_t4 = [[1, 2], [3, 4]]
    m_b_t4 = []
    multiplicar_matrices(m_a_t4, m_b_t4)
    assert False, "Se esperaba ValueError (matriz B inválida) pero no se lanzó."
except ValueError as e:  # Captura ValueError
    print(f"  Prueba error matriz B inválida: OK ({e})")  # Mensaje OK

# Prueba 5
# Caso 5: Matriz A inválida (ej: A [] * B válida).
try:
    m_a_t5 = []
    m_b_t5 = [[5, 6], [7, 8]]
    multiplicar_matrices(m_a_t5, m_b_t5)
    assert False, "Se esperaba ValueError (matriz A inválida) pero no se lanzó."
except ValueError as e:  # Captura ValueError
    print(f"  Prueba error matriz A inválida: OK ({e})")  # Mensaje OK

print("-" * 20)
print("Fin de las pruebas.")
# Nota: Si alguna prueba falla con assert, el script se detendrá ahí.
# Si falla con una excepción inesperada, también se detendrá (a menos que usemos try/except más general).
