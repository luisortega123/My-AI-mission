# sumar_matrices
def sumar_matrices(matriz1, matriz2):

    filas1 = len(matriz1)
    if filas1 == 0:
        print("Error: La matriz 1 está vacía.")
        return None
    cols1 = len(matriz1[0])

    filas2 = len(matriz2)
    if filas2 == 0:
        print("Error: La matriz 2 está vacía.")
        return None
    cols2 = len(matriz2[0])

    if filas1 != filas2 or cols1 != cols2:
        print("Error: Las dimensiones de las matrices no coinciden para la suma.")
        return None

    matriz_suma = []
    for i in range(filas1):
        una_fila_de_ceros = [0] * cols1
        matriz_suma.append(una_fila_de_ceros)
    for i in range(filas1):  # Recorre filas i
        for j in range(cols1):  # Recorre columnas j
            suma_elemento = matriz1[i][j] + matriz2[i][j]

            matriz_suma[i][j] = suma_elemento
    return matriz_suma

# multiplicar_matriz_por_escalar
def multiplicar_matriz_por_escalar(matriz1, escalar):

    if not matriz1 or not matriz1[0]:
        print("Error: La matriz está vacía o mal formada.")
        return

    if not isinstance(escalar, (int, float)):
        print("Error: El escalar debe ser un número (entero o decimal).")
        return
    
    fila = len(matriz1)
    col = len(matriz1[0])
# esto sirve para recorrer la matriz 
    matriz_mult = [] 
    for i in range(fila):
        fila_cero = [0] * col
        matriz_mult.append(fila_cero)

    for i in range(fila):
        for j in range(col):
            mult_elemento = matriz1[i][j] * escalar
            matriz_mult[i][j] = mult_elemento

    return  matriz_mult

# --- Prueba para multiplicar_matriz_por_escalar ---
matriz_test = [[1, 2, 3], [4, 5, 6]]
escalar_test = 3
resultado = multiplicar_matriz_por_escalar(matriz_test, escalar_test)
print(f"Matriz original: {matriz_test}")
print(f"Escalar: {escalar_test}")
print(f"Resultado: {resultado}")
# Esperaríamos: [[3, 6, 9], [12, 15, 18]]

print("---")

# Prueba con escalar no numérico
resultado_error = multiplicar_matriz_por_escalar(matriz_test, "hola")
print(f"Resultado con error: {resultado_error}")
# Esperaríamos mensaje de error y None

# matriz_traspuesta
def matriz_traspuesta(matriz):

    if not matriz or not matriz[0]:
        print("Error: La matriz está vacía o mal formada.")
        return None
    
    fila_org = len(matriz)
    col_org = len(matriz[0])

    matriz_trasp = []

    for i in range(col_org):
        fila_cero = [0] * fila_org
        matriz_trasp.append(fila_cero)

    for i in range(fila_org):
        for j in range(col_org):

            matriz_trasp[j][i] = matriz[i][j] 
    return matriz_trasp  
    

matriz_test = [[1, 2, 3], 
            [4, 5, 6]]
traspuesta = matriz_traspuesta(matriz_test)
print(f"Matriz Original (2x3):\n{matriz_test}") 

print("Matriz Traspuesta (3x2):")
if traspuesta:
    for fila in traspuesta:
        print(fila) 

# multiplicar_matrices
def multiplicar_matrices(matriz_a, matriz_b):

    if not matriz_a or not matriz_a[0]:
        print("Error: La matriz A está vacía o mal formada.")
        return None
    if not matriz_b or not matriz_b[0]:
        print("Error: La matriz B está vacía o mal formada.")
        return None

    fila_a = len(matriz_a)
    col_a = len(matriz_a[0])
    fila_b = len(matriz_b)
    col_b = len(matriz_b[0]) 

    if col_a != fila_b:
        print("Error: La matriz A y la B no son compatibles para multiplicacion")
        return None

    matriz_c = []
    for i in range(fila_a):
        fila_cero = [0] * col_b
        matriz_c.append(fila_cero)

    for i in range(fila_a):
        for j in range(col_b):
            suma_productos = 0

            for k in range(col_a):
                suma_productos += matriz_a[i][k] * matriz_b[k][j]
            matriz_c[i][j] = suma_productos
    return matriz_c


# --- Código para probar multiplicar_matrices ---

print("\n" + "--- Prueba multiplicar_matrices ---") # Separador

matriz_test_a = [[1, 2], 
                 [3, 4]]
matriz_test_b = [[5, 6], 
                 [7, 8]]

print(f"Matriz A:\n{matriz_test_a}")
print(f"Matriz B:\n{matriz_test_b}")

resultado_producto = multiplicar_matrices(matriz_test_a, matriz_test_b)

print("Resultado A * B:")
if resultado_producto:
    for fila in resultado_producto:
        print(fila)
        
# Resultado esperado:
# [19, 22]
# [43, 50]

print("-" * 20)

# (Opcional: añadir prueba con matrices incompatibles)
matriz_incompatible = [[1, 1], [1, 1], [1, 1]] # Es 3x2
print("Probando A(2x2) * Incompatible(3x2):")
resultado_error_mult = multiplicar_matrices(matriz_test_a, matriz_incompatible)
print(f"Resultado: {resultado_error_mult}") # Esperamos mensaje de error y None