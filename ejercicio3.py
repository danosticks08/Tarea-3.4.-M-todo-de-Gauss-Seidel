import numpy as np
import matplotlib.pyplot as plt

def jacobi(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    x_prev = np.zeros(n)
    errors = []
    
    for k in range(max_iter):
        for i in range(n):
            sum_Ax = sum(A[i][j] * x_prev[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_Ax) / A[i][i]
        
        error = np.linalg.norm(x - x_prev, ord=np.inf)
        errors.append(error)
        if error < tol:
            break
        x_prev = x.copy()
    
    return x, errors

# Definir la matriz de coeficientes A y el vector b
A = np.array([
    [12, -2, 1, 0, 0, 0, 0],
    [-3, 18, -4, 1, 0, 0, 0],
    [0, -2, 16, -1, 1, 0, 0],
    [0, 2, -1, 11, -3, 1, 0],
    [0, 0, -2, 4, 15, -2, 1],
    [0, 0, 0, -2, -3, 2, 13],
    [0, 0, 0, 0, 0, -3, 13]
])

b = np.array([20, 35, -5, 19, -12, 25, 7])

# Resolver el sistema con Jacobi
solution, errors = jacobi(A, b)

# Comparar con la solución exacta
exact_solution = np.linalg.solve(A, b)

# Mostrar soluciones
print("Solución Aproximada:", solution)
print("Solución Exacta:", exact_solution)

# Graficar los errores
plt.plot(errors, marker='o', linestyle='-')
plt.xlabel('Iteraciones')
plt.ylabel('Error Absoluto')
plt.title('Convergencia del Método de Jacobi')
plt.grid()
plt.show()
