import numpy as np
import os


N = 100
X_TRUE_VAL = 2.5
EPS = 1e-14
MAX_ITER = 100



def generate_system(n, x_val):
    np.random.seed(42)
    A = np.random.uniform(-100, 100, size=(n, n))

    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + np.random.uniform(10, 50)

    X_true = np.full(n, x_val)
    B = A @ X_true

    return A, B, X_true


def save_matrix(filename, M):
    np.savetxt(filename, M, fmt="%.15e")
    print(f"    Записано: {filename} ({M.shape})")


def save_vector(filename, v):
    np.savetxt(filename, v, fmt="%.15e")
    print(f"    Записано: {filename} (розмір {len(v)})")


def read_matrix(filename, n):
    M = np.loadtxt(filename).reshape(n, n)
    return M


def read_vector(filename):
    return np.loadtxt(filename)



def lu_decomposition(A):
    """
    LU-розклад матриці A = L·U.
    L — нижня трикутна матриця,
    U — верхня трикутна матриця з одиницями на діагоналі.

    Алгоритм:
    1. Ініціалізуємо L = 0, U = 0
    2. Діагональ U: u_ii = 1
    3. Почергово обчислюємо k-й стовпець L та k-й рядок U:
       l_ik = a_ik - Σ(j=1..k-1) l_ij * u_jk,  для i >= k
       u_ki = (a_ki - Σ(j=1..k-1) l_kj * u_ji) / l_kk,  для i > k
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Діагональ U = 1
    for i in range(n):
        U[i, i] = 1.0

    for k in range(n):
        # Обчислюємо k-й стовпець матриці L (елементи l_ik, i >= k)
        for i in range(k, n):
            s = 0.0
            for j in range(k):
                s += L[i, j] * U[j, k]
            L[i, k] = A[i, k] - s

        # Обчислюємо k-й рядок матриці U (елементи u_ki, i > k)
        for i in range(k + 1, n):
            s = 0.0
            for j in range(k):
                s += L[k, j] * U[j, i]
            U[k, i] = (A[k, i] - s) / L[k, k]

    return L, U


def save_lu(filename_l, filename_u, L, U):
    save_matrix(filename_l, L)
    save_matrix(filename_u, U)


# ═══════════════════════════════════════════════
#  Розв'язок системи за допомогою LU-розкладу
# ═══════════════════════════════════════════════
def solve_lu(L, U, B):
    """
    Розв'язує AX = B, де A = L·U.
    Крок 1: LZ = B (пряма підстановка)
    Крок 2: UX = Z (зворотна підстановка)
    """
    n = len(B)

    # Пряма підстановка: LZ = B
    Z = np.zeros(n)
    Z[0] = B[0] / L[0, 0]
    for k in range(1, n):
        s = 0.0
        for j in range(k):
            s += L[k, j] * Z[j]
        Z[k] = (B[k] - s) / L[k, k]

    # Зворотна підстановка: UX = Z
    X = np.zeros(n)
    X[n - 1] = Z[n - 1]
    for k in range(n - 2, -1, -1):
        s = 0.0
        for j in range(k + 1, n):
            s += U[k, j] * X[j]
        X[k] = Z[k] - s

    return X


# ═══════════════════════════════════════════════
# 4. Допоміжні функції
# ═══════════════════════════════════════════════
def mat_vec_mult(A, x):
    """Обчислює добуток матриці A на вектор x."""
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        result[i] = s
    return result


def vector_norm_inf(v):
    """Обчислює нескінченну норму вектора (максимум модулів)."""
    return np.max(np.abs(v))


def compute_residual_norm(A, X, B):
    """
    Обчислює точність розв'язку:
    eps = max_i |Σ_j a_ij * x_j - b_i|
    """
    AX = mat_vec_mult(A, X)
    return vector_norm_inf(AX - B)


# ═══════════════════════════════════════════════
# 5. Ітераційне уточнення розв'язку
# ═══════════════════════════════════════════════
def iterative_refinement(A, L, U, B, X0, eps, max_iter):
    """
    Ітераційне уточнення розв'язку СЛАР.

    Алгоритм:
    1. Обчислюємо нев'язку R = B - A·X0
    2. Розв'язуємо A·ΔX = R (через LU-розклад)
    3. Уточнюємо: X = X0 + ΔX
    4. Перевіряємо умови збіжності:
       - ||ΔX|| < eps
       - ||AX - B|| < eps
    5. Якщо не збіглось — повторюємо
    """
    X = X0.copy()
    print(f"\n    {'Ітерація':<12} {'||ΔX||':<20} {'||AX - B||':<20}")
    print("    " + "-" * 52)

    for iteration in range(1, max_iter + 1):
        # Обчислюємо нев'язку
        AX = mat_vec_mult(A, X)
        R = B - AX

        # Розв'язуємо A·ΔX = R через LU
        dX = solve_lu(L, U, R)

        # Уточнюємо розв'язок
        X = X + dX

        # Перевіряємо умови збіжності
        norm_dX = vector_norm_inf(dX)
        norm_res = compute_residual_norm(A, X, B)

        print(f"    {iteration:<12} {norm_dX:<20.6e} {norm_res:<20.6e}")

        if norm_dX < eps:
            print(f"\n    Збіжність досягнута за {iteration} ітерацій!")
            print(f"    (||AX-B|| обмежена машинною точністю: {norm_res:.2e})")
            return X, iteration

    print(f"\n    Увага: збіжність не досягнута за {max_iter} ітерацій.")
    return X, max_iter


# ═══════════════════════════════════════════════
# Головна програма
# ═══════════════════════════════════════════════
def main():
    print("\n[1] Генерація матриці A та вектора B...")
    A, B, X_true = generate_system(N, X_TRUE_VAL)
    save_matrix("matrix_A.txt", A)
    save_vector("vector_B.txt", B)
    print(f"    Розмірність: {N}×{N}")
    print(f"    Точний розв'язок: всі x_i = {X_TRUE_VAL}")


    print("\n[2] LU-розклад матриці A...")
    L, U = lu_decomposition(A)
    save_lu("matrix_L.txt", "matrix_U.txt", L, U)


    LU_product = L @ U
    decomp_error = vector_norm_inf((A - LU_product).flatten())
    print(f"    Перевірка: ||A - L·U|| = {decomp_error:.6e}")


    print("\n[3] Розв'язок системи AX = B через LU-розклад...")
    X0 = solve_lu(L, U, B)


    print(f"    Перші 5 елементів X:  {X0[:5]}")
    print(f"    Останні 5 елементів X: {X0[-5:]}")


    print("\n[4] Оцінка точності розв'язку...")
    eps_initial = compute_residual_norm(A, X0, B)
    error_initial = vector_norm_inf(X0 - X_true)
    print(f"    ||AX - B|| = {eps_initial:.6e}")
    print(f"    ||X - X_true|| = {error_initial:.6e}")


    print(f"\n[5] Ітераційне уточнення (eps = {EPS:.0e})...")
    X_refined, num_iter = iterative_refinement(A, L, U, B, X0, EPS, MAX_ITER)


    eps_final = compute_residual_norm(A, X_refined, B)
    error_final = vector_norm_inf(X_refined - X_true)


    print(f"\n[6] Порівняння результатів:")
    print(f"    {'Показник':<30} {'До уточнення':<20} {'Після уточнення':<20}")
    print("    " + "-" * 70)
    label1 = "||AX - B|| (невязка)"
    label2 = "||X - X_true|| (похибка)"
    label3 = "Кількість ітерацій"
    print(f"    {label1:<30} {eps_initial:<20.6e} {eps_final:<20.6e}")
    print(f"    {label2:<30} {error_initial:<20.6e} {error_final:<20.6e}")
    print(f"    {label3:<30} {'—':<20} {num_iter}")

    if eps_final < eps_initial:
        print(f"\n    Точність покращилась у {eps_initial / eps_final:.1f} разів!")

if __name__ == "__main__":
    main()