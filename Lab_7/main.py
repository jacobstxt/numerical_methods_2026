import numpy as np


N = 100
X_TRUE_VAL = 2.5
EPS = 1e-14
MAX_ITER = 100000



def generate_system(n, x_val):
    np.random.seed(42)
    A = np.random.uniform(-100, 100, size=(n, n))

    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) + np.random.uniform(10, 50)

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
    return np.loadtxt(filename).reshape(n, n)


def read_vector(filename):
    return np.loadtxt(filename)



def mat_vec_mult(A, x):
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        result[i] = s
    return result


def vector_norm_inf(v):
    return np.max(np.abs(v))


def matrix_norm_inf(A):
    n = len(A)
    max_sum = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += abs(A[i, j])
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum


def compute_residual_norm(A, X, B):
    AX = mat_vec_mult(A, X)
    return vector_norm_inf(AX - B)



def simple_iteration(A, f, X0, eps, max_iter):
    n = len(f)
    norm_A = matrix_norm_inf(A)
    tau = 2.0 / norm_A  # параметр τ: 0 < τ < 2/||A||

    # Перевірка збіжності: ||C|| = ||E - τA|| < 1
    C = np.eye(n) - tau * A
    norm_C = matrix_norm_inf(C)

    X = X0.copy()

    for k in range(1, max_iter + 1):
        AX = mat_vec_mult(A, X)
        X_new = np.zeros(n)
        for i in range(n):
            X_new[i] = X[i] - tau * (AX[i] - f[i])

        # Перевірка збіжності
        norm_diff = vector_norm_inf(X_new - X)
        X = X_new

        if norm_diff < eps:
            return X, k, norm_C

    return X, max_iter, norm_C



def jacobi(A, f, X0, eps, max_iter):
    n = len(f)
    X = X0.copy()

    # Норма матриці ітерації C = -D^(-1)(A- + A+)
    C_jacobi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C_jacobi[i, j] = -A[i, j] / A[i, i]
    norm_C = matrix_norm_inf(C_jacobi)

    for k in range(1, max_iter + 1):
        X_new = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * X[j]
            X_new[i] = (f[i] - s) / A[i, i]

        norm_diff = vector_norm_inf(X_new - X)
        X = X_new

        if norm_diff < eps:
            return X, k, norm_C

    return X, max_iter, norm_C



def seidel(A, f, X0, eps, max_iter):
    n = len(f)
    X = X0.copy()

    for k in range(1, max_iter + 1):
        X_old = X.copy()

        for i in range(n):
            s1 = 0.0
            # Вже оновлені значення (j < i)
            for j in range(i):
                s1 += A[i, j] * X[j]
            s2 = 0.0
            # Старі значення (j > i)
            for j in range(i + 1, n):
                s2 += A[i, j] * X[j]
            X[i] = (f[i] - s1 - s2) / A[i, i]

        norm_diff = vector_norm_inf(X - X_old)

        if norm_diff < eps:
            return X, k

    return X, max_iter



def main():
    print("\n[1] Генерація матриці A та вектора B...")
    A, B, X_true = generate_system(N, X_TRUE_VAL)
    save_matrix("matrix_A.txt", A)
    save_vector("vector_B.txt", B)
    print(f"    Розмірність: {N}×{N}")
    print(f"    Точний розв'язок: всі x_i = {X_TRUE_VAL}")


    diag_dom = True
    for i in range(N):
        off_diag = sum(abs(A[i, j]) for j in range(N) if j != i)
        if abs(A[i, i]) <= off_diag:
            diag_dom = False
            break
    print(f"    Діагональне переважання: {'Так' if diag_dom else 'Ні'}")


    print(f"\n[2] Характеристики матриці:")
    norm_A = matrix_norm_inf(A)
    print(f"    ||A||_inf = {norm_A:.6e}")


    X0 = np.array([1.0 * (i + 1) for i in range(N)])
    print(f"\n[3] Початкове наближення: x0_i = 1.0 * i, i = 1..{N}")
    print(f"    X0[:5] = {X0[:5]}")
    print(f"    ||X0 - X_true|| = {vector_norm_inf(X0 - X_true):.6e}")


    print(f"\n[4] Метод простої ітерації (eps = {EPS:.0e})...")
    X_si, iter_si, norm_C_si = simple_iteration(A, B, X0, EPS, MAX_ITER)
    err_si = vector_norm_inf(X_si - X_true)
    res_si = compute_residual_norm(A, X_si, B)
    print(f"    ||C|| = {norm_C_si:.6f} ({'збіжний' if norm_C_si < 1 else 'НЕ збіжний'})")
    print(f"    Ітерацій: {iter_si}")
    print(f"    ||X - X_true|| = {err_si:.6e}")
    print(f"    ||AX - B|| = {res_si:.6e}")


    print(f"\n[5] Метод Якобі (eps = {EPS:.0e})...")
    X_jac, iter_jac, norm_C_jac = jacobi(A, B, X0, EPS, MAX_ITER)
    err_jac = vector_norm_inf(X_jac - X_true)
    res_jac = compute_residual_norm(A, X_jac, B)
    print(f"    ||C_jacobi|| = {norm_C_jac:.6f} ({'збіжний' if norm_C_jac < 1 else 'НЕ збіжний'})")
    print(f"    Ітерацій: {iter_jac}")
    print(f"    ||X - X_true|| = {err_jac:.6e}")
    print(f"    ||AX - B|| = {res_jac:.6e}")


    print(f"\n[6] Метод Зейделя (eps = {EPS:.0e})...")
    X_seid, iter_seid = seidel(A, B, X0, EPS, MAX_ITER)
    err_seid = vector_norm_inf(X_seid - X_true)
    res_seid = compute_residual_norm(A, X_seid, B)
    print(f"    Ітерацій: {iter_seid}")
    print(f"    ||X - X_true|| = {err_seid:.6e}")
    print(f"    ||AX - B|| = {res_seid:.6e}")


    print(f"\n[7] Порівняння методів:")
    header = f"    {'Метод':<25} {'Ітерацій':<12} {'||X-X_true||':<18} {'||AX-B||':<18}"
    print(header)
    print("    " + "-" * 73)
    print(f"    {'Проста ітерація':<25} {iter_si:<12} {err_si:<18.6e} {res_si:<18.6e}")
    print(f"    {'Якобі':<25} {iter_jac:<12} {err_jac:<18.6e} {res_jac:<18.6e}")
    print(f"    {'Зейдель':<25} {iter_seid:<12} {err_seid:<18.6e} {res_seid:<18.6e}")

    if iter_seid < iter_jac < iter_si:
        print(f"\n    Зейдель збіжний найшвидше ({iter_seid} іт.), потім Якобі ({iter_jac} іт.),")
        print(f"    потім проста ітерація ({iter_si} іт.).")
    elif iter_seid < iter_jac:
        print(f"\n    Зейдель збіжний швидше за Якобі ({iter_seid} vs {iter_jac} ітерацій).")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()