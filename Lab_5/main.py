import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


A, B = 0, 24


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


def plot_function():
    x = np.linspace(A, B, 1000)
    y = f(x)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color="steelblue", linewidth=2,
             label=r"$f(x)=50+20\sin\!\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$")
    plt.title("Графік функції навантаження на сервер")
    plt.xlabel("Час, x (год)")
    plt.ylabel("Навантаження, f(x)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_function.png", dpi=150)
    plt.show()
    print("Графік збережено: plot_function.png")


# Обчислення точного значення інтегралу
def exact_integral():
    I0, _ = quad(f, A, B, limit=200)
    return I0

# Функція яка за допомогою методу Сімпсона знаходить наближене значення означеного інтегралу
def simpson(func, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    coeff = np.ones(N + 1)
    coeff[1:-1:2] = 4
    coeff[2:-2:2] = 2
    return h / 3 * np.dot(coeff, y)

# Залежність точності від N
def study_accuracy(I0):
    N_values = range(10, 1001, 2)   # тільки парні
    errors = [abs(simpson(f, A, B, N) - I0) for N in N_values]

    eps_target = 1e-12
    N_opt = None
    for N, e in zip(N_values, errors):
        if e < eps_target:
            N_opt = N
            break

    plt.figure(figsize=(10, 5))
    plt.semilogy(list(N_values), errors, color="darkorange", linewidth=1.5)
    if N_opt:
        eps_opt = abs(simpson(f, A, B, N_opt) - I0)
        plt.axvline(N_opt, color="red", linestyle="--",
                    label=f"$N_{{opt}}={N_opt}$,  epsopt={eps_opt:.2e}")
    plt.axhline(eps_target, color="green", linestyle=":", label=r"$\varepsilon=10^{-12}$")
    plt.title(r"Залежність похибки від числа розбиттів $N$")
    plt.xlabel("N")
    plt.ylabel(r"$\varepsilon(N) = |I(N) - I_0|$")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("plot_accuracy.png", dpi=150)
    plt.show()
    print("Графік збережено: plot_accuracy.png")

    return N_opt



# Похибка при N0
def compute_N0(N_opt):
    raw = N_opt // 10
    N0 = max(32, (raw // 8) * 8)  # мінімум 32, кратне 8
    return N0


def error_at_N0(I0, N0):
    I_N0 = simpson(f, A, B, N0)
    eps0 = abs(I_N0 - I0)
    return I_N0, eps0



# Метод Рунге-Ромберга
def runge_romberg(I0, N0):
    I_full = simpson(f, A, B, N0)
    I_half = simpson(f, A, B, N0 // 2)
    I_R = I_full + (I_full - I_half) / (2**4 - 1)
    epsR = abs(I_R - I0)
    return I_R, epsR


# Метод Ейткена
def aitken(I0, N0):
    N1, N2, N3 = N0, N0 // 2, N0 // 4
    I1 = simpson(f, A, B, N1)
    I2 = simpson(f, A, B, N2)
    I3 = simpson(f, A, B, N3)

    ratio = (I1 - I2) / (I2 - I3) if abs(I2 - I3) > 1e-30 else float("nan")
    p = np.log2(abs(ratio)) if ratio > 0 else float("nan")

    I_A = I3 + (I3 - I2) / (2**p - 1) if not np.isnan(p) else I3
    epsA = abs(I_A - I0)
    return I_A, epsA, p


# Адаптивний алгоритм
def adaptive_simpson(func, a, b, eps, depth=0, max_depth=50):
    mid = (a + b) / 2
    # Сімпсон на проміжку  [a,b]
    fa, fm, fb = func(a), func(mid), func(b)
    S1 = (b - a) / 6 * (fa + 4 * fm + fb)
    # Сімпсон на двох половинах
    lm = (a + mid) / 2
    rm = (mid + b) / 2
    flm, frm = func(lm), func(rm)
    S2 = (mid - a) / 6 * (fa + 4 * flm + fm) + (b - mid) / 6 * (fm + 4 * frm + fb)

    calls = 5  # fa, fm, fb, flm, frm

    if depth >= max_depth or abs(S2 - S1) < 15 * eps:
        return S2 + (S2 - S1) / 15, calls

    left_val, left_calls = adaptive_simpson(func, a, mid, eps / 2, depth + 1, max_depth)
    right_val, right_calls = adaptive_simpson(func, mid, b, eps / 2, depth + 1, max_depth)
    return left_val + right_val, calls + left_calls + right_calls


def study_adaptive(I0):
    eps_list = [10**(-k) for k in range(3, 13)]
    errors_ad, calls_ad = [], []

    for eps in eps_list:
        val, calls = adaptive_simpson(f, A, B, eps)
        errors_ad.append(abs(val - I0))
        calls_ad.append(calls)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "steelblue"
    ax1.set_xlabel(r"$\varepsilon$ (параметр точності)")
    ax1.set_ylabel("Похибка", color=color1)
    ax1.loglog(eps_list, errors_ad, "o-", color=color1, label="Похибка")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    color2 = "darkorange"
    ax2.set_ylabel("Кількість обчислень f", color=color2)
    ax2.semilogx(eps_list, calls_ad, "s--", color=color2, label="Обчислення f")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Адаптивний алгоритм: точність і вартість")
    fig.tight_layout()
    plt.savefig("plot_adaptive.png", dpi=150)
    plt.show()
    print("Графік збережено: plot_adaptive.png")

    return errors_ad, calls_ad, eps_list




def main():
    print("=" * 60)
    print("  Лабораторна робота №5 — Чисельне інтегрування")
    print("=" * 60)


    print("\n[1] Побудова графіку функції...")
    plot_function()


    print("\n[2] Точне значення інтегралу (scipy.integrate.quad):")
    I0 = exact_integral()
    print(f"    I0 = {I0:.15f}")


    print("\n[3] Тест складової формули Сімпсона (N=100):")
    I_test = simpson(f, A, B, 100)
    print(f"    I(100) = {I_test:.15f}")
    print(f"    похибка = {abs(I_test - I0):.6e}")


    print("\n[4] Дослідження залежності точності від N (будуємо графік)...")
    N_opt = study_accuracy(I0)
    if N_opt is None:
        print("    Точність 1e-12 не досягнута в діапазоні N=10..1000")
        N_opt = 1000
    else:
        eps_opt = abs(simpson(f, A, B, N_opt) - I0)
        print(f"    N_opt = {N_opt}")
        print(f"    epsopt = {eps_opt:.6e}")


    N0 = compute_N0(N_opt)
    print(f"\n[5] N0 = {N0}  (≈ N_opt/10, кратне 8)")
    I_N0, eps0 = error_at_N0(I0, N0)
    print(f"    I(N0) = {I_N0:.15f}")
    print(f"    eps0  = {eps0:.6e}")


    print("\n[6] Метод Рунге-Ромберга:")
    I_R, epsR = runge_romberg(I0, N0)
    print(f"    I_R  = {I_R:.15f}")
    print(f"    epsR = {epsR:.6e}")
    print(f"    Покращення відносно eps0: {eps0/epsR:.1f}x" if epsR > 0 else "")


    print("\n[7] Метод Ейткена:")
    I_A, epsA, p = aitken(I0, N0)
    print(f"    Порядок точності p = {p:.4f}  (очікується ≈ 4)")
    print(f"    I_A  = {I_A:.15f}")
    print(f"    epsA = {epsA:.6e}")
    print(f"    Покращення відносно eps0: {eps0/epsA:.1f}x" if epsA > 0 else "")


    print("\n[8] Порівняння методів:")
    print(f"    {'Метод':<25} {'Значення':<22} {'Похибка':<15}")
    print("    " + "-" * 62)
    print(f"    {'Точне (scipy)':<25} {I0:<22.15f} {'—'}")
    print(f"    {'Сімпсон N0':<25} {I_N0:<22.15f} {eps0:.6e}")
    print(f"    {'Рунге-Ромберг':<25} {I_R:<22.15f} {epsR:.6e}")
    print(f"    {'Ейткен':<25} {I_A:<22.15f} {epsA:.6e}")


    print("\n[9] Адаптивний алгоритм (будуємо графік залежності)...")
    errors_ad, calls_ad, eps_list = study_adaptive(I0)
    print(f"\n    {'eps_param':<14} {'Похибка':<15} {'Обчислень f'}")
    print("    " + "-" * 45)
    for eps, err, c in zip(eps_list, errors_ad, calls_ad):
        print(f"    {eps:<14.0e} {err:<15.6e} {c}")

    print("\n" + "=" * 60)
    print("  Готово! Графіки збережено у поточну директорію.")
    print("=" * 60)



if __name__ == "__main__":
    main()