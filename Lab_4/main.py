import math
import numpy as np
import matplotlib.pyplot as plt


def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def exact_derivative(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def central_difference(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)


def plot_Mt():
    t_values = np.linspace(0, 20, 1000)
    M_values = [M(t) for t in t_values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_values, M_values, color='steelblue', linewidth=2)
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('M(t)', fontsize=12)
    ax.set_title('Графік функції вологості ґрунту M(t)', fontsize=13)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()


def main():
    t0 = 1.0
    exact_val = exact_derivative(t0)
    print(f"1. Точне значення похідної в точці t0={t0}: {exact_val:.5f}\n")

    # --- Пункт 2. Дослідження залежності похибки від кроку h ---
    print("--- 2. Дослідження кроку h ---")
    best_h = None
    min_error = float('inf')

    # Перебираємо h від 10^-20 до 10^3
    h_values = [10 ** i for i in range(-20, 4)]

    for h in h_values:
        try:
            approx_val = central_difference(t0, h)
            error = abs(approx_val - exact_val)

            # Шукаємо мінімальну похибку
            if error < min_error:
                min_error = error
                best_h = h
        except ZeroDivisionError:
            # Ігноруємо випадки, коли через обмеження типу float 2*h стає рівним 0
            pass

    print(f"Оптимальний крок h0: {best_h:.1e}")
    print(f"Мінімальна похибка R0: {min_error:.5e}\n")

    # --- Пункти 3, 4, 5. Розрахунки для заданого кроку ---
    h_fixed = 10 ** -2
    print(f"--- 3-6. Метод Рунге-Ромберга (фіксований крок h = {h_fixed}) ---")

    # Значення похідної для кроків h та 2h
    y_prime_h = central_difference(t0, h_fixed)
    y_prime_2h = central_difference(t0, 2 * h_fixed)

    R1 = abs(y_prime_h - exact_val)
    print(f"Значення похідної (крок h): {y_prime_h:.5f}")
    print(f"Похибка при кроці h (R1): {R1:.5e}")

    # --- Пункт 6. Метод Рунге-Ромберга ---
    # Формула: y_R = y_0(h) + (y_0(h) - y_0(2h)) / 3
    y_R = y_prime_h + (y_prime_h - y_prime_2h) / 3
    R2 = abs(y_R - exact_val)

    print(f"Уточнене значення (Рунге-Ромберг): {y_R:.5f}")
    print(f"Похибка Рунге-Ромберга (R2): {R2:.5e}")
    if R2 > 0:
        print(f"Характер зміни: похибка зменшилась у {R1 / R2:.1f} разів\n")
    else:
        print("Характер зміни: похибка наближена до нуля\n")

    # --- Пункт 7. Метод Ейткена ---
    print("--- 7. Метод Ейткена ---")
    # Для Ейткена потрібен ще один крок - 4h
    y_prime_4h = central_difference(t0, 4 * h_fixed)

    # Чисельник та знаменник для формули Ейткена
    numerator = y_prime_2h ** 2 - y_prime_4h * y_prime_h
    denominator = 2 * y_prime_2h - (y_prime_4h + y_prime_h)

    if denominator != 0:
        y_E = numerator / denominator
        R3 = abs(y_E - exact_val)
        print(f"Уточнене значення (Ейткен): {y_E:.5f}")
        print(f"Похибка Ейткена (R3): {R3:.5e}")

        # Оцінка порядку точності формули (p)
        p_val = abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h))
        p = math.log(p_val) / math.log(2)
        print(f"Порядок точності формули p: {p:.2f}")
    else:
        print("Помилка обчислення: ділення на нуль (значення похідних майже однакові).")

    plot_Mt()



if __name__ == "__main__":
    main()