import math

EPS = 1e-5



def F(x):
    return x - 2 * math.sin(x) - 0.5


def dF(x):
    return 1 - 2 * math.cos(x)


def d2F(x):
    return 2 * math.sin(x)



def tabulate(a, b, h, filename="tabulation.txt"):
    intervals = []
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{'x':>10} {'F(x)':>15} {'знак':>6}\n")
        f.write("-" * 35 + "\n")

        x_prev = a
        f_prev = F(x_prev)
        f.write(f"{x_prev:>10.4f} {f_prev:>15.6f} "
                f"{'+' if f_prev >= 0 else '-':>6}\n")

        x = a + h
        while x <= b + 1e-12:
            f_cur = F(x)
            f.write(f"{x:>10.4f} {f_cur:>15.6f} "
                    f"{'+' if f_cur >= 0 else '-':>6}\n")
            # Зміна знаку = між x_prev і x є корінь
            if f_prev * f_cur < 0:
                intervals.append((x_prev, x))
            x_prev, f_prev = x, f_cur
            x += h
    return intervals



def stop_criterion(x_new, x_old, eps=EPS):
    return abs(x_new - x_old) < eps and abs(F(x_new)) < eps



def simple_iteration(x0, eps=EPS, max_iter=1000):
    tau = -1.0 / dF(x0)
    x_old = x0
    for k in range(1, max_iter + 1):
        x_new = x_old + tau * F(x_old)
        if stop_criterion(x_new, x_old, eps):
            return x_new, k
        x_old = x_new
    return x_old, max_iter



def newton(x0, eps=EPS, max_iter=1000):
    x_old = x0
    for k in range(1, max_iter + 1):
        fx = F(x_old)
        dfx = dF(x_old)
        if dfx == 0:
            raise ZeroDivisionError("F'(x) = 0 у методі Ньютона")
        x_new = x_old - fx / dfx
        if stop_criterion(x_new, x_old, eps):
            return x_new, k
        x_old = x_new
    return x_old, max_iter



def chebyshev(x0, eps=EPS, max_iter=1000):
    x_old = x0
    for k in range(1, max_iter + 1):
        fx = F(x_old)
        dfx = dF(x_old)
        d2fx = d2F(x_old)
        x_new = x_old - fx / dfx - (fx ** 2 * d2fx) / (2 * dfx ** 3)
        if stop_criterion(x_new, x_old, eps):
            return x_new, k
        x_old = x_new
    return x_old, max_iter



def secant(x0, x1, eps=EPS, max_iter=1000):
    x_prev, x_cur = x0, x1
    for k in range(1, max_iter + 1):
        f_prev = F(x_prev)
        f_cur = F(x_cur)
        if f_cur - f_prev == 0:
            raise ZeroDivisionError("Ділення на нуль у методі хорд")
        x_new = x_cur - f_cur * (x_cur - x_prev) / (f_cur - f_prev)
        if stop_criterion(x_new, x_cur, eps):
            return x_new, k
        x_prev, x_cur = x_cur, x_new
    return x_cur, max_iter



def parabola(x0, x1, x2, eps=EPS, max_iter=1000):
    a, b, c = x0, x1, x2
    for k in range(1, max_iter + 1):
        fa, fb, fc = F(a), F(b), F(c)
        h1 = b - a
        h2 = c - b
        d1 = (fb - fa) / h1
        d2 = (fc - fb) / h2
        A = (d2 - d1) / (h2 + h1)
        B = A * h2 + d2
        C = fc
        D = B * B - 4 * A * C
        # Якщо D<0, корінь інтерполянта комплексний
        if D < 0:
            sqrtD = complex(0, math.sqrt(-D))
        else:
            sqrtD = math.sqrt(D)
        # Обираємо знаменник з більшим модулем -> поправка менша
        denom1 = B + sqrtD
        denom2 = B - sqrtD
        denom = denom1 if abs(denom1) >= abs(denom2) else denom2
        dx = -2 * C / denom
        x_new = c + dx
        # Для дійсного кореня беремо real-part
        if isinstance(x_new, complex):
            x_new = x_new.real
        if stop_criterion(x_new, c, eps):
            return x_new, k
        a, b, c = b, c, x_new
    return c, max_iter



def inverse_interpolation(x0, x1, x2, eps=EPS, max_iter=1000):
    a, b, c = x0, x1, x2
    for k in range(1, max_iter + 1):
        fa, fb, fc = F(a), F(b), F(c)
        # Базисні многочлени Лагранжа в точці F=0
        La = (-fb) * (-fc) / ((fa - fb) * (fa - fc))
        Lb = (-fa) * (-fc) / ((fb - fa) * (fb - fc))
        Lc = (-fa) * (-fb) / ((fc - fa) * (fc - fb))
        x_new = a * La + b * Lb + c * Lc
        if stop_criterion(x_new, c, eps):
            return x_new, k
        a, b, c = b, c, x_new
    return c, max_iter



def main():
    print("=" * 72)
    print("F(x) = x - 2*sin(x) - 0.5 = 0")
    print("=" * 72)

    a_tab, b_tab, h_tab = -3.0, 3.0, 0.25
    intervals = tabulate(a_tab, b_tab, h_tab)

    print(f"\n[Завдання 1] Табуляція на [{a_tab}, {b_tab}] з кроком {h_tab}")
    print(f"            Знайдено {len(intervals)} інтервал(ів) зі зміною знаку:")
    for left, right in intervals:
        mid = (left + right) / 2
        deriv = dF(mid)
        behavior = "зростає" if deriv > 0 else "спадає"
        print(f"              [{left:+.2f}; {right:+.2f}]  "
              f"F'≈{deriv:+.3f} -> функція {behavior}")

    # Обираємо два корені з РІЗНОЮ поведінкою (як вимагає завдання 1)
    # Беремо перші два інтервали — там зростання і спадання
    root_growing_interval = intervals[0]
    root_decreasing_interval = intervals[1]

    cases = [
        ("Корінь #1 (функція зростає)",
         root_growing_interval[0], root_growing_interval[1]),
        ("Корінь #2 (функція спадає)",
         root_decreasing_interval[0], root_decreasing_interval[1]),
    ]

    # --- Завдання 2-4: уточнення коренів 6 методами ---
    print(f"\n[Завдання 2-4] Уточнення коренів (eps={EPS})")

    for case_name, a_int, b_int in cases:
        x0 = (a_int + b_int) / 2
        print(f"\n{case_name}, x0 = {x0:+.4f}")
        print(f"{'Метод':<28}{'Корінь':>16}{'F(корінь)':>16}{'Ітер.':>8}")
        print("-" * 68)

        results = []
        # Однокрокові: потрібен один x0
        r, k = simple_iteration(x0)
        results.append(("Проста ітерація", r, k))
        r, k = newton(x0)
        results.append(("Ньютона", r, k))
        r, k = chebyshev(x0)
        results.append(("Чебишева", r, k))
        # Багатокрокові: потрібно 2 або 3 початкові точки
        r, k = secant(a_int, b_int)
        results.append(("Хорд", r, k))
        r, k = parabola(a_int, (a_int + b_int) / 2, b_int)
        results.append(("Парабол", r, k))
        r, k = inverse_interpolation(a_int, (a_int + b_int) / 2, b_int)
        results.append(("Зворотної інтерполяції", r, k))

        for name, root, iters in results:
            print(f"{name:<28}{root:>+16.8f}{F(root):>+16.2e}{iters:>8}")


if __name__ == "__main__":
    main()