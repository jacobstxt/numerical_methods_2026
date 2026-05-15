import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2


def hooke_jeeves(f, x0, h0=0.5, beta=0.5, eps1=1e-7, eps2=1e-7, max_iter=50000):
    n = len(x0)
    x_base = np.array(x0, dtype=float)
    h = h0
    path = [x_base.copy()]

    def exploratory(x_start):
        x = x_start.copy()
        for i in range(n):
            fx = f(x)
            x_try = x.copy(); x_try[i] += h
            if f(x_try) < fx:
                x = x_try
            else:
                x_try = x.copy(); x_try[i] -= h
                if f(x_try) < fx:
                    x = x_try
        return x

    total_iters = 0
    while total_iters < max_iter:
        total_iters += 1
        x_exp = exploratory(x_base)

        if f(x_exp) < f(x_base):
            x_prev = x_base.copy()
            x_base = x_exp.copy()
            path.append(x_base.copy())

            while True:
                total_iters += 1
                x_pat = 2.0 * x_base - x_prev
                x_pat_exp = exploratory(x_pat)

                if f(x_pat_exp) < f(x_base):
                    x_prev = x_base.copy()
                    x_base = x_pat_exp.copy()
                    path.append(x_base.copy())
                    if (np.linalg.norm(x_base - x_prev) < eps2 and
                            abs(f(x_base) - f(x_prev)) < eps2):
                        return x_base, f(x_base), path, total_iters
                else:
                    break
        else:
            h *= beta
            if h < eps1:
                break

    return x_base, f(x_base), path, total_iters


def system_equations(x):
    f1 = x[0]**2 + x[1]**2 - 4.0
    f2 = x[0] * x[1] - 1.0
    return [f1, f2]


def objective_from_system(x):
    eqs = system_equations(x)
    return sum(e**2 for e in eqs)


def plot_results(path_r, path_s, x_sol):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#1e1e2e')
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in axes:
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#cdd6f4', labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#45475a')
    ax1, ax2, ax3, ax4 = axes

    x_r = np.linspace(-2.0, 2.0, 300); y_r = np.linspace(-0.5, 3.0, 300)
    X, Y = np.meshgrid(x_r, y_r)
    Z = 100*(Y - X**2)**2 + (1-X)**2
    ax1.contourf(X, Y, np.log1p(Z), levels=50, cmap='inferno')
    ax1.contour(X, Y, np.log1p(Z), levels=20, colors='white', linewidths=0.3, alpha=0.25)
    traj = np.array(path_r)
    ax1.plot(traj[:,0], traj[:,1], 'o-', color='#a6e3a1', lw=1.2, ms=3, alpha=0.85,
             label=f'Траєкторія ({len(path_r)} кроків)')
    ax1.plot(traj[0,0], traj[0,1], 's', color='#89b4fa', ms=8, label='Старт')
    ax1.plot(1, 1, '*', color='#f38ba8', ms=12, label='Мінімум (1,1)')
    ax1.set_title('Функція Розенброка\nТраєкторія спуску', color='#cdd6f4', fontsize=10)
    ax1.set_xlabel('x₁', color='#cdd6f4', fontsize=9); ax1.set_ylabel('x₂', color='#cdd6f4', fontsize=9)
    ax1.legend(fontsize=8, facecolor='#313244', labelcolor='#cdd6f4', framealpha=0.8)

    fv = [rosenbrock(p) for p in path_r]
    ax2.semilogy(fv, color='#89dceb', lw=1.5)
    ax2.set_title('Збіжність — Розенброк', color='#cdd6f4', fontsize=10)
    ax2.set_xlabel('Базисна точка (крок)', color='#cdd6f4', fontsize=9)
    ax2.set_ylabel('f(x) (лог. масштаб)', color='#cdd6f4', fontsize=9)
    ax2.grid(True, alpha=0.2, color='#45475a')

    xs = np.linspace(-2.8, 2.8, 500); ys = np.linspace(-2.8, 2.8, 500)
    Xs, Ys = np.meshgrid(xs, ys)
    ax3.contour(Xs, Ys, Xs**2+Ys**2-4, levels=[0], colors=['#89b4fa'], linewidths=2)
    ax3.contour(Xs, Ys, Xs*Ys-1, levels=[0], colors=['#a6e3a1'], linewidths=2)
    from matplotlib.lines import Line2D
    sols = [(0.5176,1.9319),(1.9319,0.5176),(-0.5176,-1.9319),(-1.9319,-0.5176)]
    ax3.plot(*zip(*sols), '*', color='#f38ba8', ms=10, label="Точні розв'язки")
    ax3.plot(x_sol[0], x_sol[1], 'D', color='#fab387', ms=10, markeredgecolor='white', mew=1.5,
             label=f'Знайдено: ({x_sol[0]:.4f}, {x_sol[1]:.4f})')
    ts = np.array(path_s)
    ax3.plot(ts[:,0], ts[:,1], '--', color='#cba6f7', lw=1.2, alpha=0.7, label='Траєкторія')
    ax3.plot(ts[0,0], ts[0,1], 's', color='#89b4fa', ms=8, label='Старт')
    ax3.set_xlim(-2.5, 2.5); ax3.set_ylim(-2.5, 2.5)
    ax3.set_title('Система рівнянь\nКриві та траєкторія', color='#cdd6f4', fontsize=10)
    ax3.set_xlabel('x', color='#cdd6f4', fontsize=9); ax3.set_ylabel('y', color='#cdd6f4', fontsize=9)
    legend_e = [Line2D([0],[0],c='#89b4fa',lw=2,label='x²+y²=4'),
                Line2D([0],[0],c='#a6e3a1',lw=2,label='xy=1')]
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles=legend_e+handles, labels=['x²+y²=4','xy=1']+labels,
               fontsize=8, facecolor='#313244', labelcolor='#cdd6f4', framealpha=0.8)
    ax3.grid(True, alpha=0.12, color='#45475a')
    ax3.axhline(0, color='#45475a', lw=0.5); ax3.axvline(0, color='#45475a', lw=0.5)

    fo = [objective_from_system(p) for p in path_s]
    ax4.semilogy(fo, color='#fab387', lw=1.5)
    ax4.set_title('Збіжність — система рівнянь', color='#cdd6f4', fontsize=10)
    ax4.set_xlabel('Базисна точка (крок)', color='#cdd6f4', fontsize=9)
    ax4.set_ylabel('F(x,y) (лог. масштаб)', color='#cdd6f4', fontsize=9)
    ax4.grid(True, alpha=0.2, color='#45475a')

    fig.suptitle('Лабораторна №9 — Метод Хука-Дживса', color='#cdd6f4', fontsize=13, fontweight='bold')
    plt.show()
    print("Графік збережено.")


def main():
    print("=" * 60)
    print("  Лабораторна робота №9 — Метод Хука-Дживса")
    print("=" * 60)

    print("\n[1] Тест: Функція Розенброка")
    print("    Мінімум: (1, 1), f = 0")
    x0_r = [-1.2, 0.5]
    x_opt, f_opt, path_r, iters_r = hooke_jeeves(
        rosenbrock, x0_r, h0=0.5, beta=0.5, eps1=1e-8, eps2=1e-8
    )
    print(f"    Початкова точка:    ({x0_r[0]}, {x0_r[1]})")
    print(f"    Знайдений мінімум:  ({x_opt[0]:.8f}, {x_opt[1]:.8f})")
    print(f"    f(x*) = {f_opt:.4e}")
    print(f"    Кроків траєкторії:  {len(path_r)}")
    print(f"    Ітерацій (всього):  {iters_r}")

    print("\n[2] Система нелінійних рівнянь:")
    print("    f₁(x,y) = x² + y² − 4 = 0")
    print("    f₂(x,y) = x·y − 1 = 0")
    print("    F(x,y) = f₁² + f₂²  →  min = 0")
    x0_s = [0.8, 1.8]
    x_sol, f_sol, path_s, iters_s = hooke_jeeves(
        objective_from_system, x0_s, h0=0.3, beta=0.5, eps1=1e-10, eps2=1e-10
    )
    eqs = system_equations(x_sol)
    print(f"\n    Початкова точка:    ({x0_s[0]}, {x0_s[1]})")
    print(f"    Знайдений розв'язок: x = {x_sol[0]:.10f}")
    print(f"                         y = {x_sol[1]:.10f}")
    print(f"    F(x,y) = {f_sol:.4e}")
    print(f"\n    Перевірка невязки:")
    print(f"      f₁ = x²+y²−4 = {eqs[0]:+.2e}  (має → 0)")
    print(f"      f₂ = x·y−1   = {eqs[1]:+.2e}  (має → 0)")
    print(f"\n    Кроків траєкторії:  {len(path_s)}")
    print(f"    Ітерацій (всього):  {iters_s}")

    print("\n[3] Траєкторія спуску (система рівнянь):")
    header = f"{'Крок':>5}  {'x':>14}  {'y':>14}  {'F(x,y)':>14}"
    print(f"    {header}")
    print(f"    {'-'*5}  {'-'*14}  {'-'*14}  {'-'*14}")
    total = len(path_s)
    indices = list(range(min(8, total)))
    if total > 13:
        indices += ['...'] + list(range(total-5, total))
    elif total > 8:
        indices += list(range(8, total))
    for idx in indices:
        if idx == '...':
            print("    ..."); continue
        p = path_s[idx]
        fv = objective_from_system(p)
        print(f"    {idx:>5}  {p[0]:>14.8f}  {p[1]:>14.8f}  {fv:>14.4e}")

    with open('trajectory.txt', 'w', encoding='utf-8') as file:
        file.write("Траєкторія спуску — система нелінійних рівнянь\n")
        file.write(f"Система: f1 = x^2 + y^2 - 4 = 0\n")
        file.write(f"         f2 = x*y - 1 = 0\n")
        file.write(f"Початкова точка: ({x0_s[0]}, {x0_s[1]})\n")
        file.write(f"Знайдений розв'язок: x = {x_sol[0]:.10f}, y = {x_sol[1]:.10f}\n")
        file.write(f"F(x,y) = {f_sol:.4e}\n")
        file.write(f"Кількість кроків: {len(path_s)}\n\n")
        file.write(f"{'Крок':>5}  {'x':>16}  {'y':>16}  {'F(x,y)':>14}\n")
        file.write(f"{'-' * 5}  {'-' * 16}  {'-' * 16}  {'-' * 14}\n")
        for i, p in enumerate(path_s):
            fv = objective_from_system(p)
            file.write(f"{i:>5}  {p[0]:>16.10f}  {p[1]:>16.10f}  {fv:>14.4e}\n")

    print("Траєкторію збережено у файл: trajectory.txt")


    plot_results(path_r, path_s, x_sol)
    print("\nГотово!")


if __name__ == "__main__":
    main()