import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



x_all = np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
y_all = np.array([3.0, 5.0, 11.0, 28.0, 85.0])


def divided_differences(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y.copy()
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table


def newton_predict(x_point, x_nodes, table):
    n = len(x_nodes)
    result = table[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x_point - x_nodes[i - 1])
        result += table[0, i] * product
    return result


def lagrange_predict(x_point, x_nodes, y_nodes):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if j != i:
                term *= (x_point - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


# ─────────────────────────────────────────────
# ДОСЛІДЖЕННЯ 1: ВПЛИВ КРОКУ
# ─────────────────────────────────────────────

print("=" * 60)
print("ДОСЛІДЖЕННЯ 1: Вплив кроку (фіксований інтервал)")
print("=" * 60)

# Генеруємо вузли рівномірно на фіксованому інтервалі
interval = (1000, 16000)
node_counts = [3, 4, 5, 7, 10]
TARGET = 6000

# Справжнє значення (5 вузлів як еталон)
table_ref = divided_differences(x_all, y_all)
ref_value = newton_predict(TARGET, x_all, table_ref)

print(f"\nФіксований інтервал: [{interval[0]}, {interval[1]}]")
print(f"Точка прогнозу: n = {TARGET}")
print(f"Еталонне значення (5 вузлів): {ref_value:.4f} мс\n")
print(f"{'Вузлів':>8} | {'Крок h':>10} | {'P(6000)':>10} | {'Похибка':>10} | {'Відн. похибка':>14}")
print("-" * 60)

research1_data = {}
for k in node_counts:
    x_k = np.linspace(interval[0], interval[1], k)
    # Інтерполюємо y значення для нових вузлів через еталонний многочлен
    y_k = np.array([newton_predict(xi, x_all, table_ref) for xi in x_k])
    table_k = divided_differences(x_k, y_k)
    pred = newton_predict(TARGET, x_k, table_k)
    h = (interval[1] - interval[0]) / (k - 1)
    err_abs = abs(pred - ref_value)
    err_rel = err_abs / abs(ref_value) * 100
    research1_data[k] = {'h': h, 'pred': pred, 'err_abs': err_abs, 'err_rel': err_rel,
                         'x_k': x_k, 'y_k': y_k, 'table': table_k}
    print(f"{k:>8} | {h:>10.0f} | {pred:>10.4f} | {err_abs:>10.4f} | {err_rel:>13.2f}%")

# ─────────────────────────────────────────────
# ДОСЛІДЖЕННЯ 2: ВПЛИВ КІЛЬКОСТІ ВУЗЛІВ
# Фіксований крок h=3000, змінний інтервал
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ДОСЛІДЖЕННЯ 2: Вплив кількості вузлів (фіксований крок)")
print("=" * 60)

FIXED_STEP = 3000
print(f"\nФіксований крок h = {FIXED_STEP}")
print(f"Точка прогнозу: n = {TARGET}\n")
print(f"{'Вузлів':>8} | {'Інтервал':>18} | {'P(6000)':>10} | {'Похибка':>10} | {'Відн. похибка':>14}")
print("-" * 65)

research2_data = {}
for k in range(2, 8):
    x_k = np.array([1000 + i * FIXED_STEP for i in range(k)])
    y_k = np.array([newton_predict(xi, x_all, table_ref) for xi in x_k])
    table_k = divided_differences(x_k, y_k)

    if TARGET < x_k[0] or TARGET > x_k[-1]:
        status = "(екстрапол.)"
    else:
        status = "(інтерпол.)"

    pred = newton_predict(TARGET, x_k, table_k)
    err_abs = abs(pred - ref_value)
    err_rel = err_abs / abs(ref_value) * 100
    interval_str = f"[{int(x_k[0])}, {int(x_k[-1])}]"
    research2_data[k] = {'pred': pred, 'err_abs': err_abs, 'err_rel': err_rel,
                         'x_k': x_k, 'y_k': y_k, 'table': table_k, 'status': status}
    print(f"{k:>8} | {interval_str:>18} | {pred:>10.4f} | {err_abs:>10.4f} | {err_rel:>13.2f}% {status}")

# ─────────────────────────────────────────────
# ДОСЛІДЖЕННЯ 3: ЕФЕКТ РУНГЕ
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ДОСЛІДЖЕННЯ 3: Ефект Рунге")
print("=" * 60)

x_plot = np.linspace(1000, 17000, 600)
runge_nodes = [3, 4, 5, 7]

print(f"\nМаксимальні відхилення від еталону за межами вузлів:")
print(f"{'Вузлів':>8} | {'Max похибка':>14} | {'При n':>8}")
print("-" * 40)

runge_data = {}
ref_curve = np.array([newton_predict(xi, x_all, table_ref) for xi in x_plot])

for k in node_counts:
    d = research1_data[k]
    curve = np.array([newton_predict(xi, d['x_k'], d['table']) for xi in x_plot])
    err_curve = np.abs(curve - ref_curve)
    max_err = np.max(err_curve)
    max_n = x_plot[np.argmax(err_curve)]
    runge_data[k] = {'curve': curve, 'err': err_curve}
    print(f"{k:>8} | {max_err:>14.4f} | {max_n:>8.0f}")

# ─────────────────────────────────────────────
# ДОСЛІДЖЕННЯ 4: ПОРІВНЯННЯ З ЛАГРАНЖЕМ
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ДОСЛІДЖЕННЯ 4: Порівняння Newton vs Lagrange")
print("=" * 60)

print(f"\nПрогноз при n = {TARGET}:")
print(f"{'Вузлів':>8} | {'Newton':>10} | {'Lagrange':>10} | {'Різниця':>12}")
print("-" * 48)

lagrange_data = {}
for k in [3, 4, 5]:
    x_k = x_all[:k]
    y_k = y_all[:k]
    table_k = divided_differences(x_k, y_k)

    newton_val = newton_predict(TARGET, x_k, table_k)
    lagrange_val = lagrange_predict(TARGET, x_k, y_k)
    diff = abs(newton_val - lagrange_val)
    lagrange_data[k] = {'newton': newton_val, 'lagrange': lagrange_val}
    print(f"{k:>8} | {newton_val:>10.6f} | {lagrange_val:>10.6f} | {diff:>12.8f}")

print("\n→ Методи Ньютона і Лагранжа дають ОДНАКОВИЙ результат")
print("  (різниця ~0 через числову точність float64)")

# ─────────────────────────────────────────────
# ПОБУДОВА ГРАФІКІВ
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Дослідницька частина: Аналіз інтерполяції многочленом Ньютона",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

colors_map = {3: '#e74c3c', 4: '#e67e22', 5: '#2ecc71', 7: '#3498db', 10: '#9b59b6'}

# ── Графік 1: Вплив кроку ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Дослід 1: Вплив кроку\n(фіксований інтервал [1000, 16000])", fontsize=11)

for k in node_counts:
    d = research1_data[k]
    curve = np.array([newton_predict(xi, d['x_k'], d['table']) for xi in x_plot])
    ax1.plot(x_plot, curve, color=colors_map[k], linewidth=1.8,
             label=f"{k} вузлів, h={int(d['h'])}")

ax1.scatter(x_all, y_all, color='black', zorder=6, s=60, label="Реальні дані", marker='D')
ax1.axvline(TARGET, color='purple', linestyle='--', linewidth=1.2, label=f"n={TARGET}")
ax1.set_xlabel("Розмір вхідних даних n", fontsize=10)
ax1.set_ylabel("Час виконання t (мс)", fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-10, 130)

# ── Графік 2: Вплив кількості вузлів ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Дослід 2: Вплив кількості вузлів\n(фіксований крок h=3000)", fontsize=11)

cmap2 = plt.cm.viridis(np.linspace(0.1, 0.9, len(research2_data)))
for idx, (k, d) in enumerate(research2_data.items()):
    if k < 2:
        continue
    curve = np.array([newton_predict(xi, d['x_k'], d['table']) for xi in x_plot])
    ax2.plot(x_plot, curve, color=cmap2[idx], linewidth=1.8,
             label=f"{k} вузлів {d['status']}")

ax2.scatter(x_all, y_all, color='black', zorder=6, s=60, marker='D', label="Реальні дані")
ax2.axvline(TARGET, color='purple', linestyle='--', linewidth=1.2, label=f"n={TARGET}")
ax2.set_xlabel("Розмір вхідних даних n", fontsize=10)
ax2.set_ylabel("Час виконання t (мс)", fontsize=10)
ax2.legend(fontsize=7.5)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-20, 150)

# ── Графік 3: Ефект Рунге ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title("Дослід 3: Ефект Рунге\n(похибка відносно еталону P₅)", fontsize=11)

for k in node_counts:
    if k == 5:
        continue
    ax3.plot(x_plot, runge_data[k]['err'], color=colors_map[k],
             linewidth=1.8, label=f"|P₅ - P_{k}|, {k} вузлів")

ax3.axvline(TARGET, color='purple', linestyle='--', linewidth=1.2, label=f"n={TARGET}")
ax3.fill_between([x_all[0], x_all[-1]], 0, 5, alpha=0.07, color='green', label="Зона вузлів")
ax3.set_xlabel("Розмір вхідних даних n", fontsize=10)
ax3.set_ylabel("Абсолютна похибка (мс)", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── Графік 4: Newton vs Lagrange ──────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title("Дослід 4: Порівняння Ньютона та Лагранжа\n(порівняння кривих при 5 вузлах)", fontsize=11)

# Newton
table5 = divided_differences(x_all, y_all)
newton_curve = np.array([newton_predict(xi, x_all, table5) for xi in x_plot])
lagrange_curve = np.array([lagrange_predict(xi, x_all, y_all) for xi in x_plot])
diff_curve = np.abs(newton_curve - lagrange_curve)

ax4.plot(x_plot, newton_curve, color='#2980b9', linewidth=2.5, label="Newton P₅(n)", zorder=3)
ax4.plot(x_plot, lagrange_curve, color='#e74c3c', linewidth=1.5,
         linestyle='--', label="Lagrange L₅(n)", zorder=4)

ax4_twin = ax4.twinx()
ax4_twin.plot(x_plot, diff_curve, color='gray', linewidth=1, linestyle=':', label="|N - L|")
ax4_twin.set_ylabel("|Newton - Lagrange|", fontsize=9, color='gray')
ax4_twin.tick_params(axis='y', labelcolor='gray')
ax4_twin.set_ylim(0, 1e-10)

ax4.scatter(x_all, y_all, color='black', zorder=6, s=60, marker='D', label="Реальні дані")
ax4.axvline(TARGET, color='purple', linestyle='--', linewidth=1.2, label=f"n={TARGET}")
ax4.set_xlabel("Розмір вхідних даних n", fontsize=10)
ax4.set_ylabel("Час виконання t (мс)", fontsize=10)
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax4.grid(True, alpha=0.3)

plt.show()
print("\nГрафіки збережено у research_plots.png")