import numpy as np
import matplotlib.pyplot as plt
import csv


data = [
    (1000,  3),
    (2000,  5),
    (4000, 11),
    (8000, 28),
    (16000, 85),
]

csv_filename = "data.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "t_ms"])
    writer.writerows(data)

print(f"Дані збережено у {csv_filename}")


# зчитування з CSV

def read_csv(filename):
    x_vals, y_vals = [], []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_vals.append(float(row["n"]))
            y_vals.append(float(row["t_ms"]))
    return np.array(x_vals), np.array(y_vals)

x_all, y_all = read_csv(csv_filename)
print(f"\nЗчитано {len(x_all)} вузлів з файлу:")
for xi, yi in zip(x_all, y_all):
    print(f"  n={int(xi):6d}  t={yi} мс")

# ─────────────────────────────────────────────
# 3. ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ
# ─────────────────────────────────────────────

def divided_differences(x, y):
    """Повертає таблицю розділених різниць."""
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y.copy()
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x[i+j] - x[i])
    return table

def print_divided_diff_table(x, table):
    n = len(x)
    header = f"{'x':>8} | {'f[x]':>10}" + "".join(f" | {'Δ^'+str(j):>12}" for j in range(1, n))
    print("\nТаблиця розділених різниць:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{x[i]:>8.0f} | {table[i,0]:>10.4f}"
        for j in range(1, n - i):
            row += f" | {table[i,j]:>12.6f}"
        print(row)
    print("-" * len(header))

# ─────────────────────────────────────────────
# 4. МНОГОЧЛЕН НЬЮТОНА
# ─────────────────────────────────────────────

def newton_predict(x_point, x_nodes, table):
    """Обчислює значення многочлена Ньютона у точці x_point."""
    n = len(x_nodes)
    result = table[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x_point - x_nodes[i-1])
        result += table[0, i] * product
    return result

# ─────────────────────────────────────────────
# 5. ДОСЛІДЖЕННЯ ПРИ РІЗНІЙ КІЛЬКОСТІ ВУЗЛІВ
# ─────────────────────────────────────────────

TARGET_N = 6000
node_counts = [3, 4, 5]

print(f"\n{'='*55}")
print(f"Прогноз часу виконання при n = {TARGET_N}")
print(f"{'='*55}")

results = {}
for k in node_counts:
    x_k = x_all[:k]
    y_k = y_all[:k]
    table_k = divided_differences(x_k, y_k)
    pred = newton_predict(TARGET_N, x_k, table_k)
    results[k] = pred
    print(f"  {k} вузли: P_{k}({TARGET_N}) = {pred:.4f} мс")



ref = results[5]
print(f"\n  Відносна похибка (відносно 5 вузлів):")
for k in [3, 4]:
    err = abs(results[k] - ref) / abs(ref) * 100
    print(f"  {k} вузли: похибка = {err:.2f}%")

# ─────────────────────────────────────────────
# 6. ГРАФІКИ
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Інтерполяція многочленом Ньютона\nПрогнозування часу виконання алгоритму",
             fontsize=13, fontweight="bold")

x_plot = np.linspace(x_all[0], x_all[-1] * 1.05, 500)
colors = {3: "orange", 4: "green", 5: "blue"}
labels = {3: "3 вузли", 4: "4 вузли", 5: "5 вузлів"}

# --- Графік 1: Інтерполяційні криві ---
ax1 = axes[0]
for k in node_counts:
    x_k = x_all[:k]
    y_k = y_all[:k]
    table_k = divided_differences(x_k, y_k)
    y_plot = [newton_predict(xi, x_k, table_k) for xi in x_plot]
    ax1.plot(x_plot, y_plot, color=colors[k], label=f"$P_{k}(n)$ — {labels[k]}", linewidth=1.8)

ax1.scatter(x_all, y_all, color="red", zorder=5, s=70, label="Експериментальні точки")
ax1.axvline(TARGET_N, color="purple", linestyle="--", linewidth=1.2, label=f"n = {TARGET_N}")
for k in node_counts:
    ax1.scatter(TARGET_N, results[k], color=colors[k], marker="*", s=150, zorder=6)

ax1.set_xlabel("Розмір вхідних даних n", fontsize=11)
ax1.set_ylabel("Час виконання t (мс)", fontsize=11)
ax1.set_title("Інтерполяційні криві Ньютона", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Графік 2: Похибка при різній кількості вузлів ---
ax2 = axes[1]
ref_curve = [newton_predict(xi, x_all, divided_differences(x_all, y_all)) for xi in x_plot]

for k in [3, 4]:
    x_k = x_all[:k]
    table_k = divided_differences(x_k, y_all[:k])
    y_k_plot = np.array([newton_predict(xi, x_k, table_k) for xi in x_plot])
    ref_arr = np.array(ref_curve)
    err = np.abs(y_k_plot - ref_arr)
    ax2.plot(x_plot, err, color=colors[k], label=f"|P₅ - P_{k}|", linewidth=1.8)

ax2.axvline(TARGET_N, color="purple", linestyle="--", linewidth=1.2, label=f"n = {TARGET_N}")
ax2.set_xlabel("Розмір вхідних даних n", fontsize=11)
ax2.set_ylabel("Абсолютна похибка (мс)", fontsize=11)
ax2.set_title("Похибка відносно P₅(n) (5 вузлів)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 7. ПОВНА ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ (5 вузлів)
# ─────────────────────────────────────────────

table_full = divided_differences(x_all, y_all)
print_divided_diff_table(x_all, table_full)

print(f"\n{'='*55}")
print(f"ПІДСУМОК: P_5({TARGET_N}) = {results[5]:.4f} мс")
print(f"{'='*55}")