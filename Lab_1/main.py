import requests
import numpy as np
import matplotlib.pyplot as plt


url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()

results = data["results"]
n = len(results)

# -------------------------------
# 2. Геометрія та Відстані
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

# Запис у файл
with open("route_analysis.txt", "w", encoding="utf-8") as f:
    f.write("№ | Широта | Довгота | Висота (м) | Відстань (м)\n")
    for i in range(n):
        f.write(f"{i:2d} | {coords[i][0]:.6f} | {coords[i][1]:.6f} | {elevations[i]:.2f} | {distances[i]:.2f}\n")


# -------------------------------
# 3. Кубічний сплайн (Натуральний)
# -------------------------------
def cubic_spline_natural(x, y):
    n = len(x)
    h = np.diff(x)
    A, B, C, D = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    B[0], B[-1] = 1, 1
    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Метод прогонки
    for i in range(1, n):
        m = A[i] / B[i - 1]
        B[i] -= m * C[i - 1]
        D[i] -= m * D[i - 1]

    M = np.zeros(n)
    M[-1] = D[-1] / B[-1]
    for i in range(n - 2, -1, -1):
        M[i] = (D[i] - C[i] * M[i + 1]) / B[i]

    a = y[:-1]
    b = np.zeros(n - 1)
    c = M[:-1] / 2
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        d[i] = (M[i + 1] - M[i]) / (6 * h[i])
    return a, b, c, d, x




def spline_eval(xi, a, b, c, d, x_nodes):
    idx = np.searchsorted(x_nodes, xi) - 1
    idx = np.clip(idx, 0, len(x_nodes) - 2)
    dx = xi - x_nodes[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


# Побудова еталона (всі вузли)
a_f, b_f, c_f, d_f, x_n_f = cubic_spline_natural(np.array(distances), np.array(elevations))
xx = np.linspace(distances[0], distances[-1], 1000)
yy_full = np.array([spline_eval(xi, a_f, b_f, c_f, d_f, x_n_f) for xi in xx])


# -------------------------------
# 4. Аналіз та Порівняння
# -------------------------------
def test_nodes(k):
    indices = np.linspace(0, len(distances) - 1, k, dtype=int)
    x_k, y_k = np.array(distances)[indices], np.array(elevations)[indices]
    a, b, c, d, x_n = cubic_spline_natural(x_k, y_k)
    yy_k = np.array([spline_eval(xi, a, b, c, d, x_n) for xi in xx])
    error = np.abs(yy_k - yy_full)
    print(f"\n--- {k} вузлів ---")
    print(f"Макс. похибка: {np.max(error):.4f} м")
    return yy_k, error


a_f, b_f, c_f, d_f, x_n_f = cubic_spline_natural(np.array(distances), np.array(elevations))

print("Коефіцієнти сплайна:")
for i in range(len(a_f)):
    print(f"Інтервал {i}: a={a_f[i]:.4f}, b={b_f[i]:.4f}, c={c_f[i]:.4f}, d={d_f[i]:.6f}")

yy_20, err_20 = test_nodes(20)

total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск: {total_descent:.2f} м")


yy_10, err_10 = test_nodes(10)
yy_15, err_15 = test_nodes(15)

# -------------------------------
# 5. Графіки
# -------------------------------
plt.figure(figsize=(12, 8))

# Профіль висоти
plt.subplot(2, 1, 1)
plt.plot(xx, yy_full, 'k', label="Еталон (21 вузол)", linewidth=2)
plt.plot(xx, yy_10, '--', label="10 вузлів")
plt.plot(xx, yy_15, ':', label="15 вузлів")
plt.scatter(distances, elevations, color='red', s=20, label="GPS точки")
plt.title("Порівняння апроксимації висоти")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)

# Градієнт
grad = np.gradient(yy_full, xx) * 100
plt.subplot(2, 1, 2)
plt.fill_between(xx, grad, color='orange', alpha=0.3)
plt.plot(xx, grad, color='red')
plt.title("Градієнт маршруту (Крутизна %)")
plt.ylabel("%")
plt.xlabel("Відстань (м)")
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# 6. Характеристики
# -------------------------------
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
energy_joules = 80 * 9.81 * total_ascent

print("\n===== ПІДСУМОК МАРШРУТУ =====")
print(f"Довжина маршруту: {distances[-1]:.2f} м")
print(f"Сумарний набір висоти: {total_ascent:.2f} м")
print(f"Максимальний підйом: {np.max(grad):.2f} %")
print(f"Максимальний спуск: {np.min(grad):.2f} %")
print("Середній градієнт (%):", np.mean(np.abs(grad)))
print(f"Енергія: {energy_joules / 1000:.2f} кДж ({energy_joules / 4184:.2f} ккал)")