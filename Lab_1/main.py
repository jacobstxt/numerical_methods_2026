import requests
import matplotlib.pyplot as plt
import numpy as np

# Список координат у форматі об'єктів (так API працює стабільніше)
locations = [
    {"latitude": 48.164214, "longitude": 24.536044},
    {"latitude": 48.164983, "longitude": 24.534836},
    {"latitude": 48.165605, "longitude": 24.534068},
    {"latitude": 48.166228, "longitude": 24.532915},
    {"latitude": 48.166777, "longitude": 24.531927},
    {"latitude": 48.167326, "longitude": 24.530884},
    {"latitude": 48.167011, "longitude": 24.530061},
    {"latitude": 48.166053, "longitude": 24.528039},
    {"latitude": 48.166655, "longitude": 24.526064},
    {"latitude": 48.166497, "longitude": 24.523574},
    {"latitude": 48.166128, "longitude": 24.520214},
    {"latitude": 48.165416, "longitude": 24.517170},
    {"latitude": 48.164546, "longitude": 24.514640},
    {"latitude": 48.163412, "longitude": 24.512980},
    {"latitude": 48.162331, "longitude": 24.511715},
    {"latitude": 48.162015, "longitude": 24.509462},
    {"latitude": 48.162147, "longitude": 24.506932},
    {"latitude": 48.161751, "longitude": 24.504244},
    {"latitude": 48.161197, "longitude": 24.501793},
    {"latitude": 48.160580, "longitude": 24.500537},
    {"latitude": 48.160250, "longitude": 24.500106}
]

url = "https://api.open-elevation.com/api/v1/lookup"
header = {'Accept': 'application/json', 'Content-Type': 'application/json'}

try:
    # Надсилаємо POST запит замість GET
    response = requests.post(url, headers=header, json={"locations": locations})

    # Перевіряємо, чи повернув сервер успішний статус (200)
    response.raise_for_status()

    data = response.json()
    elevations = [item['elevation'] for item in data['results']]

    results = data["results"]


    #Виведення даних
    n = len(results)
    print("Кількість вузлів:", n)
    print("\nТабуляція вузлів:")
    print("-" * 45)
    print(" № | Latitude  | Longitude | Elevation (m)")
    print("-" * 45)


    for i, point in enumerate(results):
     print(f"{i:2d} | {point['latitude']:.6f} | "
           f"{point['longitude']:.6f} | "
           f"{point['elevation']}")

    print("-" * 45)

    import numpy as np


    def haversine(lat1, lon1, lat2, lon2):
     R = 6371000  # Радіус Землі в метрах
     phi1, phi2 = np.radians(lat1), np.radians(lat2)
     dphi = np.radians(lat2 - lat1)
     dlambda = np.radians(lon2 - lon1)

     a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
     return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


    # Підготовка даних
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = [p["elevation"] for p in results]
    distances = [0]

    for i in range(1, n):
     d = haversine(*coords[i - 1], *coords[i])
     distances.append(distances[-1] + d)

    print("\nТабуляція (відстань, висота):")
    print(" № | Distance (m) | Elevation (m)")
    print("-" * 35)
    for i in range(n):
     print(f"{i:2d} | {distances[i]:12.2f} | {elevations[i]:13.2f}")

    # Оновлений графік
    plt.figure(figsize=(10, 5))
    plt.plot(distances, elevations, marker='o', color='blue', linewidth=2)
    plt.fill_between(distances, elevations, min(elevations) - 10, alpha=0.2, color='green')
    plt.title("Фізичний профіль рельєфу")
    plt.xlabel("Відстань від старту (метри)")
    plt.ylabel("Висота над рівнем моря (метри)")
    plt.grid(True, linestyle='--')
    plt.show()


    # --- 6. Побудова кубічних сплайнів ---
    def build_spline(x, y):
        n = len(x)
        h = np.diff(x)

        # 6. Формування системи лінійних рівнянь (тридіагональна матриця)
        # А * m = B, де m - другі похідні (коефіцієнти c)
        A = np.zeros((n, n))
        B = np.zeros(n)

        # Природні умови (другі похідні на кінцях = 0)
        A[0, 0] = 1
        A[n - 1, n - 1] = 1

        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        # 7. Метод прогонки (Алгоритм Томаса)
        def tridiagonal_solve(A, B):
            n = len(B)
            c_prime = np.zeros(n)
            d_prime = np.zeros(n)
            sol = np.zeros(n)

            # Прямий хід
            c_prime[0] = A[0, 1] / A[0, 0] if n > 1 else 0
            d_prime[0] = B[0] / A[0, 0]

            for i in range(1, n - 1):
                m = A[i, i] - A[i, i - 1] * c_prime[i - 1]
                c_prime[i] = A[i, i + 1] / m
                d_prime[i] = (B[i] - A[i, i - 1] * d_prime[i - 1]) / m

            d_prime[n - 1] = (B[n - 1] - A[n - 1, n - 2] * d_prime[n - 2]) / (
                        A[n - 1, n - 1] - A[n - 1, n - 2] * c_prime[n - 2])

            # Зворотний хід
            sol[n - 1] = d_prime[n - 1]
            for i in range(n - 2, -1, -1):
                sol[i] = d_prime[i] - c_prime[i] * sol[i + 1]
            return sol

        print("\n--- Розв'язок системи (коефіцієнти c_i) ---")
        c = tridiagonal_solve(A, B)
        print(c)

        # 8. Обчислення коефіцієнтів a, b, d
        a = np.array(y[:-1])
        b = np.zeros(n - 1)
        d = np.zeros(n - 1)

        for i in range(n - 1):
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        return a, b, c[:-1], d


    # Функція для отримання значення сплайна в точці x_new
    def evaluate_spline(x, x_nodes, a, b, c, d):
        idx = np.searchsorted(x_nodes, x) - 1
        idx = np.clip(idx, 0, len(x_nodes) - 2)
        dx = x - x_nodes[idx]
        return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


    # --- 10. Побудова графіків для 10, 15, 20 вузлів ---
    plt.figure(figsize=(12, 7))
    colors = {10: 'red', 15: 'orange', 20: 'blue'}

    for count in [10, 15, 20]:
        # Вибираємо підмножину вузлів
        indices = np.linspace(0, len(distances) - 1, count, dtype=int)
        x_nodes = np.array(distances)[indices]
        y_nodes = np.array(elevations)[indices]

        # Будуємо сплайн
        a, b, c, d_coeff = build_spline(x_nodes, y_nodes)

        # Для гладкого графіка створюємо 500 точок
        x_fine = np.linspace(min(distances), max(distances), 500)
        y_fine = [evaluate_spline(val, x_nodes, a, b, c, d_coeff) for val in x_fine]

        plt.plot(x_fine, y_fine, label=f'Сплайн ({count} вузлів)', color=colors[count])
        plt.scatter(x_nodes, y_nodes, color=colors[count], s=30)

    plt.plot(distances, elevations, 'k--', alpha=0.3, label='Оригінальні дані')
    plt.title("Порівняння кубічних сплайнів при різній кількості вузлів")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Висота (м)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Додатково: Характеристики маршруту ---
    total_len = distances[-1]
    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))

    print(f"\n--- Характеристики маршруту ---")
    print(f"Загальна довжина: {total_len:.2f} м")
    print(f"Сумарний набір висоти: {total_ascent:.2f} м")
    print(f"Сумарний спуск: {total_descent:.2f} м")

    # Енергія (маса 80 кг)
    energy = 80 * 9.81 * total_ascent
    print(f"Механічна енергія підйому (кДж): {energy / 1000:.2f}")


except requests.exceptions.HTTPError as err:
    print(f"Помилка сервера: {err}")
except Exception as e:
    print(f"Сталася непередбачувана помилка: {e}")