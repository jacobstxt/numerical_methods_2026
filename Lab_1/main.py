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


except requests.exceptions.HTTPError as err:
    print(f"Помилка сервера: {err}")
except Exception as e:
    print(f"Сталася непередбачувана помилка: {e}")