import numpy as np
import matplotlib.pyplot as plt


# Коефіцієнти многочлена (від старшого до молодшого)
COEFFS = [1.0, -3.0, 4.0, -2.0]


def P(x):
    """P(x) = x^3 - 3x^2 + 4x - 2"""
    return x**3 - 3*x**2 + 4*x - 2


def dP(x):
    """P'(x) = 3x^2 - 6x + 4"""
    return 3*x**2 - 6*x + 4


# Будуємо графік
xs = np.linspace(-1, 3, 500)
ys = P(xs)

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# Сітка і осі
ax.grid(True, alpha=0.3, linestyle='--')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)

# Сам графік
ax.plot(xs, ys, 'b-', linewidth=2.5, label='P(x) = x³ − 3x² + 4x − 2')

# Дійсний корінь x = 1
ax.plot(1, 0, 'ro', markersize=12, zorder=5)
ax.annotate('Дійсний корінь\nx = 1',
            xy=(1, 0), xytext=(1.5, -3),
            fontsize=12, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Точка перегину (де P'(x) = 0)
# Похідна P'(x) = 3x² - 6x + 4. Дискримінант = 36 - 48 = -12 < 0
# Значить P'(x) > 0 завжди → функція монотонно зростає
# Точка мінімуму швидкості зростання: x = 1 (вершина параболи P')
ax.plot(1, P(1), 'gs', markersize=10, zorder=5)

# Інформаційна табличка про комплексні корені
info_text = ('Комплексні корені:\n'
             'x₂ = 1 + i\n'
             'x₃ = 1 − i\n\n'
             '(на графіку не видно —\n'
             'вони в комплексній\n'
             'площині)')
ax.text(0.02, 0.97, info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6',
                  facecolor='lightyellow',
                  edgecolor='gray',
                  alpha=0.9))

# Підписи осей і заголовок
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('P(x)', fontsize=13)
ax.set_title('Графік многочлена P(x) = x³ − 3x² + 4x − 2',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)

# Межі
ax.set_xlim(-1, 3)
ax.set_ylim(-12, 12)

plt.tight_layout()
plt.savefig('poly_graph.png', dpi=120, bbox_inches='tight')
print("Графік збережено: poly_graph.png")

# Показуємо вікно з графіком (якщо запускаєш в PyCharm — відкриється)
plt.show()