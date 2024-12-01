import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import pandas as pd

# Уравнение второго порядка y'' - x^2 y' - (2/x^2)y = 1 + 4/x^2
def equation(x, y):
    """
    Уравнение записано в виде системы первого порядка:
    y[0] = y,
    y[1] = y'.
    """
    dydx = np.zeros((2, len(x)))
    dydx[0] = y[1]
    dydx[1] = x**2 * y[1] + (2 / x**2) * y[0] + 1 + 4 / x**2
    return dydx

# Краевые условия:
def boundary_conditions(ya, yb):
    """

    """
    return np.array([
        2 * ya[0] - ya[1] - 6,    # 2y(1/2) - y'(1/2) = 6
        yb[0] + 3 * yb[1] + 1     # y(1) + 3y'(1) = -1
    ])

# Задаём начальную сетку и начальное приближение:
x = np.linspace(0.5, 1, 100)  # сетка от x = 0.5 до x = 1
y_init = np.zeros((2, x.size))  # начальное приближение для y и y'

# Решение краевой задачи:
solution = solve_bvp(equation, boundary_conditions, x, y_init)

# Проверяем успешность решения:
if solution.success:
    print("Решение найдено успешно!")
else:
    print("Решение не найдено.")

# Вывод значений табличной функции:
x_values = solution.x
y_values = solution.y[0]
dy_values = solution.y[1]

# Формируем таблицу результатов:
table = pd.DataFrame({
    "x": x_values,
    "y(x)": y_values,
    "y'(x)": dy_values
})

# Вывод таблицы:
print("Табличные значения функции и её производной:")
print(table)

# Построение графика:
plt.figure(figsize=(12, 6))

# График y(x)
plt.plot(x_values, y_values, label="$y(x)$", color="blue", linewidth=2)

# График y'(x)
plt.plot(x_values, dy_values, label="$y'(x)$", color="red", linestyle="--", linewidth=2)

# Оформление графика:
plt.title("Решение краевой задачи")
plt.xlabel("x")
plt.ylabel("y и y'")
plt.legend()
plt.grid(True)
plt.savefig("2task.png")  # Сохраняет график в файл graph.png

table.to_csv("solution_table.csv", index=False)