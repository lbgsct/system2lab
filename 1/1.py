import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
m = 1  # масса груза
k = 1  # жёсткость пружины
v0 = 1  # начальная скорость
x0 = 0  # начальное положение

# Рассмотрим два случая
h_underdamped = 0.5  # h^2 < 4km
h_overdamped = 3.0   # h^2 > 4km

# Временной интервал
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Уравнение системы
def system(t, y, h, f_func):
    x, v = y
    dxdt = v
    dvdt = (-h * v - k * x + f_func(t)) / m
    return [dxdt, dvdt]

# Функция внешней силы (в данном случае f(t) = 0)
def external_force(t):
    return 0

# Аналитическое решение для недокритического случая
def analytic_underdamped(t, h, m, k):
    gamma = h / (2 * m)
    omega = np.sqrt(k / m - gamma**2)
    C1 = x0
    C2 = (v0 + gamma * x0) / omega
    return np.exp(-gamma * t) * (C1 * np.cos(omega * t) + C2 * np.sin(omega * t))

# Аналитическое решение для перекритического случая
def analytic_overdamped(t, h, m, k):
    gamma = h / (2 * m)
    r1 = -gamma + np.sqrt(gamma**2 - k / m)
    r2 = -gamma - np.sqrt(gamma**2 - k / m)
    C1 = (v0 - r2 * x0) / (r1 - r2)
    C2 = x0 - C1
    return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

# Численное решение
def solve_numerical(h):
    sol = solve_ivp(system, t_span, [x0, v0], t_eval=t_eval, args=(h, external_force))
    return sol.t, sol.y[0]

# Графики
plt.figure(figsize=(12, 8))

# Недокритический случай (h^2 < 4km)
t_approx, x_approx = solve_numerical(h_underdamped)
x_analytic = analytic_underdamped(t_eval, h_underdamped, m, k)
plt.subplot(2, 1, 1)
plt.plot(t_approx, x_approx, label="Численное решение", linewidth=1.5)
plt.plot(t_eval, x_analytic, label="Аналитическое решение", linestyle="--", linewidth=1.5)
plt.title("Недокритическое затухание (h^2 < 4km)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)

# Перекритический случай (h^2 > 4km)
t_approx, x_approx = solve_numerical(h_overdamped)
x_analytic = analytic_overdamped(t_eval, h_overdamped, m, k)
plt.subplot(2, 1, 2)
plt.plot(t_approx, x_approx, label="Численное решение", linewidth=1.5)
plt.plot(t_eval, x_analytic, label="Аналитическое решение", linestyle="--", linewidth=1.5)
plt.title("Перекритическое затухание (h^2 > 4km)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("graph1.png")  # Сохраняет график в файл graph.png