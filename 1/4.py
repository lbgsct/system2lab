import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
m = 1  # масса груза
k = 1  # жёсткость пружины
v0 = 1  # начальная скорость
x0 = 0  # начальное положение
b = 1   # амплитуда силы
omega = 2  # частота внешней силы

# Рассмотрим два случая
h_underdamped = 0.5  # h^2 < 4km
h_overdamped = 3.0   # h^2 > 4km

# Временной интервал
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Уравнение системы
def system(t, y, h):
    x, v = y
    dxdt = v
    dvdt = (b * np.sin(omega * t) - h * v - k * x) / m  # Добавлена сила f(t) = b * sin(omega * t)
    return [dxdt, dvdt]

# Аналитическое решение для недокритического случая
def analytic_underdamped(t, h, m, k, b, omega):
    gamma = h / (2 * m)
    omega_0 = np.sqrt(k / m - gamma**2)
    C1 = x0
    C2 = (v0 + gamma * x0) / omega_0
    x_homogeneous = np.exp(-gamma * t) * (C1 * np.cos(omega_0 * t) + C2 * np.sin(omega_0 * t))
    A = b * (k - m * omega**2) / ((k - m * omega**2)**2 + (omega * h)**2)
    B = b * omega * h / ((k - m * omega**2)**2 + (omega * h)**2)
    x_particular = A * np.sin(omega * t) + B * np.cos(omega * t)
    return x_homogeneous + x_particular

# Аналитическое решение для перекритического случая
def analytic_overdamped(t, h, m, k, b, omega):
    gamma = h / (2 * m)
    r1 = -gamma + np.sqrt(gamma**2 - k / m)
    r2 = -gamma - np.sqrt(gamma**2 - k / m)
    C1 = (v0 - r2 * x0) / (r1 - r2)
    C2 = x0 - C1
    x_homogeneous = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
    A = b * (k - m * omega**2) / ((k - m * omega**2)**2 + (omega * h)**2)
    B = b * omega * h / ((k - m * omega**2)**2 + (omega * h)**2)
    x_particular = A * np.sin(omega * t) + B * np.cos(omega * t)
    return x_homogeneous + x_particular

# Численное решение
def solve_numerical(h):
    sol = solve_ivp(system, t_span, [x0, v0], t_eval=t_eval, args=(h,))
    return sol.t, sol.y[0]

# Графики
plt.figure(figsize=(12, 8))

# (h^2 < 4km)
t_approx, x_approx = solve_numerical(h_underdamped)
x_analytic = analytic_underdamped(t_eval, h_underdamped, m, k, b, omega)
plt.subplot(2, 1, 1)
plt.plot(t_approx, x_approx, label="Численное решение", linewidth=1.5)
plt.plot(t_eval, x_analytic, label="Аналитическое решение", linestyle="--", linewidth=1.5)
plt.title("h^2 < 4km, f(t) = b * sin(omega * t)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)

# (h^2 > 4km)
t_approx, x_approx = solve_numerical(h_overdamped)
x_analytic = analytic_overdamped(t_eval, h_overdamped, m, k, b, omega)
plt.subplot(2, 1, 2)
plt.plot(t_approx, x_approx, label="Численное решение", linewidth=1.5)
plt.plot(t_eval, x_analytic, label="Аналитическое решение", linestyle="--", linewidth=1.5)
plt.title("h^2 > 4km, f(t) = b * sin(omega * t)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("graph4.png")  # Сохраняет график в файл graph.png