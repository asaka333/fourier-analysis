from typing import Tuple
import numpy as np


def periodic_ex1(t):
    t = t % 2
    if 0 <= t < 1:
        return 1
    else:
        return 0


def periodic_ex2(t):
    return (t + 1) % 2 - 1


def non_periodic_ex1(t):
    if 0 < t <= 1:
        return 1
    else:
        return 0


def non_periodic_ft_ex1(k):
    return np.where(
        k == 0,
        np.nan,
        (1 - np.exp(-1j * k)) / (np.sqrt(2 * np.pi) * 1j * k),
    )


def inv_fourier_transform(F, t, N, num_points_per_one=10):
    k = np.linspace(-N, N, num_points_per_one * N)
    delta_k = 2 * N / num_points_per_one

    return (1 / np.sqrt(2 * np.pi)) * np.sum(F(k) * np.exp(1j * k * t) * delta_k)


def plot_periodic_ex1(ax, t=np.linspace(-4, 8, 1000)):
    return plot_function(ax, periodic_ex1, t)


def plot_periodic_ex2(ax, t=np.linspace(-4, 8, 1000)):
    return plot_function(ax, periodic_ex2, t)


def plot_non_periodic_ex1(ax, t):
    return plot_function(ax, non_periodic_ex1, t)


def plot_non_periodic_ft_ex1(
    ax,
    t,
    range_color_red: Tuple[int, int] | None = None,
):
    return plot_function(ax, non_periodic_ft_ex1, t, range_color_red, False)


def plot_non_periodic_invft_ex1(ax, N, t):

    return plot_function(
        ax,
        lambda k: inv_fourier_transform(non_periodic_ft_ex1, k, N),
        t,
        is_time_coordinate=False,
    )


def plot_function(
    ax,
    periodic_func,
    t,
    range_color_red: Tuple[int, int] | None = None,
    is_time_coordinate=True,
):
    vectorized_f = np.vectorize(periodic_func)
    y = vectorized_f(t)

    ax.plot(t, y, drawstyle="steps-post")
    if range_color_red is not None:
        mask = (t >= range_color_red[0]) & (t <= range_color_red[1])
        ax.plot(t[mask], y[mask], drawstyle="steps-post", color="red")
    if is_time_coordinate:
        ax.set_ylabel("f(t)")
        ax.set_title("$f(t)$")
        ax.set_xlabel("$t$")
    else:
        ax.set_ylabel("F(k)")
        ax.set_title("$F(k)$")
        ax.set_xlabel("$k$")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.1)
    return ax


def Cn_ex1(n):
    if n == 0:
        return 1 / 2
    else:
        return (1j / (2 * n * np.pi)) * ((-1.0) ** n - 1)


def Cn_ex2(n):
    if n == 0:
        return 0
    else:
        return (1j / (n * np.pi)) * (-1.0) ** n


def plot_Cn_ex1(ax, n=-1, plot_range=20):
    return plot_Cn(ax, Cn_ex1, n, plot_range)


def plot_Cn_ex2(ax, n=-1, plot_range=20):
    return plot_Cn(ax, Cn_ex2, n, plot_range)


def plot_Cn(ax, Cn_func, n, plot_range):
    if plot_range <= 15:
        n_values = np.arange(-20, 20)
    else:
        n_values = np.arange(-plot_range - 5, plot_range + 5)
    Cn_values = np.array([Cn_func(n) for n in n_values])
    Cn_values_abs = np.abs(Cn_values)

    if n < 0:
        bar_colors = ["green" for _ in n_values]
    else:
        bar_colors = ["red" if -n <= k <= n else "green" for k in n_values]

    ax.bar(n_values, Cn_values_abs, color=bar_colors, width=0.8, label="$|C_n|$")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("$n$")
    ax.set_ylabel("$|C_n|$")
    ax.set_title("$|C_n|$")
    ax.grid(alpha=0.5)


def periodic_reconst(t, n, Cn):
    out = Cn(0) * np.exp(1j * 0 * np.pi * t)
    for i in range(n)[1:]:
        out += Cn(i) * np.exp(1j * i * np.pi * t)
        out += Cn(-i) * np.exp(1j * (-i) * np.pi * t)
    return out


def plot_periodic_reconst_ex1(ax, n):
    return plot_periodic_reconst(ax, n, Cn_ex1)


def plot_periodic_reconst_ex2(ax, n):
    return plot_periodic_reconst(ax, n, Cn_ex2)


def plot_periodic_reconst(ax, n, Cn):
    vectorized_periodic_reconst = np.vectorize(periodic_reconst)
    t = np.linspace(-4, 8, 1000)
    y_values = vectorized_periodic_reconst(t, n, Cn)
    y_values_real = np.real(y_values)

    ax.plot(t, y_values_real, drawstyle="steps-post")
    ax.set_xlabel("$t$")
    ax.set_ylabel("f(t)")
    ax.set_title(f"$f(t)$ $(N={n})$")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.1)
    return ax
