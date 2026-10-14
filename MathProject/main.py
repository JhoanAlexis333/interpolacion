import numpy as np


def scale(facts: list[int]) -> float:
    """Separación promedio entre años: ancho del kernel."""
    return (max(facts) - min(facts)) / (len(facts) - 1)


def inverse_multiquadric(r: float, c: float = 1.0) -> float:
    return 1 / (1 + (r / c) ** 2) ** 0.5


def weight(facts: list[int], poblacion: list[float]) -> list[float]:
    n = len(facts)
    c = scale(facts)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            A[i][j] = A[j][i] = inverse_multiquadric(abs(facts[i] - facts[j]), c)

    return np.linalg.solve(A, np.array(poblacion, dtype=float)).tolist()


def interpolation(x: float, facts: list[int], weights: list[float]) -> float:
    n = len(facts)
    c = scale(facts)
    prediction = 0.0

    for i in range(n):
        r = abs(x - facts[i])
        imq = inverse_multiquadric(r, c)
        prediction += weights[i] * imq

    return prediction


def ajuste_exponencial(facts: list[int], poblacion: list[float]):
    """Curva de referencia: P(t) = a * e^(b * (t - t0)). Devuelve (a, b, R2, t0)."""
    t0 = facts[0]
    t = np.array(facts, dtype=float) - t0
    p = np.array(poblacion, dtype=float)
    b, ln_a = np.polyfit(t, np.log(p), 1)
    a = np.exp(ln_a)
    pred = a * np.exp(b * t)
    ss_res = np.sum((p - pred) ** 2)
    ss_tot = np.sum((p - p.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2, t0
