import tkinter as tk
from tkinter import messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main import scale, weight, interpolation, ajuste_exponencial


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crecimiento de la población (RBF multicuadrática inversa)")
        self.geometry("980x620")

        panel = tk.Frame(self, padx=10, pady=10)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(panel, text="Años (separados por coma)").pack(anchor="w")
        self.e_years = tk.Entry(panel, width=32)
        self.e_years.insert(0, "2005, 2010, 2015")
        self.e_years.pack(anchor="w", pady=(0, 8))

        tk.Label(panel, text="Población de cada año").pack(anchor="w")
        self.e_pop = tk.Entry(panel, width=32)
        self.e_pop.insert(0, "1000, 1200, 1500")
        self.e_pop.pack(anchor="w", pady=(0, 8))

        tk.Button(panel, text="Graficar", command=self.graficar).pack(
            anchor="w", pady=(0, 14)
        )

        tk.Label(panel, text="Año a predecir").pack(anchor="w")
        self.e_x = tk.Entry(panel, width=12)
        self.e_x.insert(0, "2012")
        self.e_x.pack(anchor="w", pady=(0, 8))

        tk.Button(panel, text="Predecir", command=self.predecir_anio).pack(
            anchor="w", pady=(0, 8)
        )
        self.lbl_res = tk.Label(panel, text="", justify="left", wraplength=260)
        self.lbl_res.pack(anchor="w")

        self.fig = Figure(figsize=(6.5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.facts = []
        self.poblacion = []
        self.weights = []
        self.graficar()

    # -- lectura de datos
    def leer_datos(self):
        try:
            facts = [int(v) for v in self.e_years.get().split(",")]
            pob = [float(v) for v in self.e_pop.get().split(",")]
        except ValueError:
            raise ValueError("Escribe solo números separados por coma.")
        if len(facts) != len(pob):
            raise ValueError("Debe haber la misma cantidad de años y poblaciones.")
        if len(facts) < 2:
            raise ValueError("Se necesitan al menos 2 datos.")
        if len(set(facts)) != len(facts):
            raise ValueError("Los años no pueden repetirse.")
        orden = sorted(range(len(facts)), key=lambda i: facts[i])
        return [facts[i] for i in orden], [pob[i] for i in orden]

    # -- gráfica
    def graficar(self):
        try:
            self.facts, self.poblacion = self.leer_datos()
        except ValueError as e:
            messagebox.showerror("Datos inválidos", str(e))
            return False

        self.weights = weight(self.facts, self.poblacion)

        xs = np.linspace(self.facts[0], self.facts[-1], 300)
        ys = [interpolation(x, self.facts, self.weights) for x in xs]

        self.ax.clear()
        self.ax.plot(xs, ys, color="tab:blue", lw=2, label="RBF (interpolación)")
        self.ax.scatter(self.facts, self.poblacion, color="tab:red", zorder=3,
                        label="Datos conocidos")

        if all(p > 0 for p in self.poblacion):
            a, b, r2, t0 = ajuste_exponencial(self.facts, self.poblacion)
            ye = a * np.exp(b * (xs - t0))
            self.ax.plot(xs, ye, "--", color="tab:green", lw=1.5,
                         label=f"Exponencial de referencia (R²={r2:.3f})")

        self.ax.set_title("Crecimiento de la población")
        self.ax.set_xlabel("Año")
        self.ax.set_ylabel("Población")
        self.ax.grid(alpha=0.3)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()
        self.lbl_res.config(
            text=f"Función: f(x) = Σ Bᵢ · 1/√(1+(|x−xᵢ|/c)²),  c={scale(self.facts):g}\n"
                 f"B = {[round(b, 3) for b in self.weights]}"
        )
        return True

    # -- predicción
    def predecir_anio(self):
        # Releer los datos de las cajas y recalcular los pesos ANTES de predecir
        if not self.graficar():
            return
        try:
            x = float(self.e_x.get())
        except ValueError:
            messagebox.showerror("Dato inválido", "El año debe ser un número.")
            return
        if not (self.facts[0] <= x <= self.facts[-1]):
            messagebox.showwarning(
                "Fuera de rango",
                f"El año debe estar entre {self.facts[0]} y {self.facts[-1]}.",
            )
            return

        y = interpolation(x, self.facts, self.weights)
        self.graficar_punto(x, y)
        self.lbl_res.config(text=f"Población estimada en {x:g}: {y:,.2f}")

    def graficar_punto(self, x, y):
        self.ax.scatter([x], [y], color="black", marker="*", s=180, zorder=4,
                        label="Predicción")
        self.ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                         xytext=(8, 8))
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()
