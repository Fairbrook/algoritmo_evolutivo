import matplotlib.pyplot as plt
from genetics import Genetics
import numpy as np
import functools

# Configuraciones
m = 10
population_len = 100
mutation_rate = 0.1
seed = 0.5284292289940231
domain = (0, np.pi)

# Ecuación general
def F(x, i) -> float:
    return -np.sin(x)*np.power(np.sin((i*x**2)/np.pi), 2*m)

# Ecuación a dos dimensiones
def f(x, y): return F(x, 1)+F(y, 2)

# Punto a buscar
# print(f(2.20,1.57))


# Creación de la "genetica"
genetic = Genetics(domain=domain, population_len=population_len,
                   mutation_rate=mutation_rate, evaluation_cb=f, minima=True, seed=seed)
# Creación de la población inicial
genetic.gen_initial_population()
print("Poblacion inicial")
print(Genetics.format_evaluations(genetic.evaluate_population()))

# Tolerancia ante repetición (Cuantas veces se repite el mejor valor antes de parar)
tolerancia = 5

# Ciclo de generaciones
prevTop = None
i = 0
tops = []
averages = []
indexes = []
while tolerancia > 0:
    print(f"Poblacion {i}")
    i += 1
    indexes.append(i)

    # Creación y evaluación de la siguiente genración
    genetic.next_generation()
    evaluations = genetic.evaluate_population()
    print(Genetics.format_evaluations(evaluations))
    averages.append(functools.reduce(lambda res, x: res +
                    x[1], evaluations, 0)/len(evaluations))

    # Lógica de tolerancia
    top = evaluations[-1]
    tops.append(top[1])
    if prevTop == top:
        tolerancia -= 1
    else:
        tolerancia = 5
    prevTop = top

best = (genetic.population[-1][0], genetic.population[-1][1], tops[-1])

# Graficación del proceso
resolution = 150
fig = plt.figure(figsize=plt.figaspect(0.4))
fig.suptitle("Busqueda de solución mediante Algoritmos Evolutivos")
fig.tight_layout(pad=10)

# Evolucion
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Evolución")
ax.plot(indexes, tops, label="Mejores")
ax.plot(indexes, averages, label="Promedios")
ax.set_xlabel("Generación")
ax.set_ylabel("Evaluación")
ax.legend()

#Grafica de la función
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title("Función de Michalewicz")
x = np.linspace(0, np.pi, resolution)
y = np.linspace(0, np.pi, resolution)
X, Y = np.meshgrid(x, y)
Z = F(X, 1)+F(Y, 2)
ax.contourf(X, Y, Z, resolution)
print(best)
ax.scatter(best[0], best[1], best[2], label="Mejor punto encontrado")
ax.legend()

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.85,
                    wspace=0.4,
                    hspace=0.4)
plt.figtext(0.5, 0.01, f"Mejor punto encontrado x={best[0]:.3f} y={best[1]:.3f} z={best[2]:.3f}", ha="center",
            fontsize=10,)
plt.show()
