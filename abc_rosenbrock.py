import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

# Define the objective function
def objective_function(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Parámetros
n_bees = 50
n_dim = 2
n_iter = 1000
limit = 50
bounds = [(-3, 3), (-3, 3)]


# Algoritmo Colonia de Abejas Artificiales
def abc_algorithm(objective_function, n_bees, n_dim, n_iter, bounds, limit):

    def generate_bees(n_bees, n_dim, bounds):
        bees = np.zeros((n_bees, n_dim))
        for i in range(n_bees):
            for j in range(n_dim):
                bees[i, j] = np.random.uniform(bounds[j][0], bounds[j][1])
        return bees

    bees = generate_bees(n_bees, n_dim, bounds)
    fitness = np.array([objective_function(x) for x in bees])
    trial = np.zeros(n_bees)
    best_bee = bees[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for i_iter in range(n_iter):

        # Fase de abejas
        for i in range(n_bees):
            j = np.random.choice(n_dim)
            phi = np.random.uniform(-1, 1)
            k = np.random.choice([k for k in range(n_bees) if k != i])
            v = bees[i].copy()
            v[j] = bees[i, j] + phi * (bees[i, j] - bees[k, j])
            v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            v_fitness = objective_function(v)

            if v_fitness < fitness[i]:
                bees[i] = v
                fitness[i] = v_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Fase de abeja observadora
        prob = (np.max(fitness) - fitness) / np.sum(np.max(fitness) - fitness)
        for i in range(n_bees):
            if np.random.uniform() < prob[i]:
                j = np.random.choice(n_dim)
                phi = np.random.uniform(-1, 1)
                k = np.random.choice([k for k in range(n_bees) if k != i])
                v = bees[i].copy()
                v[j] = bees[i, j] + phi * (bees[i, j] - bees[k, j])
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
                v_fitness = objective_function(v)

                if v_fitness < fitness[i]:
                    bees[i] = v
                    fitness[i] = v_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

        # Abeja exploradora
        for i in range(n_bees):
            if trial[i] >= limit:
                bees[i] = generate_bees(1, n_dim, bounds)[0]
                fitness[i] = objective_function(bees[i])
                trial[i] = 0

                # Mejor solución
        if np.min(fitness) < best_fitness:
            best_bee = bees[np.argmin(fitness)]
            best_fitness = np.min(fitness)

        yield best_bee, best_fitness, bees

# Set up the plot
fig, ax = plt.subplots()
x = np.linspace(bounds[0][0], bounds[0][1], 100)
y = np.linspace(bounds[1][0], bounds[1][1], 100)
X, Y = np.meshgrid(x, y)
Z = objective_function([X, Y])

# Plot the function
cp = ax.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap="bone")
plt.title('Rosenbrock Function')
plt.colorbar(cp, ax=ax)

# Initialize the plot with the first frame
best_bee, best_fitness, bees = next(abc_algorithm(objective_function, n_bees, n_dim, n_iter, bounds, limit))
bees_scatter = ax.scatter(bees[:, 0], bees[:, 1], c='red', marker='o', s=50, label='Bees')
best_bee_scatter = ax.scatter(best_bee[0], best_bee[1], c='yellow', marker='*', s=200, label='Best bee')
ax.legend(loc='upper right')

# Animation update function
def update(frame):
    best_bee, best_fitness, bees = frame
    bees_scatter.set_offsets(bees)
    best_bee_scatter.set_offsets(best_bee)
    return bees_scatter, best_bee_scatter

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=abc_algorithm(objective_function, n_bees, n_dim, n_iter, bounds, limit), interval=100, blit=True)
print(best_fitness)
# Display the animation
plt.show()

