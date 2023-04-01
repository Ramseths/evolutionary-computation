import random
import numpy as np
from deap import base, creator, tools, algorithms

# Configuración del problema XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función para evaluar la red neuronal
def evaluate(individual):
    num_inputs = 2
    num_hidden = 5
    num_outputs = 1

    # Extraer pesos y bias de la red neuronal
    weights_input_hidden = np.array(individual[:num_hidden * num_inputs]).reshape((num_hidden, num_inputs))
    bias_hidden = np.array(individual[num_hidden * num_inputs:num_hidden * num_inputs + num_hidden])
    weights_hidden_output = np.array(individual[-(num_outputs * num_hidden):]).reshape((num_outputs, num_hidden))
    
    error = 0
    for i, input_ in enumerate(inputs):
        hidden = sigmoid(np.dot(input_, weights_input_hidden.T) + bias_hidden)
        output = sigmoid(np.dot(hidden, weights_hidden_output.T))
        error += abs(outputs[i] - output)

    return error,

# Crear los tipos de individuos y poblaciones
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=22)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registrar las operaciones genéticas
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", evaluate)

# Configurar el algoritmo evolutivo
def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Mejor individuo encontrado! :", hof[0])
