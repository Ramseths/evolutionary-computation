import random
from deap import base, creator, tools, algorithms

# Datos del problema de la mochila
values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
weights = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
knapsack_capacity = 300

# Función de evaluación
def evaluate(individual):
    '''
    Evaluación
    '''
    total_weight = sum(weights[i] for i in range(len(individual)) if individual[i] == 1)
    total_value = sum(values[i] for i in range(len(individual)) if individual[i] == 1)

    if total_weight > knapsack_capacity:
        return 0,
    else:
        return total_value,

# Crear los tipos de individuos y poblaciones
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(values))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registrar las operaciones genéticas
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", evaluate)

# Configurar el algoritmo evolutivo
def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", sum)
    stats.register("std", lambda x: sum(i ** 2 for i in x) ** 0.5)
    stats.register("min", min)
    stats.register("max", max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Mejor individuo encontrado:", hof[0])
