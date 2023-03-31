import numpy as np

def initialize_population(pop_size, genome_length):
    return np.random.randint(0, 2, (pop_size, genome_length))

def calculate_fitness(genome):
    return np.sum(genome)

def selection(population, fitness):
    parents = []
    for _ in range(len(population)):
        candidates = population[np.random.choice(len(population), size=2, replace=False)]
        selected = candidates[np.argmax(fitness[candidates])]
        parents.append(selected)
    return np.array(parents)

def crossover(parents):
    offspring = np.empty_like(parents)
    crossover_point = len(parents[0]) // 2
    for i in range(len(parents) // 2):
        parent1, parent2 = parents[i * 2], parents[i * 2 + 1]
        offspring[i * 2] = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
        offspring[i * 2 + 1] = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring

def mutation(offspring, mutation_rate):
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] = 1 - offspring[i, j]
    return offspring

def genetic_algorithm(pop_size, genome_length, generations, mutation_rate):
    population = initialize_population(pop_size, genome_length)
    for generation in range(generations):
        fitness = np.array([calculate_fitness(genome) for genome in population])
        parents = selection(population, fitness)
        offspring = crossover(parents)
        mutated_offspring = mutation(offspring, mutation_rate)
        population = mutated_offspring

        print(f"Generation {generation+1}, best fitness: {np.max(fitness)}")

    best_genome = population[np.argmax(fitness)]
    return best_genome

if __name__ == "__main__":
    pop_size = 50
    genome_length = 20
    generations = 100
    mutation_rate = 0.01

    best_genome = genetic_algorithm(pop_size, genome_length, generations, mutation_rate)
    print(f"Best genome: {best_genome}, fitness: {calculate_fitness(best_genome)}")
