import numpy as np
import random
import time
from copy import deepcopy


N = 5
MAGIC_NUMBER = N * (N**3 + 1) // 2
POPULATION_SIZE = 25
MAX_ITERATIONS = 10
MUTATION_RATE = 0.1


def calculate_objective(cube):
    error = 0
    for i in range(N):
        error += abs(sum(cube[i, :, :].flatten()) - MAGIC_NUMBER)
        error += abs(sum(cube[:, i, :].flatten()) - MAGIC_NUMBER)
        error += abs(sum(cube[:, :, i].flatten()) - MAGIC_NUMBER)

    diag1 = np.sum([cube[i, i, i] for i in range(N)])
    diag2 = np.sum([cube[i, i, N - i - 1] for i in range(N)])
    error += abs(diag1 - MAGIC_NUMBER) + abs(diag2 - MAGIC_NUMBER)
    return error


def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.arange(1, N**3 + 1)
        np.random.shuffle(individual)
        cube = individual.reshape((N, N, N))
        population.append(cube)
    return population


def tournament_selection(population, k=5):
    selected = random.sample(population, k)
    return min(selected, key=calculate_objective)


def crossover(parent1, parent2):
    idx1, idx2 = sorted(random.sample(range(N**3), 2))
    child1, child2 = deepcopy(parent1.flatten()), deepcopy(parent2.flatten())
    child1[idx1:idx2], child2[idx1:idx2] = parent2.flatten()[idx1:idx2], parent1.flatten()[idx1:idx2]
    # Bentuk ulang ke dalam bentuk 3D
    return child1.reshape((N, N, N)), child2.reshape((N, N, N))


def mutate(individual):
    flat = individual.flatten()
    for i in range(len(flat)):
        if random.random() < MUTATION_RATE:
            idx1, idx2 = random.sample(range(len(flat)), 2)
            flat[idx1], flat[idx2] = flat[idx2], flat[idx1]
    return flat.reshape((N, N, N))


def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')
    fitness_over_time = []
    start_time = time.time()

    for iteration in range(MAX_ITERATIONS):
        fitness_scores = [calculate_objective(ind) for ind in population]
        min_fitness = min(fitness_scores)
        fitness_over_time.append(min_fitness)

        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[fitness_scores.index(min_fitness)]

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        print(f"Iterasi {iteration}: Best Fitness = {best_fitness}")

        if best_fitness == 0:
            break
    duration = time.time() - start_time
    print("Hasil akhir:")
    print("Objective Function Akhir:", best_fitness)
    print("Durasi:", duration)
    print("State akhir dari kubus:", best_solution)
    return best_solution, fitness_over_time

best_solution, fitness_over_time = genetic_algorithm()
