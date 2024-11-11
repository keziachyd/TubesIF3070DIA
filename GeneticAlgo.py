import numpy as np
import random
from copy import deepcopy

# Konfigurasi masalah
N = 5
MAGIC_NUMBER = N * (N**3 + 1) // 2
POPULATION_SIZE = 50
MAX_ITERATIONS = 500
MUTATION_RATE = 0.1

# Fungsi untuk menghitung objective function
def calculate_objective(cube):
    # Menghitung total perbedaan antara jumlah di tiap arah dengan MAGIC_NUMBER
    error = 0
    for i in range(N):
        error += abs(sum(cube[i, :, :].flatten()) - MAGIC_NUMBER)
        error += abs(sum(cube[:, i, :].flatten()) - MAGIC_NUMBER)
        error += abs(sum(cube[:, :, i].flatten()) - MAGIC_NUMBER)
    # Tambahkan perhitungan untuk diagonal ruang dan bidang
    diag1 = np.sum([cube[i, i, i] for i in range(N)])
    diag2 = np.sum([cube[i, i, N - i - 1] for i in range(N)])
    error += abs(diag1 - MAGIC_NUMBER) + abs(diag2 - MAGIC_NUMBER)
    return error

# Fungsi untuk menginisialisasi populasi awal
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.arange(1, N**3 + 1)
        np.random.shuffle(individual)
        cube = individual.reshape((N, N, N))
        population.append(cube)
    return population

# Fungsi seleksi: memilih dua orang tua dengan metode turnamen
def tournament_selection(population, k=5):
    selected = random.sample(population, k)
    return min(selected, key=calculate_objective)

# Fungsi crossover: menggunakan Partially Matched Crossover (PMX)
def crossover(parent1, parent2):
    # Memilih dua titik untuk crossover
    idx1, idx2 = sorted(random.sample(range(N**3), 2))
    child1, child2 = deepcopy(parent1.flatten()), deepcopy(parent2.flatten())
    child1[idx1:idx2], child2[idx1:idx2] = parent2.flatten()[idx1:idx2], parent1.flatten()[idx1:idx2]
    # Bentuk ulang ke dalam bentuk 3D
    return child1.reshape((N, N, N)), child2.reshape((N, N, N))

# Fungsi mutasi: menukar dua angka acak
def mutate(individual):
    flat = individual.flatten()
    for i in range(len(flat)):
        if random.random() < MUTATION_RATE:
            idx1, idx2 = random.sample(range(len(flat)), 2)
            flat[idx1], flat[idx2] = flat[idx2], flat[idx1]
    return flat.reshape((N, N, N))

# Fungsi utama untuk Genetic Algorithm
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')
    fitness_over_time = []

    for iteration in range(MAX_ITERATIONS):
        # Evaluasi fitness untuk setiap individu
        fitness_scores = [calculate_objective(ind) for ind in population]
        min_fitness = min(fitness_scores)
        fitness_over_time.append(min_fitness)

        # Menyimpan solusi terbaik
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[fitness_scores.index(min_fitness)]

        # Buat generasi baru
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Seleksi dua orang tua
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            # Crossover dan mutasi
            child1, child2 = crossover(parent1, parent2)
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])

        # Update populasi
        population = new_population[:POPULATION_SIZE]

        # Menampilkan hasil per iterasi (opsional)
        print(f"Iterasi {iteration}: Best Fitness = {best_fitness}")

        # Hentikan jika mencapai solusi optimal
        if best_fitness == 0:
            break

    # Hasil akhir
    print("Hasil akhir:")
    print("Objective Function Akhir:", best_fitness)
    print("State akhir dari kubus:", best_solution)
    return best_solution, fitness_over_time

# Jalankan algoritma
best_solution, fitness_over_time = genetic_algorithm()
