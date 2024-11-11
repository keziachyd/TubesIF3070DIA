import random
import numpy as np
import time
import matplotlib.pyplot as plt

# Fungsi untuk menginisialisasi kubus 5x5x5 dengan angka acak dari 1 hingga 125
def initialize_cube(size=5):
    numbers = list(range(1, size**3 + 1))
    random.shuffle(numbers)
    cube = np.array(numbers).reshape(size, size, size)
    return cube

# Fungsi untuk menghitung nilai objective function
def calculate_objective(cube):
    size = cube.shape[0]
    magic_number = (size * (size**3 + 1)) / 2
    total_error = 0

    for i in range(size):
        total_error += abs(magic_number - sum(cube[i, :, :].flatten()))  # Row
        total_error += abs(magic_number - sum(cube[:, i, :].flatten()))  # Column
        total_error += abs(magic_number - sum(cube[:, :, i].flatten()))  # Depth

    total_error += abs(magic_number - sum(cube[i, i, i] for i in range(size)))  # Diagonal utama 1
    total_error += abs(magic_number - sum(cube[i, i, size - i - 1] for i in range(size)))  # Diagonal utama 2

    return total_error

# Implementasi Steepest Ascent Hill-Climbing
def steepest_ascent_hill_climbing(cube, max_iterations=1000):
    current_cube = cube.copy()
    current_objective = calculate_objective(current_cube)
    iteration_objectives = [current_objective]  # Untuk plot
    start_time = time.time()

    for iteration in range(max_iterations):
        best_neighbor = current_cube.copy()
        best_objective = current_objective
        found_better = False

        # Coba pertukaran untuk setiap pasangan posisi dalam kubus
        for i in range(current_cube.size):
            for j in range(i + 1, current_cube.size):
                pos1 = np.unravel_index(i, current_cube.shape)
                pos2 = np.unravel_index(j, current_cube.shape)
                
                # Tukar posisi dan hitung objective function
                current_cube[pos1], current_cube[pos2] = current_cube[pos2], current_cube[pos1]
                new_objective = calculate_objective(current_cube)
                
                # Jika lebih baik dari kondisi terbaik saat ini, simpan sebagai neighbor terbaik
                if new_objective < best_objective:
                    best_neighbor = current_cube.copy()
                    best_objective = new_objective
                    found_better = True
                
                # Kembalikan pertukaran
                current_cube[pos1], current_cube[pos2] = current_cube[pos2], current_cube[pos1]

        # Jika ditemukan neighbor lebih baik, perbarui current state
        if found_better:
            current_cube = best_neighbor
            current_objective = best_objective
            iteration_objectives.append(current_objective)
        else:
            # Jika tidak ada perbaikan, berhenti
            break

        # Print setiap 100 iterasi untuk melihat progres
        if iteration % 100 == 0:
            print(f"Iterasi {iteration}, Nilai Objective: {current_objective}")
        
        # Jika mencapai optimal (objective = 0), berhenti
        if current_objective == 0:
            break

    end_time = time.time()
    duration = end_time - start_time

    return current_cube, current_objective, iteration_objectives, duration

# Menjalankan eksperimen
cube = initialize_cube()
print("State Awal Kubus:")
print(cube)

# Menjalankan steepest ascent hill climbing
final_cube, final_objective, iteration_objectives, duration = steepest_ascent_hill_climbing(cube)

print("\nState Akhir Kubus:")
print(final_cube)
print("\nNilai Objective Akhir:", final_objective)
print("Durasi Pencarian:", duration, "detik")

# Plot nilai objective function terhadap iterasi
plt.plot(iteration_objectives)
plt.xlabel("Iterasi")
plt.ylabel("Nilai Objective Function")
plt.title("Perkembangan Nilai Objective Function pada Steepest Ascent Hill-Climbing")
plt.show()
