import random
import numpy as np
import math
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

# Implementasi Simulated Annealing
def simulated_annealing(cube, max_iterations=1000, initial_temp=100, cooling_rate=0.99):
    current_cube = cube.copy()
    current_objective = calculate_objective(current_cube)
    best_objective = current_objective
    iteration_objectives = [current_objective]  # Untuk plot
    stuck_count = 0  # Menghitung frekuensi 'stuck' di local optima
    acceptance_probs = []  # Untuk plot e^(deltaE/T)
    
    start_time = time.time()

    for iteration in range(max_iterations):
        # Menghitung suhu saat ini
        temperature = initial_temp * (cooling_rate ** iteration)
        
        # Cek jika suhu sudah sangat rendah, hentikan pencarian
        if temperature <= 0.1:
            break
        
        # Menukar dua posisi acak dalam kubus
        pos1 = tuple(np.random.randint(0, current_cube.shape[0], size=3))
        pos2 = tuple(np.random.randint(0, current_cube.shape[0], size=3))
        current_cube[pos1], current_cube[pos2] = current_cube[pos2], current_cube[pos1]
        
        # Hitung nilai objective baru
        new_objective = calculate_objective(current_cube)
        
        # Hitung perubahan energi dan probabilitas penerimaan
        delta_e = new_objective - current_objective
        if delta_e < 0:
            accept = True  # Terima solusi lebih baik langsung
        else:
            acceptance_probability = math.exp(-delta_e / temperature)
            accept = acceptance_probability > random.random()
            acceptance_probs.append(acceptance_probability)  # Untuk plot
            
        if accept:
            current_objective = new_objective
            if new_objective < best_objective:
                best_objective = new_objective
        else:
            # Jika tidak diterima, kembalikan ke kondisi awal dan hitung sebagai stuck
            current_cube[pos1], current_cube[pos2] = current_cube[pos2], current_cube[pos1]
            stuck_count += 1

        iteration_objectives.append(current_objective)

        # Print setiap 100 iterasi untuk melihat progres
        if iteration % 100 == 0:
            print(f"Iterasi {iteration}, Nilai Objective: {current_objective}, Temperature: {temperature}")
        
        # Jika mencapai optimal (objective = 0), berhenti
        if current_objective == 0:
            break

    end_time = time.time()
    duration = end_time - start_time

    return current_cube, current_objective, iteration_objectives, duration, acceptance_probs, stuck_count

# Menjalankan eksperimen
cube = initialize_cube()
print("State Awal Kubus:")
print(cube)

# Menjalankan simulated annealing
final_cube, final_objective, iteration_objectives, duration, acceptance_probs, stuck_count = simulated_annealing(cube)

print("\nState Akhir Kubus:")
print(final_cube)
print("\nNilai Objective Akhir:", final_objective)
print("Durasi Pencarian:", duration, "detik")
print("Frekuensi 'stuck' di local optima:", stuck_count)

# Plot nilai objective function terhadap iterasi
plt.plot(iteration_objectives)
plt.xlabel("Iterasi")
plt.ylabel("Nilai Objective Function")
plt.title("Perkembangan Nilai Objective Function pada Simulated Annealing")
plt.show()

# Plot nilai e^(deltaE/T) terhadap iterasi
plt.plot(acceptance_probs)
plt.xlabel("Iterasi")
plt.ylabel("Probabilitas Penerimaan (e^(deltaE/T))")
plt.title("Plot e^(deltaE/T) terhadap Iterasi pada Simulated Annealing")
plt.show()
