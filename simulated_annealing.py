import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

N = 5  
magicSum = N * (N**3 + 1) / 2   

def count_objective(cube):
    objective = 0
    for layer in cube:
        for row in layer:
            objective += abs(sum(row) - magicSum)
    for layer in cube:
        for column in range(N):
            column_sum = sum(layer[row][column] for row in range(N))
            objective += abs(column_sum - magicSum)
    for row in range(N):
        for column in range(N):
            pillar_sum = sum(cube[layer][row][column] for layer in range(N))
            objective += abs(pillar_sum - magicSum)
    return objective

def simulated_annealing(initial_cube, suhu_awal, cooldown, max_iteration, interval=1000):
    cube = np.copy(initial_cube)
    best_cube = np.copy(cube)
    best_objective = count_objective(cube)
    
    suhu = suhu_awal
    objectiveNow = count_objective(cube)
    
    plot_objective = []
    decay_values = []  
    stuck_count = 0
    startTime = time.time()

    # Menampilkan state awal kubus
    print("State Awal Kubus:")
    print(initial_cube)
    print("\n")  # Memberi spasi antara state awal dan proses simulasi

    for i in range(max_iteration):
        if suhu <= 0.1:  
            suhu = 0.1
       
        plot_objective.append(objectiveNow)
        
        new_cube = np.copy(cube)
        x1, y1, z1 = random.randint(0, N-1), random.randint(0, N-1), random.randint(0, N-1)
        x2, y2, z2 = random.randint(0, N-1), random.randint(0, N-1), random.randint(0, N-1)
        
        while (x1, y1, z1) == (x2, y2, z2):  # Hindari pertukaran elemen yang sama
            x2, y2, z2 = random.randint(0, N-1), random.randint(0, N-1), random.randint(0, N-1)
        
        new_cube[x1][y1][z1], new_cube[x2][y2][z2] = new_cube[x2][y2][z2], new_cube[x1][y1][z1]
        
        new_objective = count_objective(new_cube)
        delta_e = objectiveNow - new_objective
       
        decay_value = math.exp(delta_e / suhu) if suhu > 0 else 0 
        decay_values.append(decay_value) 
        
        if delta_e > 0 or decay_value > random.random():
            cube = new_cube
            objectiveNow = new_objective
        else:
            stuck_count += 1  # Increment stuck_count if new configuration is not accepted
        
        if objectiveNow < best_objective:
            best_cube = np.copy(cube)
            best_objective = objectiveNow
        
        suhu *= cooldown
        
        if (i + 1) % interval == 0:
            print(f"Iterasi {i+1}, Suhu {suhu:.4f}, Objective Saat Ini {objectiveNow}, Objective Terbaik {best_objective}")
    
    endTime = time.time()
    durasi = endTime - startTime
    
    print("\nState Akhir Kubus:")
    print(best_cube)
    print(f"Jumlah Stuck: {stuck_count}")
    print(f"Durasi Total: {durasi:.2f} detik")
    
    plt.figure()
    plt.plot(plot_objective, label="Nilai Objective")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai Fungsi Objective")
    plt.title("Perkembangan Fungsi Objective selama Iterasi")
    plt.legend()
    plt.show()
    
    plt.figure()
    decay_values = [val if val < 1e300 else 1e300 for val in decay_values]  
    plt.plot(decay_values, label="e^(Delta E / T)")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai $e^{\\Delta E / T}$")
    plt.title("$e^{\\Delta E / T}$ Selama Iterasi")
    plt.legend()
    plt.show()
    
    return best_cube, best_objective, durasi, stuck_count

# Menghasilkan kubus acak awal
initial_cube = np.arange(1, N**3 + 1)  
np.random.shuffle(initial_cube)  
initial_cube = initial_cube.reshape(N, N, N)  

initial_temp = 100.0
cooldown = 0.99  
max_iteration = 10000
interval = 1000  

best_cube, best_objective, durasi, stuck_count = simulated_annealing(initial_cube, initial_temp, cooldown, max_iteration, interval)

print("Nilai Objective Terbaik yang Ditemukan:", best_objective)
print("Konfigurasi Kubus Terbaik:")
print(best_cube)