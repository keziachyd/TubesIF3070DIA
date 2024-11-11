import numpy as np
import random
import time
import matplotlib.pyplot as plt

N = 5 
max_iteration = 20000  
magicSum = N * (N**3 + 1) // 2  

def randomize_cube():
    angka = list(range(1, N**3 + 1))
    random.shuffle(angka)
    return np.array(angka).reshape(N, N, N)

def count_objective(cube):
    row_sum_error = np.sum(np.abs(np.sum(cube, axis=0) - magicSum))
    column_sum_error = np.sum(np.abs(np.sum(cube, axis=1) - magicSum))
    depth_sum_error = np.sum(np.abs(np.sum(cube, axis=2) - magicSum))
    
    diag1_sum = sum(cube[i, i, i] for i in range(N))
    diag2_sum = sum(cube[i, i, N - i - 1] for i in range(N))
    diag3_sum = sum(cube[i, N - i - 1, i] for i in range(N))
    diag4_sum = sum(cube[N - i - 1, i, i] for i in range(N))
    
    diag_error = abs(diag1_sum - magicSum) + abs(diag2_sum - magicSum) + \
                 abs(diag3_sum - magicSum) + abs(diag4_sum - magicSum)
    
    return row_sum_error + column_sum_error + depth_sum_error + diag_error

def search_bestNeighbor(cube):
    best_neighbor = cube.copy()
    best_objective = count_objective(cube)
    
    for _ in range(100): 
        neighbor = cube.copy()
        
        idx1 = tuple(np.random.randint(0, N, size=3))
        idx2 = tuple(np.random.randint(0, N, size=3))
        
        while idx1 == idx2:
            idx2 = tuple(np.random.randint(0, N, size=3))
        
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        
        neighbor_objective = count_objective(neighbor)
        
        if neighbor_objective < best_objective:
            best_neighbor = neighbor
            best_objective = neighbor_objective
        elif neighbor_objective == best_objective and best_neighbor is cube:
            best_neighbor = neighbor  
            
    return best_neighbor, best_objective

def steepest_ascent_hill_climbing():
    initial_cube = randomize_cube()
    cube = initial_cube.copy()
    best_objective = count_objective(cube)
    history = [best_objective]
    start_time = time.time()
    
    for iterasi in range(max_iteration):
        kubus_baru, objective_baru = search_bestNeighbor(cube)
        
        if objective_baru >= best_objective:
            break
        
        cube = kubus_baru
        best_objective = objective_baru
        history.append(best_objective)
    
    duration = time.time() - start_time
    return initial_cube, cube, best_objective, history, duration, iterasi

def run_experiment():
    initial_cube, last_cube, last_objective, history, duration, iterasi = steepest_ascent_hill_climbing()
    
    print("State Awal Kubus:\n", initial_cube)
    print("\nState Akhir Kubus:\n", last_cube)
    print("\nNilai Objective Function Akhir:", last_objective)
    print("Durasi Proses Pencarian:", duration, "detik")
    print("Jumlah Iterasi Hingga Berhenti:", iterasi)
    
    plt.plot(history, label="Objective Function")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai Objective Function")
    plt.title("Performa Steepest Ascent Hill-Climbing dengan Kondisi Flat")
    plt.legend()
    plt.show()

run_experiment()

