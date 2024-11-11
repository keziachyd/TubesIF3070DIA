import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Konstanta
N = 5  # Ukuran kubus
ITERASI_MAKSIMUM = 20000  # Jumlah maksimum iterasi
MAGIC_SUM = N * (N**3 + 1) // 2  # Nilai magic number yang ditargetkan

# Inisialisasi kubus secara acak dengan angka 1 hingga N^3
def inisialisasi_kubus():
    angka = list(range(1, N**3 + 1))
    random.shuffle(angka)
    return np.array(angka).reshape(N, N, N)

# Fungsi untuk menghitung objective function
def hitung_objective(kubus):
    """
    Menghitung total error dari row, column, depth, dan diagonal
    terhadap nilai MAGIC_SUM.
    """
    # Error dari baris, kolom, dan kedalaman
    row_sum_error = np.sum(np.abs(np.sum(kubus, axis=0) - MAGIC_SUM))
    col_sum_error = np.sum(np.abs(np.sum(kubus, axis=1) - MAGIC_SUM))
    depth_sum_error = np.sum(np.abs(np.sum(kubus, axis=2) - MAGIC_SUM))
    
    # Error dari empat diagonal utama
    diag1_sum = sum(kubus[i, i, i] for i in range(N))
    diag2_sum = sum(kubus[i, i, N - i - 1] for i in range(N))
    diag3_sum = sum(kubus[i, N - i - 1, i] for i in range(N))
    diag4_sum = sum(kubus[N - i - 1, i, i] for i in range(N))
    
    diag_error = abs(diag1_sum - MAGIC_SUM) + abs(diag2_sum - MAGIC_SUM) + \
                 abs(diag3_sum - MAGIC_SUM) + abs(diag4_sum - MAGIC_SUM)
    
    # Total error sebagai objective function
    return row_sum_error + col_sum_error + depth_sum_error + diag_error

# Fungsi untuk mencari tetangga terbaik dengan menukar dua elemen secara acak
def cari_tetangga_terbaik(kubus):
    best_neighbor = kubus.copy()
    best_objective = hitung_objective(kubus)
    
    for _ in range(100):  # Mencoba 100 tetangga
        neighbor = kubus.copy()
        
        # Pilih dua posisi acak dalam kubus untuk ditukar
        idx1 = tuple(np.random.randint(0, N, size=3))
        idx2 = tuple(np.random.randint(0, N, size=3))
        
        # Pastikan idx1 dan idx2 tidak sama
        while idx1 == idx2:
            idx2 = tuple(np.random.randint(0, N, size=3))
        
        # Tukar dua elemen
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        
        # Hitung objective dari tetangga
        neighbor_objective = hitung_objective(neighbor)
        
        # Jika objective lebih baik atau sama, perbarui tetangga terbaik
        if neighbor_objective < best_objective:
            best_neighbor = neighbor
            best_objective = neighbor_objective
        elif neighbor_objective == best_objective and best_neighbor is kubus:
            best_neighbor = neighbor  # Tetangga dengan objective sama dianggap flat
            
    return best_neighbor, best_objective

# Fungsi utama untuk Steepest Ascent Hill-Climbing dengan kondisi flat
def steepest_ascent_hill_climbing():
    kubus_awal = inisialisasi_kubus()
    kubus = kubus_awal.copy()
    best_objective = hitung_objective(kubus)
    history = [best_objective]
    start_time = time.time()
    
    for iterasi in range(ITERASI_MAKSIMUM):
        # Cari tetangga terbaik
        kubus_baru, objective_baru = cari_tetangga_terbaik(kubus)
        
        # Jika tidak ada perbaikan atau flat, berhenti
        if objective_baru >= best_objective:
            break
        
        # Perbarui kubus dan objective
        kubus = kubus_baru
        best_objective = objective_baru
        history.append(best_objective)
    
    duration = time.time() - start_time
    return kubus_awal, kubus, best_objective, history, duration, iterasi

# Menjalankan eksperimen dan menampilkan hasilnya
def run_experiment():
    # Menjalankan algoritma hill climbing
    kubus_awal, kubus_akhir, objective_akhir, history, duration, iterasi = steepest_ascent_hill_climbing()
    
    # Menampilkan hasil
    print("State Awal Kubus:\n", kubus_awal)
    print("\nState Akhir Kubus:\n", kubus_akhir)
    print("\nNilai Objective Function Akhir:", objective_akhir)
    print("Durasi Proses Pencarian:", duration, "detik")
    print("Jumlah Iterasi Hingga Berhenti:", iterasi)
    
    # Plot nilai objective function terhadap iterasi
    plt.plot(history, label="Objective Function")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai Objective Function")
    plt.title("Performa Steepest Ascent Hill-Climbing dengan Kondisi Flat")
    plt.legend()
    plt.show()

# Menjalankan eksperimen
run_experiment()

