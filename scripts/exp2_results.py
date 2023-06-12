import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ustawienia eksperymentu
n_informative = 10
n_chunks = 300
chunk_size = 25

# Wczytanie wyników
evaluator_scores = np.load('../results/exp2_scores.npy')

metryki = [adjusted_rand_score, normalized_mutual_info_score]

# # Wizualizacja wyników - wykresy
# for m, metryka in enumerate(metryki):
#     plt.figure(figsize=(15, 12))
#     plt.plot(evaluator_scores[0, :, m], label="IncrKmeans")
#     plt.plot(evaluator_scores[1, :, m], label="MiniBatchKMeans")
#     plt.plot(evaluator_scores[2, :, m], label="BIRCH")
#     plt.xlabel("Numer iteracji")
#     plt.ylabel(f"{metryka.__name__}")
#     plt.legend()
#     plt.savefig(f"../results/exp2_{metryka.__name__}.png")
#     plt.show()

# Testy statystyczne
print("------------------------------------")

print("IncrKmeans vs MiniBatchKMeans")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(evaluator_scores[0, :, m], evaluator_scores[1, :, m])
    print(f"{metryka.__name__} : t = {t}, p = {p}")

print("------------------------------------")

print("IncrKmeans vs BIRCH")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(evaluator_scores[0, :, m], evaluator_scores[2, :, m])
    print(f"{metryka.__name__} : t = {t}, p = {p}")

print("------------------------------------")

print("MiniBatchKMeans vs BIRCH")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(evaluator_scores[1, :, m], evaluator_scores[2, :, m])
    print(f"{metryka.__name__} : t = {t}, p = {p}")



