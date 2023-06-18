# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Ustawienia eksperymentu
n_informative = 10
n_chunks = 500
chunk_size = 250

# Wczytanie wyników
evaluator_scores = np.load('../results/exp2/exp2_scores.npy')

# Określenie metryk
metryki = [adjusted_rand_score, normalized_mutual_info_score]


"""
    Wizualizacja wyników
"""
for m, metryka in enumerate(metryki):
    plt.figure(figsize=(32, 20))
    plt.plot(evaluator_scores[0, :, m], label="IncrKmeans")
    plt.plot(evaluator_scores[1, :, m], label="MiniBatchKMeans")
    plt.xlabel("Numer iteracji", fontsize='26')
    plt.ylabel(f"{metryka.__name__}", fontsize='26')
    plt.legend(fontsize='26')
    plt.gca().xaxis.set_tick_params(labelsize=24)
    plt.gca().yaxis.set_tick_params(labelsize=24)
    plt.gca().grid(True, linestyle='--', linewidth=1)
    plt.savefig(f"../results/exp2/exp2_{metryka.__name__}.png")
    plt.show()

print("------------------------------------")

print("IncrKmeans vs MiniBatchKMeans")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(evaluator_scores[0, :, m], evaluator_scores[1, :, m])
    print(f"{metryka.__name__} : t = {t}, p = {p}")

print("------------------------------------", "\n")


"""
    Podział na foldy
"""
print("IncrKmeans vs MiniBatchKMeans")
for m, metryka in enumerate(metryki):
    fold_size = 50
    for i in range(10):
        t, p = ttest_ind(evaluator_scores[0, i * fold_size: (i + 1) * fold_size, m],
                         evaluator_scores[1, i * fold_size: (i + 1) * fold_size, m])
        print(f"{metryka.__name__} : t = {t}, p = {p}")
    print("------------------------------------")

for m, metryka in enumerate(metryki):
    fold_size = 50
    plt.figure(figsize=(32, 20))
    plt.plot(np.mean(evaluator_scores[0, :, m].reshape(10, fold_size), axis=1), label="IncrKmeans")
    plt.plot(np.mean(evaluator_scores[1, :, m].reshape(10, fold_size), axis=1), label="MiniBatchKMeans")
    plt.xlabel("Numer iteracji", fontsize='26')
    plt.ylabel(f"{metryka.__name__}", fontsize='26')
    plt.legend(fontsize='26')
    plt.gca().xaxis.set_tick_params(labelsize=24)
    plt.gca().yaxis.set_tick_params(labelsize=24)
    plt.gca().grid(True, linestyle='--', linewidth=1)
    plt.savefig(f"../results/exp2/exp2_mean_{metryka.__name__}.png")
    plt.show()

print()

"""
    Testy statystyczne
"""
n_classifiers = 2
t_statistic = np.zeros((n_classifiers, n_classifiers))
p_values = np.zeros((n_classifiers, n_classifiers))
better_results = np.zeros((n_classifiers, n_classifiers), dtype=bool)
statistical_significance = np.zeros((n_classifiers, n_classifiers), dtype=bool)

for i in range(n_classifiers):
    for j in range(n_classifiers):
        if i == j:
            continue
        t, p = ttest_rel(evaluator_scores[i, :, m], evaluator_scores[j, :, m])
        t_statistic[i, j] = t
        p_values[i, j] = p
        better_results[i, j] = np.mean(evaluator_scores[i, :, m]) > np.mean(evaluator_scores[j, :, m])

alpha = 0.05
statistical_significance = p_values < alpha

advantage = better_results * statistical_significance

print(t_statistic, "\n")
print(p_values, "\n")
print(better_results, "\n")
print(statistical_significance, "\n")
print(advantage, "\n")


"""
    Wyniki do tabeli statystycznej
"""
print(" ------------------------------------")
print("|            IncrKmeans              |")
print(" ------------------------------------")
for m, metryka in enumerate(metryki):
    fold_size = 50
    print("| ", metryka.__name__)
    for i in range(10):
        mean_score = np.mean(evaluator_scores[0, i * fold_size:(i+1)*fold_size, m])
        std_score = np.std(evaluator_scores[0, i * fold_size:(i+1)*fold_size, m])
        formatted_score = "{:.3f}({:d})".format(mean_score, int(std_score * 1000))
        print(formatted_score.rstrip("0").rstrip("."))
    print("------------------------------------")

print()

print(" ------------------------------------")
print("|          MiniBatchKMeans           |")
print(" ------------------------------------")
for m, metryka in enumerate(metryki):
    fold_size = 50
    print("| ", metryka.__name__)
    for i in range(10):
        mean_score = np.mean(evaluator_scores[1, i * fold_size:(i+1)*fold_size, m])
        std_score = np.std(evaluator_scores[1, i * fold_size:(i+1)*fold_size, m])
        formatted_score = "{:.3f}({:d})".format(mean_score, int(std_score * 1000))
        print(formatted_score.rstrip("0").rstrip("."))
    print("------------------------------------")
