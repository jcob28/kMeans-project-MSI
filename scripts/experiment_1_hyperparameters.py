# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
from incremental_kmeans_method import IncrKmeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from strlearn.streams import StreamGenerator
import matplotlib.pyplot as plt
import seaborn as sns


# Ustawienia eksperymentu
n_informative = 10
n_chunks = 200
chunk_size = 200
p_values = [0.05, 0.5, 1, 2, 100]
iter_values = [5, 10, 100, 500, 1000]

# Inicjalizacja strumienia danych
strumien = StreamGenerator(random_state=42, n_chunks=n_chunks,
                           chunk_size=chunk_size, n_classes=2,
                           n_features=n_informative, n_informative=n_informative,
                           n_redundant=0, n_clusters_per_class=1)

# Podział danych na zbiór treningowy i testowy
X, y = make_classification(n_samples=n_chunks * chunk_size, n_features=n_informative, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja modeli z różnymi parametrami
models = []
for p in p_values:
    for iter_value in iter_values:
        model = IncrKmeans(k=2, init='k-means++', random_state=42, p=p, iter_v=iter_value)
        models.append(model)

# Ewaluacja modeli na strumieniu danych
evaluator_scores = []
for model in models:
    evaluator_scores.append([])

# Metryki
metryki = [adjusted_rand_score, normalized_mutual_info_score]

# Ewaluacja modeli
for i in range(n_chunks):
    X_chunk, y_chunk = strumien.get_chunk()

    for j, model in enumerate(models):
        model.partial_fit(X_chunk, y_chunk)
        y_pred = model.predict(X_test)
        scores = [metric(y_test, y_pred) for metric in metryki]
        evaluator_scores[j].append(scores)

# Zapis wyników do pliku
evaluator_scores = np.array(evaluator_scores)
np.save('../results/evaluator_scores.npy', evaluator_scores)

# Wyswietlenie wyników
print(evaluator_scores)

print()

# Test t-Studenta
print("| t-Student")
for i, p in enumerate(p_values):
    for j, iter_value in enumerate(iter_values):
        print(f"| {p} | {iter_value} | {evaluator_scores[i * len(iter_values) + j].mean(axis=0)} |")

print()

# Test Wilcoxona
print("| Wilcoxon")
for i, p in enumerate(p_values):
    for j, iter_value in enumerate(iter_values):
        print(f"| {p} | {iter_value} | {evaluator_scores[i * len(iter_values) + j].mean(axis=0)} |")

# Wykresy
sns.set()

# t-Student
plt.figure(figsize=(16, 12))
for i, p in enumerate(p_values):
    for j, iter_value in enumerate(iter_values):
        plt.plot(evaluator_scores[i * len(iter_values) + j][:, 0], label=f"{p} {iter_value}")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
plt.title("t-Student", fontsize=16)
plt.ylabel("Wartość metryki", fontsize=14)
plt.xlabel("Numer iteracji", fontsize=14)
plt.savefig("exp-1-t-Student.png")
plt.show()

# Wilcoxon
plt.figure(figsize=(16, 12))
for i, p in enumerate(p_values):
    for j, iter_value in enumerate(iter_values):
        plt.plot(evaluator_scores[i * len(iter_values) + j][:, 1], label=f"{p} {iter_value}")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
plt.title("Wilcoxon", fontsize=16)
plt.ylabel("Wartość metryki", fontsize=14)
plt.xlabel("Numer iteracji", fontsize=14)
plt.savefig("exp-1-wilcoxon.png")
plt.show()
