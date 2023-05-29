# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
from incremental_kmeans_method import IncrKmeans
from birch_algorithm import AlgBirch
from mini_batch_kmeans_algorithm import AlgMiniBatchKMeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from strlearn.streams import StreamGenerator
import matplotlib.pyplot as plt
import seaborn as sns


# Ustawienia eksperymentu
n_informative = 10
# n_chunks = 1000
# chunk_size = 250
n_chunks = 100
chunk_size = 25

# Inicjalizacja strumienia danych
strumien = StreamGenerator(random_state=42, n_chunks=n_chunks,
                           chunk_size=chunk_size, n_classes=2,
                           n_features=n_informative, n_informative=n_informative,
                           n_redundant=0, n_clusters_per_class=1)

# Podział danych na zbiór treningowy i testowy
X, y = make_classification(n_samples=n_chunks * chunk_size, n_features=n_informative, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja modeli
models = [
    IncrKmeans(k=2, init='k-means++', random_state=42, p=2, iter_v=1000),
    AlgMiniBatchKMeans(n_clusters=2, random_state=42),
    AlgBirch(n_clusters=2, random_state=42)
]

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
np.save('../results/evaluator_scores_comparison.npy', evaluator_scores)

# Wyswietlenie wyników
print(evaluator_scores)

print()

# Test t-Studenta
print("| t-Student")
for model in models:
    print(f"| {model.__class__.__name__} | {evaluator_scores[models.index(model)].mean(axis=0)} |")

print()

# Test Wilcoxona
print("| Wilcoxon")
for model in models:
    print(f"| {model.__class__.__name__} | {evaluator_scores[models.index(model)].mean(axis=0)} |")

# Wykresy
sns.set()

# t-Student
plt.figure(figsize=(16, 12))
for model in models:
    plt.plot(evaluator_scores[models.index(model)][:, 0], label=model.__class__.__name__)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
plt.title("t-Student", fontsize=16)
plt.ylabel("Wartość metryki", fontsize=14)
plt.xlabel("Numer iteracji", fontsize=14)
plt.savefig("without-drifts-t-student.png")
plt.show()

# Wilcoxon
plt.figure(figsize=(16, 12))
for model in models:
    plt.plot(evaluator_scores[models.index(model)][:, 1], label=model.__class__.__name__)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
plt.title("Wilcoxon", fontsize=16)
plt.ylabel("Wartość metryki", fontsize=14)
plt.xlabel("Numer iteracji", fontsize=14)
plt.savefig("without-drifts-wilcoxon.png")
plt.show()

