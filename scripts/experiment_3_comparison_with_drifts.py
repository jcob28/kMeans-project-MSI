import numpy as np
from incremental_kmeans_method import IncrKmeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from strlearn.metrics import balanced_accuracy_score
from birch_algorithm import AlgBirch
from mini_batch_kmeans_algorithm import AlgMiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Ustawienia eksperymentu
n_informative = 10
n_chunks = 200
chunk_size = 200
p_values = [0.05, 0.5, 1, 2, 100]
iter_values = [5, 10, 100, 500, 1000]
drifts = [3, 5, 10]

# Inicjalizacja strumienia danych
strumienie = []
evaluators = []
models = []

for drift in drifts:
    strumien = StreamGenerator(random_state=42, n_chunks=n_chunks, chunk_size=chunk_size,
                               n_classes=2, n_features=n_informative, n_informative=n_informative,
                               n_redundant=0, n_clusters_per_class=1, n_drifts=drift)
    evaluators.append(TestThenTrain(metrics=[balanced_accuracy_score]))

    # Podział danych na zbiór treningowy i testowy
    X, y = make_classification(n_samples=n_chunks * chunk_size, n_features=n_informative, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Algorytmy
    incr_kmeans = IncrKmeans(k=2, init='k-means++', random_state=42)
    mini_batch_kmeans = AlgMiniBatchKMeans(n_clusters=2, random_state=42)
    birch = AlgBirch(n_clusters=2, random_state=42)

    models.append([incr_kmeans, mini_batch_kmeans, birch])

# Ewaluacja modeli
evaluator_scores = []
for _ in range(len(drifts)):
    evaluator_scores.append([])

# Metryki
metryki = [adjusted_rand_score, normalized_mutual_info_score]

# Ewaluacja modeli
for i in range(n_chunks):
    for j, model_group in enumerate(models):
        for model in model_group:
            X_chunk, y_chunk = strumienie[j].get_chunk()
            model.partial_fit(X_chunk, y_chunk)
            y_pred = model.predict(X_test)
            scores = [metric(y_test, y_pred) for metric in metryki]
            evaluator_scores[j].append(scores)
            evaluators[j].add_result(scores, y_test, y_pred)

# Wyniki
for i, drift in enumerate(drifts):
    evaluator_scores[i] = np.array(evaluator_scores[i])

    # Zapis wyników do pliku
    np.save(f'evaluator_scores_comparison_drift_{drift}.npy', evaluator_scores[i])

    # Wyswietlanie wyników
    print(evaluator_scores)

    print()

    # Test t-Studenta
    print("| t-Student")
    for i, drift in enumerate(drifts):
        print(f"Drift: {drift}")
        for j, model_group in enumerate(models):
            for k, model in enumerate(model_group):
                print(f"| {model.__class__.__name__} | {evaluator_scores[i][:, k].mean(axis=0)[0]:.4f} |")

    print()

    # Test Wilcoxona
    print("| Wilcoxon")
    for i, drift in enumerate(drifts):
        print(f"Drift: {drift}")
        for j, model_group in enumerate(models):
            for k, model in enumerate(model_group):
                print(f"| {model.__class__.__name__} | {evaluator_scores[i][:, k].mean(axis=0)[1]:.4f} |")

    # Wykresy
    sns.set()

    for i, drift in enumerate(drifts):
        evaluator_scores[i] = np.array(evaluator_scores[i])

        # Wykresy t-Studenta
        plt.figure(figsize=(16, 12))
        for j, model_group in enumerate(models):
            for k, model in enumerate(model_group):
                plt.plot(evaluator_scores[i][:, k][:, 0], label=model.__class__.__name__)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
        plt.title(f"t-Student - Drift {drift}", fontsize=16)
        plt.ylabel("Wartość metryki", fontsize=14)
        plt.xlabel("Numer iteracji", fontsize=14)
        plt.show()

        # Wykresy Wilcoxon
        plt.figure(figsize=(16, 12))
        for j, model_group in enumerate(models):
            for k, model in enumerate(model_group):
                plt.plot(evaluator_scores[i][:, k][:, 1], label=model.__class__.__name__)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
        plt.title(f"Wilcoxon - Drift {drift}", fontsize=16)
        plt.ylabel("Wartość metryki", fontsize=14)
        plt.xlabel("Numer iteracji", fontsize=14)
        plt.show()

