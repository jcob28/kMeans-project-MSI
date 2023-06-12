# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
from incremental_kmeans_method import IncrKmeans
from birch_algorithm import AlgBirch
from mini_batch_kmeans_algorithm import AlgMiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from strlearn.streams import StreamGenerator


# Ustawienia eksperymentu
n_informative = 10
n_chunks = 400
chunk_size = 25
n_drifts = [3, 5, 10]

# Inicjalizacja modeli
models = [
    IncrKmeans(k=2, init='k-means++', random_state=None, p=2, iter_v=50),
    AlgMiniBatchKMeans(n_clusters=2, random_state=None),
    AlgBirch(n_clusters=2, random_state=None)
]

# Inicjalizacja strumieni danych
strumienie = []
for n_drift in n_drifts:
    strumien = StreamGenerator(random_state=None, n_chunks=n_chunks, chunk_size=chunk_size, n_classes=2,
                               n_features=n_informative, n_informative=n_informative,
                               n_redundant=0, n_clusters_per_class=1, n_drifts=n_drift)
    strumienie.append(strumien)

# Ewaluacja modeli na strumieniach danych
evaluator_scores = [[] for _ in models]

# Metryki
metryki = [adjusted_rand_score, normalized_mutual_info_score]

# Ewaluacja modeli na strumieniach danych
for i in range(n_chunks):
    for j, strumien in enumerate(strumienie):
        X_chunk, y_chunk = strumien.get_chunk()

        for k, model in enumerate(models):
            model.partial_fit(X_chunk, y_chunk, chunk_size=chunk_size)
            y_pred = model.predict(X_chunk[:, :n_informative].reshape(-1, n_informative))
            if y_pred.size > 0 and y_chunk.size > 0:
                scores = [metric(y_chunk.ravel(), y_pred.ravel()) for metric in metryki]
                evaluator_scores[k].append(scores)

# Zapis wyników do pliku
evaluator_scores = np.array(evaluator_scores)
np.save('../results_new/exp3/exp3_scores_n.npy', evaluator_scores)

# Wyswietlenie wyników
print(evaluator_scores)
