# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
from incremental_kmeans_method import IncrKmeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from strlearn.streams import StreamGenerator


# Ustawienia eksperymentu
n_informative = 10
n_chunks = 200
chunk_size = 200
p_values = [0.05, 0.5, 1, 2, 100]
iter_values = [5, 10, 100, 500, 1000]

# Inicjalizacja strumienia danych
strumien = StreamGenerator(random_state=None, n_chunks=n_chunks,
                           chunk_size=chunk_size, n_classes=2,
                           n_features=n_informative, n_informative=n_informative,
                           n_redundant=0, n_clusters_per_class=1)

# Inicjalizacja modeli z różnymi parametrami
models = []
for p in p_values:
    for iter_value in iter_values:
        model = IncrKmeans(k=2, init='k-means++', random_state=None, p=p, iter_v=iter_value)
        models.append(model)

# Ewaluacja modeli na strumieniu danych
evaluator_scores = [[] for _ in models]

# Metryki
metryki = [adjusted_rand_score, normalized_mutual_info_score]

# Ewaluacja modeli - najpierw predict na test chunk, potem fit na test chunk
for i in range(n_chunks):
    X_chunk, y_chunk = strumien.get_chunk()

    for j, model in enumerate(models):
        y_pred = np.array(model.predict(X_chunk[:, :n_informative].reshape(-1, n_informative)))
        if y_pred.size > 0 and y_chunk.size > 0:
            scores = [metric(y_chunk.ravel(), y_pred.ravel()) for metric in metryki]
            evaluator_scores[j].append(scores)
        model.partial_fit(X_chunk, y_chunk, chunk_size=chunk_size)

# Zapis wyników do pliku
evaluator_scores = np.array(evaluator_scores)
np.save('../results/exp1_evaluator_scores.npy', evaluator_scores)

# Wyświetlenie wyników
for i, model in enumerate(models):
    print('Model: p = {}, iter_v = {}'.format(model.p, model.iter_v))
    print('Adjusted Rand Score: {}'.format(np.mean(evaluator_scores[i, :, 0])))
    print('Normalized Mutual Info Score: {}'.format(np.mean(evaluator_scores[i, :, 1])))
    print('')
