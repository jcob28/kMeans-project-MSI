# Implementacja inkrementalnego kMeans dla strumieni danych

<h3 align="center">
Metody sztucznej inteligencji - Projekt
</h3>

## 📁 Zawartość plików
- scripts:
  - *incremental_kmeans_method.py* - implementacja algorytmu inkrementalnego KMeans w języku Python, kompatybilna z biblioteką *scikit-learn* i jej metodami: *BaseEstimator* i *ClassifierMixin*
  - *exp1_hyperparameters.py* - eksperyment nr 1 polegający na hiperparametryzacji, która umożliwia wybór najlepszego parametru *p* do dalszych eksperymentów
  - *exp2_comparison_without_drifts.py* - eksperyment nr 2 porównujący inkrementalny KMeans z algorytmem Mini Batch KMeans (bez dryfów koncepcji)
  - *exp3_comparison_with_drofts.py* - eksperyment nr 3 porównujący inkrementalny KMeans z algorytmem Mini Batch KMeans (z dryfami koncepcji)
  - *exp1_results.py* - wizualizacja wyników eksperymentu nr 1
  - *exp2_results.py* - wizualizacja wyników eksperymentu nr 2
  - *exp3_results.py* - wizualizacja wyników eksperymentu nr 3
  - *mini_batch_kmeans_algorithm.py* - algorytm Mini Batch KMEANS
  - *birch_algorithm.py* - algorytm BIRCH
- results - wizualizacja wyników poszczególnych eksperymentów

## 💿 Instalacja
- Sklonowanie repozytorium
```shell
git clone https://github.com/jcob28/kMeans-project-MSI.git
```

- Instalacja *scikit-learn*
```shell
pip install -U scikit-learn
```

- Instalacja *stream-learn*
```shell
pip3 install -U stream-learn
```

## ⚡️ Przeprowadzenie eksperymentów
W celu przeprowadzenia eksperymentów należy uruchomić skrypty:
- *exp1_hyperparameters.py*
- *exp2_comparison_without_drifts.py*
- *exp3_comparison_with_drofts.py*

## 💡 Wizualizacja wyników
Aby zwizualizować wyniki, należy uruchomić skrypty:
- *exp1_results.py*
- *exp2_results.py*
- *exp3_results.py*
