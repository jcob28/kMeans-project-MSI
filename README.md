# Implementacja inkrementalnego kMeans dla strumieni danych

<h3 align="center">
Metody sztucznej inteligencji - Projekt
</h3>

## 📁 Zawartość plików
- *incremental_kmeans_method.py* - implementacja algorytmu inkrementalnego kMeans w języku Python, kompatybilna z biblioteką *scikit-learn* i jej metodami: *BaseEstimator* i *ClassifierMixin*
- *experiments.py* - implementacja wstępnego eksperymentu pokazującego działanie badanego algorytmu, kompatybilna z biblioteką *scikit-learn* i jej metodami: *adjusted_rand_score* i *normalized_mutual_info_score* oraz z biblioteką *stream-learn* i jej metodami: *StreamGenerator* i *TestThenTrain*

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

## ⚡️ Przeprowadzenie eksperymentu 
W celu przeprowadzenia eksperymentu, należy uruchomić skrypt *experiments.py*, mając go w jednym folderze ze skryptem *incremental_kmeans_method.py*.

