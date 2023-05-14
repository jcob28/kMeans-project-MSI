# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


# klasa dziedzicząca po BaseEstimator i ClassifierMixin
class IncrKmeans(BaseEstimator, ClassifierMixin):

    """
        Konstruktor:    - ustawienie wartości domyślnych dla par. k, init i random_state
                        - oraz utworzenie atrybutów centroidy, licznik_klastrow i klasy
    """
    def __init__(self, k=2, init='k-means++', random_state=None):
        # liczba klastrów
        self.k = k
        # inicjalizacja centroidów
        self.init = init
        # ziarno losowości
        self.random_state = random_state
        # centroidy klastrów
        self.centroidy = None
        # liczba przykładów w klastrach
        self.licznik_klastrow = None
        # klasy przykładów
        self.klasy = None

    """
        Metoda predict:     - przypisanie każdego punktu do najbliższego klastra 
                            - zwrócenie przypisanych klastrów
    """
    def predict(self, X):
        najblizszy_klaster = self._przypisz_klastry(X, self.centroidy)  # przypisanie klastrów

        return najblizszy_klaster  # zwrócenie przypisanych klastrów

    """
        Metoda partial_fit:     - przyjęcie macierzy X, wektora y i wartości classes
                                - inicializacja centroidów
                                - aktualizacja klas przykładów
                                - przypisanie punktów do najbliższego klastra
                                - aktualizacja centroidów
    """
    def partial_fit(self, X, y=None, classes=None):
        # inicjalizacja centroidów
        if self.centroidy is None:
            if self.init != 'k-means++':
                # losowa inicjalizacja centroidów
                self.centroidy = X[np.random.choice(X.shape[0], self.k, replace=False)]
            else:
                # inicjalizacja centroidów metodą incr-kmeans
                self.centroidy = self._inicjalizuj_KMeansPlusPlus(X, self.k, self.random_state)

        # przypisanie klastrów z uwzględnieniem aktualizacji centroidów
        najblizszy_klaster = self._przypisz_klastry(X, self.centroidy)
        self._aktualizuj_centroidy(X, najblizszy_klaster)

        return self

    """
        Metoda _przypisz_klastry:       - obliczenie odległości od centroidów
                                        - przypisanie klastrów do najbliższych centroidów
                                        - zwrócenie przypisanych klastrów
    """
    def _przypisz_klastry(self, X, centroidy):
        # obliczenie odległości od centroidów
        odleglosci = np.linalg.norm(X[:, np.newaxis, :] - centroidy, axis=2)
        # przypisanie klastrów do najbliższych centroidów
        najblizszy_klaster = np.argmin(odleglosci, axis=1)

        return najblizszy_klaster

    """
        Metoda _aktualizuj_centroidy:    - aktualizacja centroidów jako średnieh punktów w klastrach
                                         - iteracja po każdym klastrze i utworzenie maski, 
                                             która zwraca True dla punktów należących do klastra
                                         - obliczenie średniej dla punktów należących do klastra
                                         - aktualizacja centroidu
                                         - aktualizacja licznika klastrów
                                            
    """
    def _aktualizuj_centroidy(self, X, najblizszy_klaster):
        for k in range(self.k):
            maska = najblizszy_klaster == k

            if np.sum(maska) > 0:
                self.centroidy[k] = np.average(X[maska], axis=0)

    """
        Metoda _inicjalizuj_incr_kmeans:     - losowy wybór pierwszego centroidu
                                             - iteracja po kolejnych klastrach
                                             - obliczenie minimalnych odległości od centroidów 
                                             - obliczenie prawdopodobieństwa wylosowania punktu (p)
                                             - losowy wybór centroidu zgodnie z prawdopodobieństwem p
                                             - zwrot wygenerowanych centroidów
    """
    def _inicjalizuj_KMeansPlusPlus(self, X, k, random_state):
        # pierwszy centroid losowo
        randomowe = np.random.default_rng(random_state)
        pierwszy_centroid = randomowe.choice(X.shape[0])
        centroidy = [X[pierwszy_centroid]]

        # kolejne centroidy za pomocą algorytmu KMeans++
        for x in range(1, k):
            odleglosci = np.linalg.norm(X[:, np.newaxis, :] - centroidy, axis=2)
            minimalne_odleglosci = np.min(odleglosci, axis=1)

            # p - kwadrat odległości punkty od najbliższego centroidu /
            #       suma kwadratów odległości wszystkich punktów od najbliższych centroidów
            nastepny_centroid = randomowe.choice(np.arange(X.shape[0]),
                                                 p=np.square(minimalne_odleglosci) /
                                                   np.sum(np.square(minimalne_odleglosci)))

            centroidy.append(X[nastepny_centroid])

        return np.array(centroidy)