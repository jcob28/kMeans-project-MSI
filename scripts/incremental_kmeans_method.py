# Authors: Weronika Budzowska, Jakub Leśniak


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist

# klasa dziedzicząca po BaseEstimator i ClassifierMixin
class IncrKmeans(BaseEstimator, ClassifierMixin):

    """
            Konstruktor:    - ustawienie wartości domyślnych dla par. k, init i random_state
                            - oraz utworzenie atrybutów centroidy, licznik_klastrow i klasy
    """
    def __init__(self, k=2, init='k-means++', random_state=None, p=2, iter_v=100):
        # liczba klastrów
        self.k = k
        # metoda inicjalizacji centroidów
        self.init = init
        # ziarno losowości
        self.random_state = random_state
        # centroidy klastrów
        self.centroidy = None
        # liczba przykładów w klastrach
        self.licznik_klastrow = None
        # klasy przykładów
        self.klasy = None
        # parametr metody
        self.p = p
        # liczba iteracji
        self.iter_v = iter_v

    """
           Metoda predict:     - przypisanie każdego punktu do najbliższego klastra 
                               - zwrócenie przypisanych klastrów
    """
    def predict(self, X):
        najblizszy_klaster = self._przypisz_klastry(X, self.centroidy)  # przypisanie klastrów

        return np.array(najblizszy_klaster) # zwrócenie przypisanych klastrów

    """
            Metoda partial_fit:     - przyjęcie macierzy X, wektora y i wartości classes
                                    - inicializacja centroidów
                                    - aktualizacja klas przykładów
                                    - przypisanie punktów do najbliższego klastra
                                    - aktualizacja centroidów
    """
    def partial_fit(self, X, y, chunk_size):
        # inicializacja centroidów
        if self.centroidy is None:
            self.centroidy = self._inicjalizuj_centroidy(X)
            self.licznik_klastrow = np.zeros(self.k, dtype=int)
            self.klasy = np.zeros(X.shape[0])

        for _ in range(chunk_size):
            przypisane_klasy = self._przypisz_klastry(X, self.centroidy)
            if np.any(przypisane_klasy):
                self.licznik_klastrow += np.bincount(przypisane_klasy, minlength=self.k)
                self.centroidy = self._oblicz_nowe_centroidy(X, przypisane_klasy)

    """
            Metoda _przypisz_klastry:       - obliczenie odległości od centroidów
                                            - przypisanie klastrów do najbliższych centroidów
                                            - zwrócenie przypisanych klastrów
    """
    def _przypisz_klastry(self, X, centroidy):
        if centroidy is None or centroidy.shape[0] == 0:
            return []

        odleglosci = cdist(X, centroidy, metric='minkowski', p=self.p)
        najblizszy_klaster = np.argmin(odleglosci, axis=1)
        return najblizszy_klaster

    """
            Metoda _inicjalizuj_centroidy:   - wybór metody inicjalizacji centroidów
    """
    def _inicjalizuj_centroidy(self, X):
        if self.init == 'k-means++':
            return self._kmeans_plus_plus(X)
        else:
            raise ValueError("Nieznana metoda inicjalizacji centroidów: {}".format(self.init))

    """
            Metoda _inicjalizuj_incr_kmeans:     - losowy wybór pierwszego centroidu
                                                 - iteracja po kolejnych klastrach
                                                 - obliczenie minimalnych odległości od centroidów 
                                                 - obliczenie prawdopodobieństwa wylosowania punktu (p)
                                                 - losowy wybór centroidu zgodnie z prawdopodobieństwem p
                                                 - zwrot wygenerowanych centroidów
    """
    def _kmeans_plus_plus(self, X):
        generator_liczb_losowych = np.random.RandomState(seed=self.random_state)
        centroidy = np.empty((self.k, X.shape[1]))
        centroidy[0] = X[generator_liczb_losowych.randint(X.shape[0])]

        for i in range(1, self.k):
            odleglosci = cdist(X, centroidy[:i], metric='minkowski', p=self.p)
            kwadraty_odleglosci = np.min(odleglosci, axis=1) ** 2
            suma_kwadratow_odleglosci = np.sum(kwadraty_odleglosci)
            prawdopodobienstwa = kwadraty_odleglosci / suma_kwadratow_odleglosci
            indeks_wybranego_centroidu = generator_liczb_losowych.choice(X.shape[0], p=prawdopodobienstwa)
            centroidy[i] = X[indeks_wybranego_centroidu]

        return centroidy

    """
            Metoda _oblicz_nowe_centroidy:   - obliczenie nowych centroidów
    """
    def _oblicz_nowe_centroidy(self, X, przypisane_klasy):
        nowe_centroidy = np.empty((self.k, X.shape[1]))
        for i in range(self.k):
            if np.sum(przypisane_klasy == i) > 0:
                nowe_centroidy[i] = np.mean(X[przypisane_klasy == i], axis=0)
            else:
                nowe_centroidy[i] = np.zeros(X.shape[1])
        return nowe_centroidy
