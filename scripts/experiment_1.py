# Authors: Weronika Budzowska, Jakub Leśniak


from incremental_kmeans_method import IncrKmeans
from mini_batch_kmeans_algorithm import AlgMiniBatchKMeans
from birch_algorithm import AlgBirch
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import ttest_ind, wilcoxon
from scipy.spatial.distance import minkowski
import matplotlib.pyplot as plt
import sys


"""
    Eksperyment 1.
                    Parametry:
                    · n_chunks=66,
                    · chunk_size=50,
                    · random_state=33,
                    · 3 klastry
"""

metryki = [adjusted_rand_score, normalized_mutual_info_score, minkowski]

strumien1 = StreamGenerator(random_state=33, n_chunks=66,
                            chunk_size=50, n_classes=3,
                            n_features=10, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1)

strumien2 = StreamGenerator(random_state=33, n_chunks=66,
                            chunk_size=50, n_classes=3,
                            n_features=10, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1)

strumien3 = StreamGenerator(random_state=33, n_chunks=66,
                            chunk_size=50, n_classes=3,
                            n_features=10, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1)


model1 = IncrKmeans(k=3, random_state=33)   # Inicjalizacja modelu Incremental KMeans
model2 = AlgMiniBatchKMeans(n_clusters=3, random_state=33)  # Inicjalizacja modelu Mini-Batch KMeans
model3 = AlgBirch(n_clusters=3, random_state=33)    # Inicjalizacja modelu BIRCH


ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien1, model1)
ScoresIncrKMeans = ewaluator.scores

ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien2, model2)
ScoresMiniBatch = ewaluator.scores

ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien3, model3)
ScoresBirch = ewaluator.scores


f = open("../results/results1.npy", "w")
sys.stdout = f


print("| t-Student")
print("| Incremental KMeans VS Mini-Batch KMeans")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncrKMeans[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: t = {round(t, 5)}, p = {round(p, 5)}")
print()

print("| t-Student")
print("| Incremental KMeans VS BIRCH")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncrKMeans[0, :, m], ScoresBirch[0, :, m])
    print(f"{metryka.__name__}: t = {round(t, 5)}, p = {round(p, 5)}")
print()

print()
print("| Wilcoxon")
print("| Incremental KMeans VS Mini-Batch KMeans")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncrKMeans[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: statistic = {round(statistic, 5)}, p-value = {round(pvalue, 5)}")
print()

print("| Wilcoxon")
print("| Incremental KMeans VS BIRCH")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncrKMeans[0, :, m], ScoresBirch[0, :, m])
    print(f"{metryka.__name__}: statistic = {round(statistic, 5)}, p-value = {round(pvalue, 5)}")
print()


plt.figure(figsize=(20, 14))

for m, metryka in enumerate(metryki):
    plt.plot(ScoresIncrKMeans[0, :, m], label=metryka.__name__ + " Incremental KMeans")
    plt.plot(ScoresMiniBatch[0, :, m], label=metryka.__name__+" Mini-Batch KMeans")
    plt.plot(ScoresBirch[0, :, m], label=metryka.__name__ + " BIRCH")

plt.title("Porównanie jakości klasyfikacji dla Incremental KMeans, Mini-Batch KMeans i BIRCH")
# plt.ylim(-0.5, 1)
plt.ylabel('Wartość metryki')
plt.xlabel('Numer iteracji')
plt.legend()
plt.show()


f.close()


sys.stdout = sys.__stdout__
with open('../results/results1.npy', 'r') as x:
    print(x.read())
