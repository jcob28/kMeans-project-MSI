from incremental_kmeans_method import IncrKmeans
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

stream = StreamGenerator(random_state=10, n_chunks=500, chunk_size=200,
                         n_classes=5, n_features=10, n_informative=5,
                         n_redundant=0, n_clusters_per_class=1)

metrics = [adjusted_rand_score, normalized_mutual_info_score]
model = IncrKmeans(random_state=10, k=3)

evaluator = TestThenTrain(metrics)
evaluator.process(stream, model)

plt.figure()
for i, metrics in enumerate(metrics):
    plt.plot(evaluator.scores[0, :, i], label=metrics.__name__)
plt.legend()
plt.show()

exit(0)
