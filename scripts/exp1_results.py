# Authors: Weronika Budzowska, Jakub Leśniak

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienia eksperymentu
n_informative = 10
n_chunks = 200
chunk_size = 200
p_values = [0.05, 0.5, 1, 2, 100]
iter_values = [5, 10, 100, 500, 1000]

# Wizualizacja wyników
evaluator_scores = np.load('../results/exp1_evaluator_scores.npy')
evaluator_scores = np.mean(evaluator_scores, axis=1)
evaluator_scores = np.reshape(evaluator_scores, (len(p_values), len(iter_values), 2))
evaluator_scores = np.mean(evaluator_scores, axis=2)
ax = sns.heatmap(evaluator_scores, annot=True, fmt='.6f', xticklabels=iter_values, yticklabels=p_values)
ax.set(xlabel='Liczba iteracji', ylabel='Parametr p')
plt.savefig('../results/exp1_heatmap.png')
plt.show()
