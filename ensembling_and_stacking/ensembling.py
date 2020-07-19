import numpy as np
from scipy import stats


def mean_predictions(probas: np.array) -> np.array:
    return np.mean(probas, axis=1)


def max_voting(preds: np.array) -> np.array:
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)


def rank_mean(probas: np.array) -> np.array:
    ranked = []
    for i in range(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)
    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)
