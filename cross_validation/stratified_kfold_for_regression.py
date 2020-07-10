# Stratified-Kfold for regression
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection


def create_folds(data: pd.DataFrame) -> pd.DataFrame:
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule
    # I take the floor of the value, you can also just round it
    num_bins = np.floor(1 + np.log2(len(data)))

    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"],
        bins=num_bins,
        labels=False
    )

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f

    # drop the bins column
    data = data.drop("bins", axis=1)

    return data


if __name__ == "__main__":
    X, y = datasets.make_regression(
        n_samples=15000,
        n_features=100,
        n_targets=1
    )

    df = pd.DataFrame(
        X,
        columns=[f"f_{i}" for i in range(X.shape[1])]
    )
    df.loc[:, "target"] = y

    df = create_folds(df)
