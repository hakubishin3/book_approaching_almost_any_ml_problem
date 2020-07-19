import io
import numpy as np
from numpy.core import test
import pandas as pd
from typing import Callable, Union, Any

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer


def load_vectors(fname: str) -> dict:
    fin = io.open(
        fname,
        "r",
        encoding="utf-8",
        newline="\n",
        errors="ignore"
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def sentence_to_vec(
        s: str, embedding_dict: dict, stop_words: list,
        tokenizer: Callable[..., list[Any[str]]]
        ) -> np.array:
    """
    Given a sentence and other information,
    this function returns embedding for the whole sentence.
    """
    # convert sentence to string and lowercase it
    words = str(s).lower()

    # tokenize the sentence
    words = tokenizer(words)

    # remove stop word tokens
    words = [w for w in words if not w in stop_words]

    # keep only alpha-numeric tokens
    words = [w for w in words if w.isalpha()]

    # initialize empty list to store embeddings
    M = []
    for w in words:
        # for event word, fetch the embedding from the dictionary
        # and append to list of embeddings
        if w in embedding_dict:
            M.append(embedding_dict[w])

    # if we dont have any vectors, return zeros
    if len(M) == 0:
        return np.zeros(300)

    # convert list of embedding to array
    M = np.array(M)

    # calculate sum over axis=0
    v = M.sum(axis=0)

    # return normalized vector
    return v / np.sqrt((v ** 2).sum())


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("imdb.csv")

    # load embeddings into memory
    embeddings = load_vectors("sample.vec")

    # create sentence embeddings
    vectors = []
    for review in df["review"].values:
        vectors.append(
            sentence_to_vec(
                s=review,
                embedding_dict=embeddings,
                stop_words=[],
                tokenizer=word_tokenize
            )
        )
    vectors = np.array(vectors)
