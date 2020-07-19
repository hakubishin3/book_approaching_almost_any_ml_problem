import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    def __init__(self, embedding_matrix: np.array) -> None:
        super(LSTM, self).__init__()

        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        # we define an input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        # embedding matrix is userd as weights of the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )

        # we dont want to train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        # a simple bidirectional LSTM with hidden size if 128
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )

        # input (512) = 128 + 128 for mean and same for max pooling
        self.out = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass data through embedding layer
        # the input is just the tokens
        x = self.embedding(x)

        # move embedding output to lstm
        x, _ = self.lstm(x)

        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        # concatenate mean and max pooling
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.out(out)

        return out

