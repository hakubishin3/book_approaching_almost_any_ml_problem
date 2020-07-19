import numpy as np
import torch
import torch.nn as nn
import transformers
from typing import Union


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCH = 10
BERT_PATH = "./input/bert_base_uncased/"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)


class BERTDataset:
    def __init__(self, review: Union[list, np.array[str]], target: Union[list, np.array[int]]):
        self.review = review
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.review)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        # for a given item index, return a dictionary of inputs
        review = str(self.review[item])
        review = " ".join(review.split())

        # encode_plus comes from huggingface's transformers
        # and exists for all tokenizers they offer it can be userd to
        # convert a given string to ids, mask and token type ids which are needed
        # for models like Bert.
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        # ids are ids of tokens generated after tokenizing reviews
        ids = inputs["inputs_ids"]

        # mask is 1 where we have input and 0 where we have padding
        mask = inputs["attention_mask"]

        # token type ids behave the same way as mask in this specific case
        # in case of two sentences, this is 0 for first sentence and 1 for second sentence
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(
                ids, dtype=torch.long
            ),
            "mask": torch.tensor(
                mask, dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                token_type_ids, dtype=torch.long
            ),
            "targets": torch.tensor(
                self.target[item], dtype=torch.float
            )
        }



class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()

        # we fetch the model from the BERT_PATH
        self.bert = transformers.BertModel.from_pretrained(
            BERT_PATH
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids: torch.LongTensor, mask: torch.LongTensor, token_type_ids: torch.LongTensor
    ) -> torch.Tensor:
        # Beert is its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # we use th output of pooler which is of the size
        # (batch_size, hidden_size)
        # hidden_size can be 768 or 1024 depending on
        # if we are using bert base or large respectively
        # in out case, it is 768
        _, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
