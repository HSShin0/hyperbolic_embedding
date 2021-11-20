"""Dataset and Dataloader module.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - https://github.com/facebookresearch/poincare-embeddings
"""

import os
import re
from typing import Any, Dict, Optional, Tuple, Union

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet as wn
from torch.utils.data import Dataset
from tqdm import tqdm

DEBUG = False
SEED = 0
torch.manual_seed(SEED)


class TaxonomiesDataset(Dataset):
    """Taxonomies Dataset from WordNet."""

    def __init__(
        self, filepath: str, n_neg: int = 10, word2idx: Optional[Dict[str, int]] = None
    ) -> None:

        self.n_neg = n_neg
        self.data: pd.DataFrame = pd.read_csv(filepath)
        assert isinstance(
            self.data, pd.DataFrame
        ), f"Failed to load data from {filepath}"
        assert "id1" in self.data.columns, f"The data has no `id1` column"
        assert "id2" in self.data.columns, f"The data has no `id2` column"

        # TODO: validate `word2idx` if it is not No
        if not word2idx:
            word2idx = self._get_word2idx()
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.n_words = len(self.word2idx)
        ordered_numeric_data: torch.LongTensor = self._get_numeric_data()
        self.adj_mat = self._get_adjacency_matrix(
            ordered_numeric_data["indices"],
            self.n_words,
            ordered_numeric_data["weight"],
        )
        self.nonzero_ids_in_adj = np.argwhere(self.adj_mat)

    @staticmethod
    def _get_adjacency_matrix(
        indices: Union[np.ndarray, torch.LongTensor],
        rank: int,
        weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        if weight is None:
            weight = np.ones(len(indices))
        adj_mat = np.zeros((rank, rank), dtype=np.int)
        for idx, (src, tgt) in enumerate(
            indices
        ):  # (src, tgt) is (hyponym, hypernym) in this cas)e
            adj_mat[src, tgt] = weight[idx]
            adj_mat[tgt, src] = weight[idx]
        return adj_mat

    def _get_numeric_data(
        self, return_weight: bool = True
    ) -> Dict[str, Union[np.ndarray, torch.Tensor, torch.LongTensor]]:
        cols = []
        for col_name in ["id1", "id2"]:
            numeric_col = self.data[col_name].map(lambda x: self.word2idx[x]).values
            cols.append(numeric_col)
        indices = np.stack(cols, axis=1)  # (n_words, 2)
        # NOTE: `weight` is discarded
        numeric_data = {"indices": torch.LongTensor(indices)}
        if return_weight:
            weight = self.data["weight"].values
            numeric_data.update({"weight": weight})
        return numeric_data

    def _get_word2idx(self) -> Dict[str, int]:
        hypos = set(self.data.id1.unique())
        hypers = set(self.data.id2.unique())
        unique_words = hypos.union(hypers)
        # fix order for reproducibility
        unique_words = sorted(unique_words)
        return {word: idx for idx, word in enumerate(unique_words)}

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        pos_sample = self.nonzero_ids_in_adj[idx]  # "(u, v)" in the paper
        neg_samples = self._get_negative_samples(
            pos_sample[0], self.n_neg
        )  # N(u) negative samples of `u`

        # transform to integer type tensor
        pos_sample = torch.LongTensor(pos_sample)
        neg_samples = torch.LongTensor(neg_samples)

        return {"pos": pos_sample, "negs": neg_samples}

    def __len__(self) -> int:
        return len(self.data)

    def _get_negative_samples(self, idx: int, n_samples: int = 10) -> np.ndarray:
        """Sample a set of negative samples `N(u)` for idx `u`."""
        neg_ids = np.argwhere(self.adj_mat[idx] == 0).reshape(-1)
        # n_samples = min(n_samples, len(neg_ids))
        # TODO: better method for negative sampling?
        # the below is quite ad-hoc way for batchwise operation
        # if we take a negative sample multiple times, what happen?
        replace = False if n_samples <= len(neg_ids) else True
        return np.random.choice(neg_ids, size=n_samples, replace=replace)


# TODO(hsshin): filter via regular expressions
def get_transitive_closure(
    word: str, pos: str = "n", save_root: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get transivive closure of a word in the WordNet dataset.

    Copied and Modified from: https://github.com/facebookresearch/poincare-embeddings/wordnet/transitive_closure.py
    """
    try:
        print(wn.all_synsets)
    except LookupError:
        nltk.download("wordnet")

    # make sure each edge is included only once
    edges = set()
    for synset in tqdm(wn.all_synsets(pos=pos)):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))  # (hypo, hyper)

        # also write transitive closure for all instances of a synset
        # TODO(hsshin): revisit here
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    hypo_hypers = pd.DataFrame(list(edges), columns=["id1", "id2"])
    hypo_hypers["weight"] = 1
    if DEBUG:
        print(hypo_hypers.head())

    # Extract the set of (hyponym, hypernym) that have `word` as a hypernym
    hyponyms_of_word = set(hypo_hypers[hypo_hypers.id2 == word].id1.unique())
    hyponyms_of_word.add(word)

    # Select relations that have words in `hyponyms_of_word` as hypo and hypernym
    closure = hypo_hypers[
        hypo_hypers.id1.isin(hyponyms_of_word) & hypo_hypers.id2.isin(hyponyms_of_word)
    ]

    # # TODO(hsshin): revisit here
    # prefix = word.split(".", 1)[0]
    # with open(f"mammals_filter.txt", "r") as fin:
    #     filt = re.compile(f'({"|".join([l.strip() for l in fin.readlines()])})')

    # filtered_closure = closure[~closure.id1.str.cat(" " + closure.id2).str.match(filt)]

    if save_root:
        os.makedirs(save_root, exist_ok=True)
    hypo_hypers.to_csv(os.path.join(save_root, f"{pos}_closure.csv"), index=False)
    # filtered_closure.to_csv(os.path.join(save_root, f"{word}_closure.csv"), index=False)
    closure.to_csv(os.path.join(save_root, f"{word}_closure.csv"), index=False)

    # return filtered_closure, hypo_hypers
    return closure, hypo_hypers


if __name__ == "__main__":
    # run `PYTHONPATH=. python src/data.py` in the repo root dir
    DEBUG = True
    # get_transitive_closure("mammal.n.01")
    dataset = TaxonomiesDataset("./datasets/wordnet/mammal_closure.csv")
    A = dataset.adj_mat

    # sample a relation
    SAMPLE_IDX = 2723
    temp = dataset[SAMPLE_IDX]
    sampled_data = dataset[SAMPLE_IDX]  # src, tgt have no special order here
    pos, neg_samples = sampled_data["pos"], sampled_data["negs"]

    src, tgt = pos.numpy()
    neg_samples = neg_samples.numpy()

    print(dataset.idx2word[src], dataset.idx2word[tgt])
    input()
    for idx in [src, tgt]:
        related_ids = np.argwhere(A[idx] == 1)  # shape (-1, 1)
        print(f"{dataset.idx2word[idx]}:")
        count = 0
        for idx_np in related_ids:
            idx = idx_np[0]
            print(f"\t{dataset.idx2word[idx]}")
            count += 1
            if count >= 10:
                print(f"######## End of first {count} items ########")
                break

    print(f"Negative samples of {dataset.idx2word[src]}:")
    for idx in neg_samples:
        print(f"\t{dataset.idx2word[idx]}")
