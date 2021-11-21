"""Trainer module.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com
"""

from typing import TYPE_CHECKING, Any, Dict
from tqdm import tqdm

if TYPE_CHECKING:
    from src.optimizer import PoincareSGD
    from src.distance import PoincareDistance
    from src.models import BaseManifold
    from src.data import TaxonomiesDataset

import torch
import torch.nn as nn


class Trainer:
    # TODO: use more proper type hints
    def __init__(
        self,
        model: "BaseManifold",
        distance: "PoincareDistance",
        loss_ftn: nn.CrossEntropyLoss,
        optimizer: "PoincareSGD",
        dataloader: "TaxonomiesDataset",
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.distance = distance
        self.loss_ftn = loss_ftn
        self.optimizer = optimizer
        self.dataloader = dataloader

        loss_ftn = nn.CrossEntropyLoss()
        
        # Read configuration
        self.epochs = config["epochs"]
        self.eval_every = config["eval_every"]
        self.emb_dim = config["emb_dim"]
        self.n_neg = config["n_neg"]

    def train(self) -> None:
        for epoch in range(self.epochs):
            avg_loss = self.train_one_epoch(epoch)

            if (epoch + 1) % self.eval_every == 0:
                self.evaluate()

    def train_one_epoch(self, epoch: int) -> float:
        total_loss = 0
        pbar = tqdm(
            self.dataloader, desc=f"Epoch: {epoch}\tTrain Loss: {total_loss:.2f}"
        )
        n_iter = 0
        for batch in pbar:
            n_iter += 1
            pos, negs = batch["pos"], batch["negs"]
            u_v_neg_u = torch.cat([pos, negs], dim=-1)
            embed_u_v_neg_u = self.model(u_v_neg_u)
            u, v = (
                embed_u_v_neg_u[:, 0:1],
                embed_u_v_neg_u[:, 1:],
            )  # (N, 1, d), (N, 1 + n_neg, d)
            u_expanded = u.expand_as(v)  # (N, 1 + n_neg, d)
            batch_dist = self.distance.apply(
                u_expanded.reshape(-1, self.emb_dim), v.reshape(-1, self.emb_dim), 1e-5
            )  # (N * 1 + n_neg)

            batch_dist = batch_dist.reshape(-1, 1 + self.n_neg)  # (N, 1 + n_neg)
            # print(batch_dist.shape)
            # negative batch dist as score for Cross-Entropy with `0` ground-truth label.
            loss = self.loss_ftn(
                -batch_dist,
                torch.zeros(
                    len(batch_dist), dtype=torch.long, device=batch_dist.device
                ),
            )
            # compute euclid grad
            loss.backward()
            # Riemannian SGD for Poincare disk
            self.optimizer.step()
            # accumulate batch loss
            total_loss += loss.item()
            # Update progress bar description
            pbar.set_description(
                f"Epoch: {epoch}\tTrain Loss: {total_loss / n_iter:.2f}"
            )
        train_loss = total_loss / n_iter
        return train_loss

    def evaluate(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass
