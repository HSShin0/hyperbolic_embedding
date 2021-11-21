"""Trainer module.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from tqdm import tqdm

if TYPE_CHECKING:
    from src.optimizer import PoincareSGD
    from src.distance import PoincareDistance
    from src.models import BaseManifold
    from src.data import TaxonomiesDataset
    from wandb.sdk.wandb_run import Run

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


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
        wandb_run: Optional["Run"] = None,
        device: torch.device = torch.device("cpu"),
        burn_in_epochs: int = 10,
    ) -> None:
        self.model = model
        self.distance = distance
        self.loss_ftn = loss_ftn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.burn_in_epochs = burn_in_epochs

        loss_ftn = nn.CrossEntropyLoss()

        # Read configuration
        self.epochs = config["epochs"]
        self.eval_every = config["eval_every"]
        self.emb_dim = config["emb_dim"]
        self.n_neg = config["n_neg"]
        self.exp_root = config["exp_root"]

        # Evaluate "scores"
        self.best_score = 0

    def train(self) -> None:
        for epoch in range(self.epochs):
            epoch_log = {}
            self.model.train()

            if epoch < self.burn_in_epochs:
                self.optimizer.burn_in()

            avg_loss = self.train_one_epoch(epoch)
            epoch_log["train_loss"] = avg_loss

            if (epoch + 1) % self.eval_every == 0:
                self.model.eval()
                scores = self.evaluate()
                for k, v in scores.items():
                    print(f"{k}:\t{v:.2f}")
                epoch_log.update(scores)

                score = scores["mAP"]  # or Use negative "mean_rank"
                if score > self.best_score:
                    self.best_score = score
                    savepath = os.path.join(self.exp_root, "best.pt")
                    additional_info = {
                        "epoch": epoch,
                        "best_score": self.best_score,
                    }
                    self.save_checkpoint(savepath, additional_info)
                    print(f"Best model is saved in {savepath}")

            if self.wandb_run:
                self.wandb_run.log(epoch_log)

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

            self.model = self.model.to(self.device)
            embed_u_v_neg_u = self.model(u_v_neg_u.to(self.device))
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

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate via Mean Rank and Average Precision."""
        adj_mat = self.dataloader.dataset.adj_mat
        W = self.model.embed.weight
        N, d = W.shape

        def compute_distance(W: torch.Tensor) -> torch.Tensor:
            W_src = W.unsqueeze(0).expand(N, N, d)  # (1, N, d) > (N, N, d)
            W_tgt = W.unsqueeze(1).expand(N, N, d)  # (N, 1, d) > (N, N, d)
            dist_flat = self.distance.apply(
                W_src.reshape(-1, d), W_tgt.reshape(-1, d), 1e-5
            )
            return dist_flat.reshape(N, N)

        dist_mat = compute_distance(W).cpu()

        # Change distance between the same node "d(u,u)" = 0 -> huge number
        dist_mat.diagonal(0).fill_(1e12)

        APs = []
        for adj, dist in zip(adj_mat, dist_mat):
            y_true = 1 * (adj > 0)  # 1 or 0
            y_score = -dist
            ap = average_precision_score(y_true, y_score)
            APs.append(ap)
        total_aps = torch.Tensor(APs)
        # TODO: implement computation of Mean Rank
        return {"mAP": total_aps.mean().item(), "mean_rank": 1}

    def save_checkpoint(
        self, savepath: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if additional_info:
            checkpoint.update(additional_info)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        torch.save(checkpoint, savepath)
