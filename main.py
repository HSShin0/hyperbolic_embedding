"""Train embeddings to hyperbolic spaces.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - M. Nickel and D. Kiela, "Poincare Embeddings for Learning Hierarchical
 Representations"
"""

import argparse

# import wandb
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import TaxonomiesDataset
from src.distance import PoincareDistance
from src.models import Poincare
from src.optimizer import PoincareSGD
from src.trainer import Trainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--space-type",
        type=str,
        choices=["euclid", "poincare"],
        default="poincare",
        help="embbeding space type",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="./datasets/wordnet/mammal_closure.csv",
        help="Data csv filepath.",
    )
    parser.add_argument(
        "--n-neg",
        type=int,
        default=10,
        help="Number of negative samples for each positive pair.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--emb-dim", type=int, default=2, help="Embedding dimension.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval-every", type=int, default=5, help="Evaluate every `eval_every`-epochs."
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default="exp/temp",
        help="Root dir for saving checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dataset = TaxonomiesDataset(args.datapath, args.n_neg)
    # TODO: figure out the reason:
    # setting `num_workers` > 0 make it slow
    # See: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/23
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=0)

    # Prepare Poincare Embedding model
    model = Poincare(dataset.n_words, args.emb_dim)
    distance = PoincareDistance()
    loss_ftn = nn.CrossEntropyLoss()

    optimizer = PoincareSGD(model.parameters(), lr=0.001, eps=1e-5)
    trainer = Trainer(
        model=model,
        distance=distance,
        loss_ftn=loss_ftn,
        optimizer=optimizer,
        dataloader=dataloader,
        config=vars(args),
    )

    trainer.train()

    """
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch: {epoch}\tTrain Loss: {total_loss:.2f}")
        n_iter = 0
        for batch in pbar:
            n_iter += 1
            pos, negs = batch["pos"], batch["negs"]
            # print(pos, negs)
            # print(pos.shape, negs.shape)
            u_v_neg_u = torch.cat([pos, negs], dim=-1)
            embed_u_v_neg_u = model(u_v_neg_u)
            # print(embed_u_v_neg_u.shape)
            u, v = (
                embed_u_v_neg_u[:, 0:1],
                embed_u_v_neg_u[:, 1:],
            )  # (N, 1, d), (N, 1 + n_neg, d)
            u_expanded = u.expand_as(v)  # (N, 1 + n_neg, d)
            batch_dist = distance.apply(
                u_expanded.reshape(-1, args.emb_dim), v.reshape(-1, args.emb_dim), 1e-5
            )  # (N * 1 + n_neg)

            batch_dist = batch_dist.reshape(-1, 1 + args.n_neg)  # (N, 1 + n_neg)
            # print(batch_dist.shape)
            # negative batch dist as score for Cross-Entropy with `0` ground-truth label.
            loss = loss_ftn(
                -batch_dist,
                torch.zeros(
                    len(batch_dist), dtype=torch.long, device=batch_dist.device
                ),
            )
            # compute euclid grad
            loss.backward()
            # Riemannian SGD for Poincare disk
            optimizer.step()
            # accumulate batch loss
            total_loss += loss.item()
            # Update progress bar description
            pbar.set_description(
                f"Epoch: {epoch}\tTrain Loss: {total_loss / n_iter:.2f}"
            )
        train_loss = total_loss / n_iter

        # # for debugging
        # mask = model.embed.weight.grad.abs().sum(-1) > 0
        # print(torch.where(mask)[0])
        # print(u_v_neg_u.reshape(-1).sort()[0])
        """
