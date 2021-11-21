"""Train embeddings to hyperbolic spaces.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - M. Nickel and D. Kiela, "Poincare Embeddings for Learning Hierarchical
 Representations"
"""

import argparse

import torch.nn as nn
import wandb
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
    parser.add_argument("--wlog", action="store_true", help="Use WanDB logger")
    parser.add_argument(
        "--wlog-name", type=str, default="", help="Run ID in WanDB logger."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    wandb_run = None
    if args.wlog:
        wandb_run = wandb.init(
            project="hyperbolic-embedding", name=args.wlog_name, config=vars(args)
        )
        assert isinstance(
            wandb_run, wandb.sdk.wandb_run.Run
        ), "Failed initializing WanDB"

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
        wandb_run=wandb_run,
    )

    trainer.train()
