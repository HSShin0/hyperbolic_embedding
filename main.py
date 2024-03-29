"""Train embeddings to hyperbolic spaces.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - M. Nickel and D. Kiela, "Poincare Embeddings for Learning Hierarchical
 Representations"
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
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
    parser.add_argument(
        "--wlog", action="store_true", default=False, help="Use WanDB logger"
    )
    parser.add_argument(
        "--wlog-name", type=str, default="", help="Run ID in WanDB logger."
    )
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Use GPU")
    parser.add_argument(
        "--init-lr", "-lr", type=float, default=0.01, help="Initial learning rate"
    )
    parser.add_argument(
        "--burn-in-epochs",
        "-be",
        type=int,
        default=10,
        help="Number of `burn-in` epochs. Use smaller learning rate.",
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
        args.exp_root = args.exp_root.rstrip(os.path.sep)
        # TODO: check `wlog_name` is available for using dirname
        args.exp_root = os.path.join(os.path.dirname(args.exp_root), args.wlog_name)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = TaxonomiesDataset(args.datapath, args.n_neg)
    # TODO: figure out the reason:
    # setting `num_workers` > 0 makes it slow
    # See: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/23
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=0)

    # Prepare Poincare Embedding model
    model = Poincare(dataset.n_words, args.emb_dim)
    distance = PoincareDistance()
    loss_ftn = nn.CrossEntropyLoss()

    optimizer = PoincareSGD(
        model.parameters(), lr=args.init_lr, eps=1e-5, burn_in_lr_ratio=0.1
    )
    trainer = Trainer(
        model=model,
        distance=distance,
        loss_ftn=loss_ftn,
        optimizer=optimizer,
        dataloader=dataloader,
        config=vars(args),
        wandb_run=wandb_run,
        device=device,
        burn_in_epochs=args.burn_in_epochs,
    )
    trainer.train()
