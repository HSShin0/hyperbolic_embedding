"""Riemannian SGD Optimizers.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - M. Nickel and D. Kiela, "Poincare Embeddings for Learning Hierarchical
 Representations"
 - http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
"""

from typing import Any

import torch

#DEBUG = True
DEBUG = False


class PoincareSGD(torch.optim.Optimizer):
    """SGD Optimizer on Poincare disk."""

    # TODO: find typing hint for params
    def __init__(self, params: Any, lr: float = 1e-3, eps: float = 1e-5) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= eps < 1e-1:
            raise ValueError(f"Invalid epsilon: {eps} - should be in [0, 0.1)")
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                lr = group["lr"]
                eps = group["eps"]
                if DEBUG:
                    if (
                        torch.any(p.grad == torch.inf)
                        or torch.any(p.grad == -torch.inf)
                        or torch.any(p.grad == torch.nan)
                    ):
                        import pdb;pdb.set_trace()

                # euclid grad > riemannian grad
                riem_grad = ((1 - (p ** 2).sum(dim=-1, keepdim=True)) ** 2) * p.grad / 4

                # retraction (linear approx of exp_p(- lr * riem_grad))
                p.data = p.data - lr * riem_grad
                # projection
                p_norm = torch.sqrt((p.data ** 2).sum(-1))
                proj_mask = p_norm > 1
                p.data[proj_mask] = p.data[proj_mask] / (
                    p_norm[proj_mask].unsqueeze(1) + eps
                )