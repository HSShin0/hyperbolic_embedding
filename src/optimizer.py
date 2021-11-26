"""Riemannian SGD Optimizers.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - M. Nickel and D. Kiela, "Poincare Embeddings for Learning Hierarchical
 Representations"
 - http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
"""

from typing import Any, Optional

import torch

# DEBUG = True
DEBUG = False
NORM_THRESHOLD = 1000.0


class PoincareSGD(torch.optim.Optimizer):
    """SGD Optimizer on Poincare disk."""

    # TODO: find typing hint for params
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        eps: float = 1e-5,
        burn_in_lr_ratio: float = 1e-1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= eps < 1e-1:
            raise ValueError(f"Invalid epsilon: {eps} - should be in [0, 0.1)")
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)
        self.burn_in_lr_ratio = burn_in_lr_ratio
        self.burn_in_state = False

    def step(self, return_l2_norm: bool = False) -> Optional[torch.Tensor]:
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if DEBUG:
                    if torch.any(torch.isinf(p.grad)) or torch.any(torch.isnan(p.grad)):
                        import pdb;pdb.set_trace()
                # euclid grad > riemannian grad
                riem_grad = ((1 - (p ** 2).sum(dim=-1, keepdim=True)) ** 2) * p.grad / 4
                # naive method for preventing too large gradient vector
                # TODO: revisit here
                grad_l2 = (riem_grad ** 2).sum(1).sqrt()
                mask = grad_l2 > NORM_THRESHOLD
                if DEBUG and torch.any(mask):
                    import pdb;pdb.set_trace()
                riem_grad[mask] /= grad_l2[mask].unsqueeze(1)

                # retraction (linear approx of exp_p(- lr * riem_grad))
                p.data = p.data - lr * riem_grad
                # projection
                p_norm = torch.sqrt((p.data ** 2).sum(-1))
                proj_mask = p_norm > 1
                p.data[proj_mask] = p.data[proj_mask] / (
                    p_norm[proj_mask].unsqueeze(1) + eps
                )
        if self.burn_in_state:
            self._escape_burn_in()

    def burn_in(self) -> None:
        for group in self.param_groups:
            group["lr"] *= self.burn_in_lr_ratio
            if DEBUG:
                print(f'lr: {group["lr"]}')
        self.burn_in_state = True

    def _escape_burn_in(self) -> None:
        if not self.burn_in_state:
            return
        for group in self.param_groups:
            group["lr"] /= self.burn_in_lr_ratio
            if DEBUG:
                print(f'lr: {group["lr"]}')
        self.burn_in_state = False
