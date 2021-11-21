"""Distance module.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - https://github.com/facebookresearch/poincare-embeddings
"""
from typing import Tuple

import torch

DEBUG = False


class PoincareDistance(torch.autograd.Function):
    """Distance in Poiincare disk."""

    # TODO: see implementation of `Function` for `ctx`
    @staticmethod
    def forward(ctx, u: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute distance between u and v in Poincare disk.

        Args:
            u, v: shape (N, d) where `d` is the embedding dimension.
        Return:
            d(u, v) = arccosh(1 + 2 * {|u-v|**2} / {(1 - |u|**2)(1 - |v|**2)})
        """
        # Save eps
        ctx.eps = eps
        # NOTE: clamp gives zero grad for the point near boundary
        # if we use the ordinary autograd
        u_sq = torch.clamp((u ** 2).sum(-1), 0, 1 - eps)  # (N,)
        alpha = 1 - u_sq
        v_sq = torch.clamp((v ** 2).sum(-1), 0, 1 - eps)  # (N,)
        beta = 1 - v_sq
        gamma = 1 + 2 * ((u - v) ** 2).sum(-1) / (alpha * beta)  # (N,)
        # Save data for backprop
        # NOTE: alpha, beta, gamma are redundant
        ctx.save_for_backward(u, v, alpha, beta, gamma)

        dist = torch.acosh(gamma)
        return dist

    @staticmethod
    def backward(ctx, grad_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Compute df/du, df/dv for given df/dz(=`grad_`) where z = d(u, v)."""
        # load saved data during forward operation
        u, v, alpha, beta, gamma = ctx.saved_tensors
        # df/du = df/dz * dz/du, where z = d(u,v)
        du = grad_.unsqueeze(1) * PoincareDistance.grad(
            u, v, alpha, beta, gamma, ctx.eps
        )
        # df/dv = df/dz * dz/dv
        dv = grad_.unsqueeze(1) * PoincareDistance.grad(
            v, u, beta, alpha, gamma, ctx.eps
        )
        # TODO: check what happen without last `None` return
        return du, dv, None

    @staticmethod
    def grad(
        theta: torch.Tensor,
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Compute partial derivative of d(theta, x) w.r.t. `theta`.
        Args:
            alpha: (1 - |theta|**2)
            beta: (1 - |x|**2)
            gamma: 1 + 2 * |theta - x|**2 / (alpha * beta)

        Return:
            partial derivative d(theta, x) w.r.t. `theta` of shape (N, d) (= theta.shape)
        """
        x_sq = -beta + 1
        c_theta = (x_sq - 2 * torch.einsum("bi,bi->b", theta, x) + 1) / alpha ** 2
        c_x = -1 / alpha
        # fixed numerical unstable c_common because of (gamma == 1)
        # caused by (theta == x) by adding `eps` without any logical reasoning.
        # TODO: any better way for this? instead of adding `eps`?
        c_common = 4 / (beta * torch.sqrt(gamma ** 2 - 1) + eps)
        dtheta = c_common.unsqueeze(1) * (
            c_theta.unsqueeze(1) * theta + c_x.unsqueeze(1) * x
        )  # (N, 2)
        if DEBUG:
            for idx, tsr in enumerate([x_sq, c_theta, c_x, c_common, dtheta]):
                if torch.any(torch.isinf(tsr)) or torch.any(torch.isnan(tsr)):
                    print(idx)
                    import pdb;pdb.set_trace()
        return dtheta


if __name__ == "__main__":
    N = 8
    D = 7
    EPS = 1e-9

    #############################################
    # # Jacobian mismatch for clamped `u`, `v`
    # u = torch.randn(N, D, dtype=torch.double)
    # v = torch.randn(N, D, dtype=torch.double)
    #############################################
    u = torch.randn(N, D, dtype=torch.double) / 1e2
    v = torch.randn(N, D, dtype=torch.double) / 1e2
    u = torch.clamp(u, -1 + EPS, 1 - EPS)
    v = torch.clamp(v, -1 + EPS, 1 - EPS)
    u.requires_grad = True
    v.requires_grad = True

    # Test backward implementation
    distance = PoincareDistance()
    torch.autograd.gradcheck(distance.apply, (u, v, EPS))

    # Simulate numerical unstable setting
    v_ = u.detach().clone()
    torch.autograd.gradcheck(distance.apply, (u, v_, EPS))
