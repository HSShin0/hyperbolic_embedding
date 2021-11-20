"""Embedding models.

 - Author: Hyung-Seok Shin
 - Contact: shin.hyungseok@gmail.com

Ref:
 - https://github.com/facebookresearch/poincare-embeddings
"""


import torch
import torch.nn as nn


class BaseManifold(nn.Module):
    def __init__(self):
        super().__init__()


class Euclidean(BaseManifold):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class Poincare(BaseManifold):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, eps: float = 1e-5
    ) -> None:
        super().__init__()
        # TODO: initialize near zero (then use burn-in)
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Embedding weight from Uniform(-0.001, 0.001)."""
        self.embed.weight.data = 2 * torch.rand_like(self.embed.weight.data)
        self.embed.weight.data -= 1
        self.embed.weight.data *= 0.001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class Lorentz(BaseManifold):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


if __name__ == "__main__":
    model = Euclidean(5, 2)
    # model = Poincare(5, 2)
    print(model.embed.weight)
    print(model(torch.LongTensor([0, 3])))
