from typing import Tuple, Union

import einops
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class MMCRLoss(Module):
    r"""
    The Maximum Manifold Capacity Representations loss.
    Given a set of input datapoints :math: `Z_i \in \mathbb{R}^{k \times d}` (where :math: `N` is the number of datapoints, :math: `k` is the number of augmented views per sample, and :math: `d` is the feature dimension), calculates MMCR Loss that maximize the extent of centroid manifold.

    .. math::
        \mathcal{L} = \lambda (\frac{\sum_{i=1}^N||Z_i||_{*}}{N}) - ||\mathbf{C}||_{*}

    Args:
        lmbda (float, optional): trade-off parameter. By default, it sets to 0.
        n_aug (int, optional): number of augmented views per sample. Required when your input Tensor is 2-dimensional.
    """

    def __init__(self, lmbda: float = 0.0, n_aug: int = None) -> None:
        super(MMCRLoss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug

    def forward(self, z: Union[Tensor, list]) -> Tensor:
        r"""
        Shape:
            z: :math: `(N, k, d)` or :math: `(N\times k, d)` where :math: `k` means the number of augmented views and :math: `d` means the feature dimension. :math: `k` can be a fixed number or a variable.
        """
        batch_size = len(z)
        local_nuc = torch.tensor(0.0)

        if isinstance(z, Tensor) and len(z.shape) == 2:
            assert (
                self.n_aug is not None
            ), "You must specify `n_aug` or reshape your tensor into 3 dimension like (B, n_aug, C)."
            z = einops.rearrange(z, "(B N) C -> B N C", N=self.n_aug)

        if isinstance(z, Tensor):
            z = F.normalize(z, dim=-1)
            z_local = einops.rearrange(z, "B N C -> B C N")

            centroids = torch.mean(z_local, dim=-1)
            if self.lmbda != 0.0:
                local_nuc = torch.linalg.svdvals(z_local).sum()
            global_nuc = torch.linalg.svdvals(centroids).sum()

            loss = self.lmbda * (local_nuc / batch_size) - global_nuc

        else:  # Variable n_aug
            z_local = [F.normalize(z_, dim=-1) for z_ in z]
            centroids = [torch.mean(z_, dim=0) for z_ in z_local]
            centroids = torch.stack(centroids)

            if self.lmbda != 0.0:
                local_nuc = torch.sum(
                    [torch.linalg.svdvals(z_local_) for z_local_ in z_local]
                )
            global_nuc = torch.linalg.svdvals(centroids).sum()

            loss = self.lmbda * (local_nuc / batch_size) - global_nuc

        return loss


if __name__ == "__main__":
    batch_size = 4
    n_aug = 8
    n_channel = 32

    # MMCR_loss without n_aug specification
    mmcr_loss = MMCRLoss()
    A = torch.randn((batch_size, n_aug, n_channel))
    print(mmcr_loss(A))

    # MMCR_loss with n_aug specification
    mmcr_loss = MMCRLoss(n_aug=n_aug)
    B = torch.randn((batch_size * n_aug, n_channel))
    print(mmcr_loss(B))

    # MMCR_loss with variable n_aug
    mmcr_loss = MMCRLoss()
    C = [
        torch.randn((2, n_channel)),
        torch.randn((4, n_channel)),
        torch.randn((8, n_channel)),
    ]
    print(mmcr_loss(C))
