import torch
from torch.autograd import Function

from pointops._C import farthest_point_sampling_cuda


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: coords: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.zeros(new_offset[b - 1].item(), device='cuda', dtype=torch.int32)
        tmp = torch.full((n,), 1e10, device='cuda')
        farthest_point_sampling_cuda(b, n_max, xyz, offset.int(), new_offset.int(), tmp, idx)
        del tmp
        return idx


farthest_point_sampling = FarthestPointSampling.apply
